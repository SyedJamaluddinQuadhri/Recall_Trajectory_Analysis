"""
tuned_lens.py — Train and apply per-layer "tuned lens" probes.

WHAT THIS SOLVES:
Every transformer layer's hidden state (residual stream) is a big vector
of numbers that isn't directly readable as "what word does the model
think comes next." Only the FINAL layer's state is designed to be
projected into vocabulary space (through final_layernorm + lm_head).

The raw "logit lens" trick (projecting an EARLY layer's state through
the FINAL layer's projection matrices) is noisy because early layers
represent information in a different geometry than the final layer.

The "tuned lens" (Belrose et al. 2023) fixes this: we train one cheap
transform per layer whose job is to translate that layer's
representation into something that, when projected through the final
unembedding, gives a sensible probability distribution. This is trained
ONCE per model on generic text, and then reused for every experiment.

Reference: Belrose et al., "Eliciting Latent Predictions from Transformers
with the Tuned Lens" (2023).

v2 MEMORY FIX (found during a deep audit — this is important enough
that it could otherwise block Stage 1 from running at all):
The original design used a FULL nn.Linear(hidden_dim, hidden_dim) per
layer. For Phi-2 (hidden_dim=2560, 32 layers) that is ~210M parameters
for the lens alone — 0.84GB of fp32 weights, plus another 1.68GB for
Adam's optimizer states, totalling 2.52GB BEFORE the base model (2.1GB
in 4-bit) or any activations are even counted. Combined, that already
exceeds the project's stated 4GB VRAM budget with zero room for
anything else — Stage 1 would very likely OOM on an RTX 2050 as
originally designed.

This version uses LOW-RANK translators instead (a down-projection to a
small rank, then an up-projection back to hidden_dim, LoRA-style, with
a residual/identity bypass). This cuts the lens to ~21M total
parameters (a 10x reduction) — about 0.25GB combined weights+optimizer
— bringing the whole pipeline comfortably under 4GB. The near-identity
initialization property (start at raw logit lens, improve from there
via training) is preserved by zero-initializing the up-projection, so
the translator is an exact no-op before any training happens, exactly
as the original full-rank identity-init design intended.

Training also now calls backward() ONCE PER LAYER inside the loop
(instead of summing all 32 layers' losses into one graph before a
single backward() call). Since each layer's translator has disjoint
parameters from every other layer's, this produces mathematically
IDENTICAL gradients to the original approach, but never holds more
than one layer's logits tensor in memory at a time — avoiding an
additional ~3.3GB of peak activation memory that the original
accumulate-then-backward-once pattern required.
"""

import os
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import get_logger, get_project_root, free_gpu_memory

log = get_logger("instrumentation.tuned_lens")

LENS_SAVE_PATH = str(get_project_root() / "models" / "tuned_lens_phi2.pt")

DEFAULT_RANK = 128  # low-rank bottleneck dimension for each layer's translator


class LowRankTranslator(nn.Module):
    """
    One layer's tuned-lens translator, parameterized as a low-rank
    correction on top of an identity bypass:

        output = x + up(down(x))

    `up`'s weight AND bias are zero-initialized, so at initialization
    this is an EXACT no-op (output == input) — mirroring the original
    full-rank design's identity initialization, just achieved via a
    residual bypass instead of an identity weight matrix (which isn't
    available here since down/up are rectangular, not square).

    Parameter count: 2 * hidden_dim * rank + hidden_dim
    (vs. hidden_dim^2 + hidden_dim for a full nn.Linear — a ~10x
    reduction at rank=128, hidden_dim=2560).
    """

    def __init__(self, hidden_dim: int, rank: int = DEFAULT_RANK):
        super().__init__()
        self.down = nn.Linear(hidden_dim, rank, bias=False)
        self.up = nn.Linear(rank, hidden_dim, bias=True)

        nn.init.normal_(self.down.weight, std=0.02)  # arbitrary — irrelevant while up=0
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.down(x))


class TunedLens(nn.Module):
    """
    One learned low-rank translator per transformer layer.

    For layer l, translator[l] maps that layer's residual stream into
    a representation that the model's OWN final layernorm + unembedding
    can meaningfully decode into a vocabulary distribution.

    Total size for Phi-2 (32 layers, hidden_dim=2560, rank=128):
    ~21M parameters (~0.08GB fp32 weights, ~0.25GB including Adam
    optimizer states during training) — see the module docstring above
    for the full memory analysis and why this replaced the original
    full-rank design.
    """

    def __init__(self, hidden_dim: int, n_layers: int, rank: int = DEFAULT_RANK):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.translators = nn.ModuleList([
            LowRankTranslator(hidden_dim, rank) for _ in range(n_layers)
        ])

    def project(
        self,
        hidden_state: torch.Tensor,
        layer_idx: int,
        final_ln: nn.Module,
        unembed: nn.Module,
    ) -> torch.Tensor:
        """
        Project a single layer's hidden state into vocabulary logits.

        Args:
            hidden_state: [batch, seq, hidden_dim] — this layer's residual stream
            layer_idx:    which translator to use
            final_ln:     the model's own final layer norm module
            unembed:      the model's own unembedding (lm_head) module

        Returns:
            logits: [batch, seq, vocab_size]
        """
        translated = self.translators[layer_idx](hidden_state)
        ln_dtype = next(final_ln.parameters()).dtype
        normed = final_ln(translated.to(ln_dtype))
        logits = unembed(normed)
        return logits.float()


def train_tuned_lens(
    model,
    tokenizer,
    structure: dict,
    corpus_texts: List[str],
    n_epochs: int = 2,
    batch_size: int = 4,
    lr: float = 1e-4,
    max_length: int = 128,
    rank: int = DEFAULT_RANK,
    save_path: str = LENS_SAVE_PATH,
) -> TunedLens:
    """
    Train the tuned lens translators on a generic text corpus.

    Training objective: for each layer l, minimize the KL divergence
    between the translator's projected distribution at layer l and the
    TRUE final-layer output distribution (detached, used as target).

    v2 MEMORY FIX: calls loss.backward() ONCE PER LAYER inside the loop,
    rather than accumulating all 32 layers' losses into a single graph
    before one backward() call. Because each layer's translator has
    entirely disjoint parameters from every other layer's (and the base
    model + final_ln/unembed are frozen — see model_loader.load_phi2()),
    this produces mathematically IDENTICAL accumulated gradients to the
    original all-at-once approach, while never holding more than one
    layer's projected logits tensor in memory at a time. See the module
    docstring for the concrete VRAM numbers this saves.

    Args:
        model:        loaded Phi-2 model (from model_loader.load_phi2) —
                      should already have all parameters frozen.
        tokenizer:    Phi-2 tokenizer
        structure:    dict from get_model_structure()
        corpus_texts: list of generic sentences (e.g. Wikipedia sample)
        n_epochs:     training epochs over the corpus
        batch_size:   batch size (keep small — Phi-2 forward pass dominates memory)
        lr:           learning rate for the translators
        max_length:   max token length per training example
        rank:         low-rank bottleneck dimension (see LowRankTranslator)
        save_path:    where to save the trained lens

    Returns:
        Trained TunedLens module
    """
    device = next(model.parameters()).device
    n_layers  = structure["n_layers"]
    hidden_dim = structure["hidden_dim"]
    final_ln  = structure["final_ln"]
    unembed   = structure["unembed"]

    lens = TunedLens(hidden_dim, n_layers, rank=rank).to(device).float()
    optimizer = torch.optim.Adam(lens.parameters(), lr=lr)

    lens_params = sum(p.numel() for p in lens.parameters())
    log.info(f"Training tuned lens: {n_layers} layers, rank={rank}, "
              f"{lens_params:,} total params, {len(corpus_texts)} sentences")

    for epoch in range(n_epochs):
        total_loss = 0.0
        n_batches = 0

        for i in range(0, len(corpus_texts), batch_size):
            batch_texts = corpus_texts[i:i + batch_size]
            enc = tokenizer(
                batch_texts, return_tensors="pt", truncation=True,
                max_length=max_length, padding=True,
            ).to(device)

            with torch.no_grad():
                out = model(**enc, output_hidden_states=True)
                hidden_states = out.hidden_states  # tuple of (n_layers+1) tensors
                true_logits = out.logits.detach()
                true_log_probs = F.log_softmax(true_logits, dim=-1)

            optimizer.zero_grad()
            batch_loss_value = 0.0

            # v2 FIX: backward() per layer, not accumulated across all 32.
            # Skip layer 0 (raw embeddings) — start from layer 1.
            for l in range(1, n_layers + 1):
                h_l = hidden_states[l].float()
                pred_logits = lens.project(h_l, l - 1, final_ln, unembed)
                pred_log_probs = F.log_softmax(pred_logits, dim=-1)

                kl = F.kl_div(
                    pred_log_probs, true_log_probs,
                    log_target=True, reduction="batchmean"
                ) / n_layers  # divide here so the effective step size matches
                              # "average KL across layers," same as the original

                kl.backward()  # frees this layer's graph immediately after
                batch_loss_value += kl.item()

            optimizer.step()  # one step per batch, using all 32 layers' accumulated grads

            total_loss += batch_loss_value
            n_batches += 1

            if n_batches % 20 == 0:
                log.info(f"Epoch {epoch} batch {n_batches}: avg_loss={total_loss/n_batches:.4f}")

        log.info(f"Epoch {epoch} complete. Avg loss: {total_loss/max(n_batches,1):.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({"state_dict": lens.state_dict(), "rank": rank,
                "hidden_dim": hidden_dim, "n_layers": n_layers}, save_path)
    log.info(f"Tuned lens saved to {save_path}")

    # Release optimizer state and any transient training-time memory —
    # matters if this is called from a longer-running process (e.g. a
    # notebook) rather than a one-shot CLI script that exits right after.
    del optimizer
    free_gpu_memory()

    return lens


def load_tuned_lens(
    hidden_dim: int, n_layers: int, path: str = LENS_SAVE_PATH, device: str = "cuda",
) -> TunedLens:
    """
    Load a previously trained tuned lens.

    v2 CHANGE: the saved checkpoint now stores `rank` alongside the
    state dict (see train_tuned_lens's torch.save call), so the lens
    is reconstructed with the SAME rank it was trained with, rather
    than assuming a hardcoded architecture. Falls back to DEFAULT_RANK
    for loading older checkpoints saved before this change (with a
    warning, since those old checkpoints are full-rank and this
    fallback path won't apply to them — see the except branch).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Tuned lens not found at {path}. Run scripts/run_stage1_train_lens.py first."
        )

    checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        rank = checkpoint.get("rank", DEFAULT_RANK)
        lens = TunedLens(hidden_dim, n_layers, rank=rank)
        lens.load_state_dict(checkpoint["state_dict"])
    else:
        # Backward-compatibility path for checkpoints saved by the original
        # full-rank implementation (a bare state_dict, no wrapping dict).
        log.warning(
            "Loading an OLD-format tuned lens checkpoint (full-rank, pre-v2). "
            "This will fail to load into the new low-rank TunedLens architecture. "
            "Retrain with scripts/run_stage1_train_lens.py to get a v2 checkpoint."
        )
        raise ValueError(
            "Old-format checkpoint detected and is incompatible with the current "
            "low-rank TunedLens. Please retrain: python scripts/run_stage1_train_lens.py"
        )

    lens = lens.to(device).float()
    lens.eval()
    return lens


def build_training_corpus(n_sentences: int = 2000) -> List[str]:
    """
    Build a generic training corpus for the tuned lens.
    Uses a Wikipedia sample — the lens just needs to learn general
    "how does layer l's representation map to final vocabulary,"
    it does not need to be factual QA specific.
    """
    from datasets import load_dataset
    log.info(f"Streaming wikitext for tuned-lens training corpus...")
    wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="train",
                        streaming=True, trust_remote_code=True)

    sentences = []
    for article in wiki:
        text = article["text"]
        chunks = text.split(". ")[:3]
        for c in chunks:
            if 20 < len(c) < 300:
                sentences.append(c.strip() + ".")
        if len(sentences) >= n_sentences:
            break

    log.info(f"Built training corpus: {len(sentences)} sentences")
    return sentences[:n_sentences]
