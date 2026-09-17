"""
trajectory_extractor.py — The core instrumentation: given a prompt and
a target answer (single OR multi-token), extract the FULL layer-wise
trajectory in a SINGLE forward pass.

v3 CHANGE FROM v1/v2:
The forward-pass-building and lens-projection logic is now split into
reusable helper functions (`build_teacher_forced_input`,
`project_trajectory_from_hidden_states`) so that Stage 4's activation
patching code (src/causal/activation_patching.py) can compute a
trajectory from a PATCHED forward pass using the EXACT SAME logic as
a normal forward pass — previously that file duplicated a simplified,
single-token version of this logic, which could silently drift out of
sync and only ever looked at the last token position. Now both files
call the same functions, so "before" and "after" trajectories in the
causal experiment are guaranteed to be computed identically.

v2 CHANGE (kept from before):
The original implementation only tracked the trajectory of ONE token
(in practice, often just the first word of the answer via
`.split()[0]`). This silently produced misleading trajectories for any
multi-token answer ("Marie Curie", "the photoelectric effect", "World
War Two"). This version teacher-forces the FULL answer span in one
forward pass and computes a JOINT trajectory across all answer tokens:
    P_l(answer)  = geometric mean of per-token probabilities at layer l
    H_l          = mean per-token entropy across the answer span
    R_l          = MAX per-token rank across the answer span
    M_l          = mean MLP write-norm across the answer's token positions
Still exactly ONE forward pass — the efficiency claim is unchanged.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from src.utils import get_logger
from src.instrumentation.tuned_lens import TunedLens

log = get_logger("instrumentation.trajectory")


@dataclass
class LayerTrajectory:
    """Container for one query's full layer-wise trajectory."""
    prompt: str
    answer_text: str
    answer_token_ids: List[int]
    n_layers: int
    n_answer_tokens: int = 1

    prob_trajectory: List[float] = field(default_factory=list)      # P_l(answer), joint geo-mean
    entropy_trajectory: List[float] = field(default_factory=list)   # H_l, mean across answer tokens
    rank_trajectory: List[int] = field(default_factory=list)        # R_l, MAX rank across answer tokens
    mlp_norm_trajectory: List[float] = field(default_factory=list)  # M_l, mean across answer positions

    per_token_prob_trajectories: List[List[float]] = field(default_factory=list)
    top10_tokens_per_layer: List[List[int]] = field(default_factory=list)


class MLPNormCapture:
    """
    Registers forward hooks on every transformer block's MLP sublayer
    to capture the L2 norm of its output AT A SET OF SPECIFIED TOKEN
    POSITIONS — the "write magnitude" signal the MLP-spike-alignment
    feature depends on.

    Usage:
        capture = MLPNormCapture(model, structure, token_positions=[10, 11, 12])
        capture.attach()
        ... run forward pass ...
        norms = capture.get_norms()
        capture.detach()
    """

    def __init__(self, model, structure: dict, token_positions: List[int] = None):
        self.model = model
        self.layers = structure["layers"]
        self.n_layers = structure["n_layers"]
        self.hooks = []
        self.norms_per_layer = {}
        self.token_positions = token_positions if token_positions is not None else [-1]

    def _make_hook(self, layer_idx: int):
        def hook(module, input, output):
            selected = output[:, self.token_positions, :]
            norms_per_position = selected.norm(dim=-1).mean(dim=0)
            self.norms_per_layer[layer_idx] = norms_per_position.mean().item()
        return hook

    def attach(self):
        self.norms_per_layer = {}
        for i, block in enumerate(self.layers):
            h = block.mlp.register_forward_hook(self._make_hook(i))
            self.hooks.append(h)

    def detach(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def get_norms(self) -> List[float]:
        return [self.norms_per_layer.get(i, 0.0) for i in range(self.n_layers)]


def tokenize_answer(tokenizer, answer_text: str) -> List[int]:
    """Tokenize an answer span with a leading space, matching how the
    model would naturally continue a prompt like '...\\nA: {answer}'."""
    answer_text = answer_text.strip()
    if not answer_text:
        return []
    return tokenizer.encode(" " + answer_text, add_special_tokens=False)


def build_teacher_forced_input(
    tokenizer, prompt: str, answer_token_ids: List[int], device
) -> Tuple[torch.Tensor, List[int]]:
    """
    Shared helper: build the (prompt + answer) input tensor and the list
    of token positions whose hidden state PREDICTS each answer token.

    Used identically by extract_trajectory() (normal forward pass) and
    by activation_patching.py (patched forward pass), so both compute
    trajectories over exactly the same positions with exactly the same
    conventions.

    Returns:
        (input_ids [1, seq_len], predict_positions — one index per answer token)
    """
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
    full_ids = prompt_ids + answer_token_ids
    input_ids = torch.tensor([full_ids], device=device)

    n_prompt_tokens = len(prompt_ids)
    n_answer_tokens = len(answer_token_ids)
    predict_positions = list(range(n_prompt_tokens - 1, n_prompt_tokens - 1 + n_answer_tokens))

    return input_ids, predict_positions


def project_trajectory_from_hidden_states(
    hidden_states,
    lens: TunedLens,
    structure: dict,
    predict_positions: List[int],
    answer_token_ids: List[int],
    top_k: int = 10,
) -> Tuple[List[float], List[float], List[int], List[List[int]], List[List[float]]]:
    """
    Shared helper: given a forward pass's hidden_states (from either a
    NORMAL or a PATCHED forward pass — this function doesn't care which),
    project every layer through the tuned lens at the answer-predicting
    positions and compute the joint per-layer signal sequences.

    This is the single source of truth for "how do we turn hidden states
    into a trajectory" — both extract_trajectory() and the activation
    patching code call this, so a patched trajectory is computed with
    identical logic to a normal one.

    v2 MEMORY FIX: this function is ONLY ever used for inference/feature
    extraction (never inside a training loop — tuned-lens training has
    its own separate loop in tuned_lens.py that calls lens.project()
    directly). It is therefore safe, and now explicit, to wrap the core
    per-layer loop in torch.no_grad() — previously this ran with
    gradient tracking enabled by default (the surrounding no_grad()
    block in the caller had already closed before this function was
    invoked), needlessly building and retaining an autograd graph for
    32 layers' worth of large logits tensors on every single trajectory
    extraction call, across every example in Stage 2's dataset and
    every patched forward pass in Stage 4 — for zero benefit, since
    nothing here ever calls .backward().

    Returns:
        (prob_trajectory, entropy_trajectory, rank_trajectory,
         top10_tokens_per_layer, per_token_prob_trajectories)
    """
    n_layers = structure["n_layers"]
    final_ln = structure["final_ln"]
    unembed  = structure["unembed"]
    device = hidden_states[0].device

    n_answer_tokens = len(answer_token_ids)
    prob_trajectory, entropy_trajectory, rank_trajectory = [], [], []
    top10_tokens_per_layer = []
    per_token_probs = [[] for _ in range(n_answer_tokens)]

    positions_tensor = torch.tensor(predict_positions, device=device)

    with torch.no_grad():
        for l in range(1, n_layers + 1):
            h_l = hidden_states[l][:, positions_tensor, :].float()  # [1, n_answer_tokens, hidden]
            logits_l = lens.project(h_l, l - 1, final_ln, unembed).squeeze(0)  # [n_answer_tokens, vocab]
            probs_l = F.softmax(logits_l, dim=-1)

            layer_token_probs, layer_token_ranks, layer_token_entropies = [], [], []

            for t_idx, token_id in enumerate(answer_token_ids):
                p_t = probs_l[t_idx, token_id].item()
                layer_token_probs.append(max(p_t, 1e-12))
                per_token_probs[t_idx].append(p_t)

                entropy_t = -(probs_l[t_idx] * torch.log(probs_l[t_idx] + 1e-12)).sum().item()
                layer_token_entropies.append(entropy_t)

                rank_t = (probs_l[t_idx] > probs_l[t_idx, token_id]).sum().item()
                layer_token_ranks.append(rank_t)

                if t_idx == 0:
                    topk_ids = torch.topk(probs_l[t_idx], k=top_k).indices.tolist()
                    top10_tokens_per_layer.append(topk_ids)

            # Joint probability across the answer span via geometric mean —
            # avoids the joint probability trivially vanishing for longer answers.
            log_probs = [torch.log(torch.tensor(p)).item() for p in layer_token_probs]
            geo_mean_prob = float(torch.exp(torch.tensor(sum(log_probs) / len(log_probs))))
            prob_trajectory.append(geo_mean_prob)

            entropy_trajectory.append(sum(layer_token_entropies) / len(layer_token_entropies))
            rank_trajectory.append(max(layer_token_ranks))  # worst-case token drives the rank signal

    return prob_trajectory, entropy_trajectory, rank_trajectory, top10_tokens_per_layer, per_token_probs


def extract_trajectory(
    model,
    tokenizer,
    lens: TunedLens,
    structure: dict,
    prompt: str,
    answer_text: Optional[str] = None,
    top_k: int = 10,
    max_answer_tokens: int = 8,
) -> LayerTrajectory:
    """
    Run ONE forward pass over (prompt + answer) and extract the complete
    layer-wise JOINT trajectory across every token of the answer.

    Args:
        model:        loaded Phi-2 (from model_loader.load_phi2)
        tokenizer:    Phi-2 tokenizer
        lens:         trained TunedLens (from tuned_lens.load_tuned_lens)
        structure:    dict from model_loader.get_model_structure()
        prompt:       the question/prompt text, e.g. "Q: The capital of France is\\nA:"
        answer_text:  the FULL answer text to track. If None, the model's own
                      greedy continuation is generated first and used.
        top_k:        how many top tokens to record per layer (early-presence feature)
        max_answer_tokens: cap on tracked answer tokens (bounds cost)

    Returns:
        LayerTrajectory aggregated across the full answer span.
    """
    device = next(model.parameters()).device
    n_layers = structure["n_layers"]

    if answer_text is None:
        prompt_inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            gen_out = model.generate(
                **prompt_inputs, max_new_tokens=max_answer_tokens,
                do_sample=False, pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = gen_out[0][prompt_inputs["input_ids"].shape[1]:]
        answer_text = tokenizer.decode(new_tokens, skip_special_tokens=True).split("\n")[0].strip()

    answer_token_ids = tokenize_answer(tokenizer, answer_text)[:max_answer_tokens]
    if not answer_token_ids:
        answer_token_ids = [tokenizer.eos_token_id]

    input_ids, predict_positions = build_teacher_forced_input(tokenizer, prompt, answer_token_ids, device)
    n_answer_tokens = len(answer_token_ids)

    mlp_capture = MLPNormCapture(model, structure, token_positions=predict_positions)
    mlp_capture.attach()
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True)
    mlp_capture.detach()
    mlp_norms = mlp_capture.get_norms()

    prob_traj, entropy_traj, rank_traj, top10, per_token_probs = project_trajectory_from_hidden_states(
        out.hidden_states, lens, structure, predict_positions, answer_token_ids, top_k
    )

    traj = LayerTrajectory(
        prompt=prompt,
        answer_text=answer_text,
        answer_token_ids=answer_token_ids,
        n_layers=n_layers,
        n_answer_tokens=n_answer_tokens,
    )
    traj.mlp_norm_trajectory = mlp_norms
    traj.prob_trajectory = prob_traj
    traj.entropy_trajectory = entropy_traj
    traj.rank_trajectory = rank_traj
    traj.top10_tokens_per_layer = top10
    traj.per_token_prob_trajectories = per_token_probs

    return traj


def extract_trajectory_batch(
    model, tokenizer, lens, structure,
    prompts: List[str], answers: List[Optional[str]] = None,
) -> List[LayerTrajectory]:
    """Convenience wrapper to extract trajectories for many prompts sequentially."""
    if answers is None:
        answers = [None] * len(prompts)
    trajectories = []
    for i, (p, a) in enumerate(zip(prompts, answers)):
        traj = extract_trajectory(model, tokenizer, lens, structure, p, a)
        trajectories.append(traj)
        if i % 50 == 0:
            log.info(f"Extracted {i}/{len(prompts)} trajectories")
    return trajectories
