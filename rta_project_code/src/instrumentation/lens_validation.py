"""
lens_validation.py — Diagnostic checks for a trained TunedLens, BEFORE
trusting it for any trajectory extraction.

WHY THIS EXISTS:
The original pipeline trained the tuned lens (Stage 1) and then used it
for everything downstream with no verification step. A badly-trained
lens could silently produce garbage trajectories that still happen to
separate correct/confabulated examples for the WRONG reason — for
instance, if the lens is simply worse at decoding early layers
regardless of content, that alone could manufacture a fake
"crystallization" pattern with nothing to do with factual recall.

Two checks matter most:
  1. Final-layer sanity check: at the LAST layer, the tuned lens's
     translator should barely need to do anything (it started at
     near-identity init and the final layer's hidden state is already
     in the right basis) — so the lens's final-layer prediction should
     near-exactly match the model's own true final-layer prediction.
     If it doesn't, something is wrong with the lens implementation
     itself, independent of training quality.
  2. Monotonic improvement check: as you move from early to late layers,
     the lens's top-1 accuracy (does it predict the actual next token
     correctly) should generally IMPROVE, not fluctuate erratically or
     decrease. Some noise layer-to-layer is expected and fine; a lens
     that gets WORSE on average in later layers indicates a training
     problem.
"""

from typing import List, Dict

import torch
import torch.nn.functional as F

from src.utils import get_logger
from src.instrumentation.tuned_lens import TunedLens

log = get_logger("instrumentation.lens_validation")


def validate_tuned_lens(
    model, tokenizer, lens: TunedLens, structure: dict,
    validation_texts: List[str],
) -> Dict:
    """
    Run the two diagnostic checks above over a held-out set of generic
    sentences (should be DIFFERENT sentences from the ones used to train
    the lens, to catch overfitting too).

    Args:
        model, tokenizer: loaded Phi-2
        lens:             trained TunedLens to validate
        structure:        dict from get_model_structure()
        validation_texts: held-out generic sentences (NOT the training corpus)

    Returns:
        dict with:
          final_layer_agreement_rate: fraction of tokens where the lens's
              final-layer top-1 prediction matches the model's TRUE
              final-layer top-1 prediction (should be very close to 1.0)
          layer_accuracy: list of top-1 accuracy per layer (against the
              actual next token in the validation text) — should trend
              upward from early to late layers
          is_monotonic_enough: bool — a soft check that accuracy in the
              last third of layers is on average higher than the first
              third (allows for some non-monotonicity, which is normal)
    """
    device = next(model.parameters()).device
    n_layers = structure["n_layers"]
    final_ln = structure["final_ln"]
    unembed  = structure["unembed"]

    layer_correct = [0] * n_layers
    layer_total = [0] * n_layers
    final_agree = 0
    final_total = 0

    for text in validation_texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        if enc["input_ids"].shape[1] < 2:
            continue

        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)

        true_final_logits = out.logits[0]  # [seq, vocab] — model's real output
        hidden_states = out.hidden_states

        # Compare the lens's LAST-layer prediction against the model's TRUE output
        h_final = hidden_states[n_layers][:, :, :].float()
        lens_final_logits = lens.project(h_final, n_layers - 1, final_ln, unembed).squeeze(0)

        true_top1 = true_final_logits.argmax(dim=-1)
        lens_top1 = lens_final_logits.argmax(dim=-1)
        final_agree += (true_top1 == lens_top1).sum().item()
        final_total += true_top1.numel()

        # Layer-by-layer top-1 accuracy against the ACTUAL next token in the text
        actual_next_tokens = enc["input_ids"][0, 1:]  # what really comes next
        for l in range(1, n_layers + 1):
            h_l = hidden_states[l][:, :-1, :].float()  # predict positions 0..n-2 -> tokens 1..n-1
            logits_l = lens.project(h_l, l - 1, final_ln, unembed).squeeze(0)
            preds = logits_l.argmax(dim=-1)
            correct = (preds == actual_next_tokens).sum().item()
            layer_correct[l - 1] += correct
            layer_total[l - 1] += actual_next_tokens.numel()

    final_agreement_rate = final_agree / max(final_total, 1)
    layer_accuracy = [layer_correct[i] / max(layer_total[i], 1) for i in range(n_layers)]

    first_third = layer_accuracy[: n_layers // 3]
    last_third = layer_accuracy[-(n_layers // 3):]
    is_monotonic_enough = (sum(last_third) / len(last_third)) > (sum(first_third) / len(first_third))

    result = {
        "final_layer_agreement_rate": final_agreement_rate,
        "layer_accuracy": layer_accuracy,
        "is_monotonic_enough": bool(is_monotonic_enough),
        "first_third_avg_accuracy": float(sum(first_third) / len(first_third)),
        "last_third_avg_accuracy": float(sum(last_third) / len(last_third)),
    }

    log.info(f"Final-layer agreement rate: {final_agreement_rate:.4f} (target: > 0.95)")
    log.info(f"First-third avg accuracy:   {result['first_third_avg_accuracy']:.4f}")
    log.info(f"Last-third avg accuracy:    {result['last_third_avg_accuracy']:.4f}")
    log.info(f"Monotonic improvement:      {is_monotonic_enough} (expect True)")

    if final_agreement_rate < 0.95:
        log.warning(
            "Final-layer agreement is below 0.95 — the tuned lens's translator "
            "for the LAST layer should be nearly a no-op. This suggests a bug "
            "in the lens implementation itself (check TunedLens.project() and "
            "the identity initialization), not just insufficient training."
        )
    if not is_monotonic_enough:
        log.warning(
            "Layer accuracy does NOT clearly improve from early to late layers. "
            "This suggests the tuned lens needs more training (more epochs or "
            "more corpus sentences) before trajectories extracted with it can "
            "be trusted. Re-run Stage 1 with a larger --sentences value."
        )

    return result
