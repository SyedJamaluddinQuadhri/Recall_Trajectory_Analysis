"""
run_stage1b_validate_lens.py — Stage 1b (NEW): Validate the trained
tuned lens BEFORE trusting it for trajectory extraction.

Run this immediately after Stage 1 and BEFORE Stage 2. If the checks
here fail, go back and retrain the lens with more corpus sentences or
more epochs rather than proceeding — every downstream result depends
on the lens being reliable.

Runtime: ~5 minutes
Output:  evaluation/results/lens_validation.json

Usage:
    python scripts/run_stage1b_validate_lens.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import get_logger, get_project_root, save_json
from src.instrumentation.model_loader import load_phi2, get_model_structure
from src.instrumentation.tuned_lens import load_tuned_lens, build_training_corpus
from src.instrumentation.lens_validation import validate_tuned_lens

log = get_logger("script.stage1b")


def main():
    log.info("=" * 60)
    log.info("RTA Stage 1b — Tuned Lens Validation")
    log.info("=" * 60)

    model, tokenizer = load_phi2()
    structure = get_model_structure(model)
    lens = load_tuned_lens(structure["hidden_dim"], structure["n_layers"])

    # IMPORTANT: use DIFFERENT sentences from Stage 1's training corpus,
    # to also catch overfitting, not just underfitting.
    log.info("Building held-out validation corpus (different from training corpus)...")
    validation_texts = build_training_corpus(n_sentences=300)

    result = validate_tuned_lens(model, tokenizer, lens, structure, validation_texts)

    root = get_project_root()
    save_json(result, str(root / "evaluation" / "results" / "lens_validation.json"))

    log.info("\n" + "=" * 60)
    if result["final_layer_agreement_rate"] > 0.95 and result["is_monotonic_enough"]:
        log.info("✓ Tuned lens PASSED validation. Safe to proceed to Stage 2.")
    else:
        log.warning("✗ Tuned lens FAILED validation checks.")
        log.warning("Recommendation: re-run Stage 1 with more corpus sentences and/or")
        log.warning("more epochs, or a higher --rank if accuracy is uniformly poor across")
        log.warning("all layers (suggests the low-rank bottleneck itself is too narrow):")
        log.warning("  python scripts/run_stage1_train_lens.py --sentences 5000 --epochs 3")
        log.warning("  python scripts/run_stage1_train_lens.py --rank 256")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
