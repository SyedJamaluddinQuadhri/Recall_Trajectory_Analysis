"""
run_stage1_train_lens.py — Stage 1: Instrument Phi-2 and train the tuned lens.

Runtime: ~3 hours on RTX 2050
Output:  models/tuned_lens_phi2.pt

Usage:
    python scripts/run_stage1_train_lens.py
    python scripts/run_stage1_train_lens.py --sentences 500   # quick test run
    python scripts/run_stage1_train_lens.py --rank 256        # more expressive lens (uses more VRAM)
"""

import argparse
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import get_logger
from src.instrumentation.model_loader import load_phi2, get_model_structure
from src.instrumentation.tuned_lens import train_tuned_lens, build_training_corpus, DEFAULT_RANK

log = get_logger("script.stage1")


def main():
    parser = argparse.ArgumentParser(description="Stage 1: Train tuned lens")
    parser.add_argument("--sentences", type=int, default=2000, help="Training corpus size")
    parser.add_argument("--epochs", type=int, default=2, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--rank", type=int, default=DEFAULT_RANK,
                        help="Low-rank bottleneck dimension for each layer's translator "
                             "(default 128; ~21M total params for Phi-2, ~0.25GB with Adam "
                             "state). Increase (e.g. 256) for a more expressive lens if "
                             "Stage 1b validation shows poor accuracy, at the cost of more "
                             "VRAM — rank=256 roughly doubles lens memory to ~0.5GB.")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("RTA Stage 1 — Model Instrumentation & Tuned Lens Training")
    log.info("=" * 60)

    model, tokenizer = load_phi2()
    structure = get_model_structure(model)

    corpus = build_training_corpus(n_sentences=args.sentences)

    lens = train_tuned_lens(
        model, tokenizer, structure, corpus,
        n_epochs=args.epochs, batch_size=args.batch_size, rank=args.rank,
    )

    log.info("\n✓ Stage 1 complete!")
    log.info("Next step: python scripts/run_stage1b_validate_lens.py")


if __name__ == "__main__":
    main()
