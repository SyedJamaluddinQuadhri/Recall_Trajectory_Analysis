"""
run_stage2_build_dataset.py — Stage 2: Build the labeled factual QA
dataset AND extract trajectories for every labeled example.

Runtime: ~4-6 hours (generation + trajectory extraction combined)
Output:  data/processed/labeled_qa.json
         data/trajectories/all_trajectories.pkl
         data/trajectories/baseline_norms.json

Usage:
    python scripts/run_stage2_build_dataset.py
    python scripts/run_stage2_build_dataset.py --samples 500   # quick test
"""

import argparse
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import get_logger, get_project_root, save_json, save_pickle, load_json
from src.instrumentation.model_loader import load_phi2, get_model_structure
from src.instrumentation.tuned_lens import load_tuned_lens
from src.instrumentation.trajectory_extractor import extract_trajectory
from src.dataset.build_factual_qa import build_labeled_dataset
from src.features.shape_descriptors import compute_baseline_norms

log = get_logger("script.stage2")


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Build labeled dataset + trajectories")
    parser.add_argument("--samples", type=int, default=3000, help="Samples per source dataset")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("RTA Stage 2 — Ground Truth Dataset & Trajectory Extraction")
    log.info("=" * 60)

    root = get_project_root()
    labeled_path = str(root / "data" / "processed" / "labeled_qa.json")
    traj_path = str(root / "data" / "trajectories" / "all_trajectories.pkl")
    baseline_path = str(root / "data" / "trajectories" / "baseline_norms.json")

    # ── Step 1: Build labeled dataset (or load if cached) ────────────────────
    if os.path.exists(labeled_path):
        log.info("Loading cached labeled dataset...")
        labeled = load_json(labeled_path)
    else:
        labeled = build_labeled_dataset(
            max_samples_per_source=args.samples,
            save_path=labeled_path,
        )

    # ── Step 2: Extract trajectory for every labeled example ─────────────────
    log.info(f"Extracting trajectories for {len(labeled)} examples...")
    model, tokenizer = load_phi2()
    structure = get_model_structure(model)
    lens = load_tuned_lens(structure["hidden_dim"], structure["n_layers"])

    trajectories = []
    labels = []
    mlp_norm_sequences = []

    for i, item in enumerate(labeled):
        prompt = f"Q: {item['question']}\nA:"
        # v2 FIX: track the trajectory of the model's FULL answer span,
        # not just its first word. The original `.split()[0]` truncation
        # silently tracked an uninformative token (e.g. "the", "in") for
        # any multi-token answer like "Marie Curie" or "World War Two" —
        # extract_trajectory() now teacher-forces the entire answer and
        # computes a joint trajectory across all of its tokens.
        traj = extract_trajectory(
            model, tokenizer, lens, structure, prompt,
            answer_text=item["model_answer"] if item["model_answer"] else None,
        )
        trajectories.append(traj)
        labels.append(1 if item["label"] == "confabulated" else 0)
        mlp_norm_sequences.append(traj.mlp_norm_trajectory)

        if i % 100 == 0:
            log.info(f"Extracted {i}/{len(labeled)} trajectories")

    # ── Step 3: Compute baseline MLP norms (needed for MLP-alignment feature) ─
    log.info("Computing baseline MLP norms across all examples...")
    baseline_norms = compute_baseline_norms(mlp_norm_sequences)
    save_json(baseline_norms, baseline_path)

    # ── Step 4: Save everything ────────────────────────────────────────────────
    save_pickle({"trajectories": trajectories, "labels": labels}, traj_path)
    log.info(f"Saved {len(trajectories)} trajectories to {traj_path}")

    log.info("\n✓ Stage 2 complete!")
    log.info("Next step: python scripts/run_stage3_train_classifier.py")


if __name__ == "__main__":
    main()
