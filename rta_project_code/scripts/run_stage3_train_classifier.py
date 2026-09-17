"""
run_stage3_train_classifier.py — Stage 3: Extract the four shape
descriptors from all cached trajectories and train the classifier.

Runtime: ~10 minutes (fast — descriptors are cheap to compute, classifier is tiny)
Output:  models/trajectory_classifier/classifier.pt
         models/trajectory_classifier/scaler.pkl

Usage:
    python scripts/run_stage3_train_classifier.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.utils import get_logger, get_project_root, load_pickle, load_json
from src.features.shape_descriptors import extract_all_descriptors
from src.classifier.trajectory_classifier import (
    train_classifier,
    cross_validate_classifier,
    features_dict_to_vector,
)

log = get_logger("script.stage3")


def main():
    log.info("=" * 60)
    log.info("RTA Stage 3 — Feature Extraction & Classifier Training")
    log.info("=" * 60)

    root = get_project_root()
    traj_path = str(root / "data" / "trajectories" / "all_trajectories.pkl")
    baseline_path = str(root / "data" / "trajectories" / "baseline_norms.json")

    data = load_pickle(traj_path)
    if data is None:
        log.error("No trajectories found. Run scripts/run_stage2_build_dataset.py first.")
        return

    trajectories = data["trajectories"]
    labels = data["labels"]
    baseline_norms = load_json(baseline_path)

    log.info(f"Loaded {len(trajectories)} trajectories")
    log.info(f"Label distribution: correct={labels.count(0)}, confabulated={labels.count(1)}")

    # Extract all descriptors
    log.info("Extracting shape descriptors...")
    all_descriptors = [extract_all_descriptors(t, baseline_norms) for t in trajectories]
    X = np.array([features_dict_to_vector(d) for d in all_descriptors])
    y = np.array(labels)

    # ── Single-split evaluation (80/20 stratified, seed=42) ──────────────
    # Trains the production model and saves classifier.pt + scaler.pkl.
    # This number is retained as a secondary/sanity-check figure.
    result = train_classifier(X, y)
    ci = result["test_auroc_ci"]

    # ── 5-fold stratified CV (headline metric for the paper) ─────────────
    # Uses the same class-weighted MLP, same scaler, same seed=42.
    # Methodology now matches the ablation study in run_stage5_evaluate.py.
    log.info("")
    log.info("Running 5-fold stratified CV (headline metric)...")
    cv_result = cross_validate_classifier(X, y, n_folds=5)

    # ── Final side-by-side summary ────────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info("AUROC SUMMARY (full 6-feature model, class-weighted MLP)")
    log.info("=" * 60)
    log.info(
        f"  Single-split AUROC  (sanity check, 80/20 stratified, seed=42): "
        f"{result['test_auroc']:.4f}  "
        f"(95% CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}])"
    )
    log.info(
        f"  5-fold CV AUROC     (headline metric, StratifiedKFold, seed=42): "
        f"{cv_result['mean_auroc']:.4f} ± {cv_result['std_auroc']:.4f}  "
        f"(folds: {[f'{a:.4f}' for a in cv_result['fold_aurocs']]})"
    )
    log.info("=" * 60)
    log.info("")
    log.info("✓ Stage 3 complete!")
    log.info("Next step: python scripts/run_stage1b_validate_lens.py (if not already run),")
    log.info("           then python scripts/run_stage4_causal_patching.py")


if __name__ == "__main__":
    main()
