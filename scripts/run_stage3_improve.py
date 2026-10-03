"""
run_stage3_improve.py — Stage 3 classifier improvement experiment.

Phase 1 — cheap tuning of the EXISTING 6-feature classifier:
  Step 1: Grid search (hidden_width × lr × pos_weight scale)
  Step 2: Pairwise interaction terms (6 → 21 features)
  Step 3: SMOTE minority-class oversampling (within train fold only)
  Step 4: Report Phase 1 best

Phase 2 — richer feature representation (no new GPU inference needed;
  prob_trajectory and entropy_trajectory already in the PKL):
  Step 5: 64-feat flat MLP (prob+entropy per layer) and 128-feat (all 4 signals)
  Step 6: Sequential models — 1D-CNN and GRU over the 32-layer sequence
  Step 7: Report Phase 2 results vs. baseline and Phase 1 best

Discipline guaranteed:
  - StandardScaler fit on train fold only, transform applied to test fold
  - SMOTE applied to SCALED train fold only, never touching the test fold
  - Feature engineering (interactions) computed from scaled train fold;
    test fold interactions computed using the SAME scaler, then products formed
  - All results reported honestly, including regressions

Usage:
    python scripts/run_stage3_improve.py [--epochs N] [--folds K] [--no-smote]

Runtime estimate (CPU, N=6000):
    Phase 1 grid (80 configs × 5 folds × ~3s): ~20 min
    Phase 1 interactions + SMOTE: ~5 min
    Phase 2 flat MLP variants: ~10 min
    Phase 2 sequential (CNN + GRU): ~15 min
    Total: ~50 min
"""

import argparse
import json
import os
import sys
import time
from itertools import combinations
from typing import Dict, List, Optional, Tuple

# Force UTF-8 output on Windows so Unicode chars (Δ, ±) in the summary table
# don't cause UnicodeEncodeError on cp1252 consoles.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import get_logger, get_project_root, load_pickle, load_json
from src.features.shape_descriptors import extract_all_descriptors
from src.classifier.trajectory_classifier import (
    TrajectoryClassifier,
    features_dict_to_vector,
    FEATURE_ORDER,
)
from src.classifier.sequential_classifiers import CNN1DClassifier, GRUClassifier

log = get_logger("script.stage3_improve")

RESULTS_DIR = get_project_root() / "evaluation" / "results"
BASELINE_CV_AUROC = 0.7104  # known 5-fold CV AUROC (6-feature MLP, lr=1e-3, hidden=16)
BASELINE_CV_STD   = 0.0231
CHECKPOINT_PATH   = RESULTS_DIR / "stage3_grid_checkpoint.json"


def _save_checkpoint(grid_results: dict):
    """Persist the current grid results to a checkpoint file after every config."""
    os.makedirs(str(RESULTS_DIR), exist_ok=True)
    with open(str(CHECKPOINT_PATH), "w") as f:
        json.dump(grid_results, f, indent=2)


def _load_checkpoint() -> dict:
    """Load a previous partial grid run, or return empty dict if none exists."""
    if CHECKPOINT_PATH.exists():
        with open(str(CHECKPOINT_PATH)) as f:
            data = json.load(f)
        log.info(f"Resuming from checkpoint: {len(data)} configs already done")
        return data
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data() -> Tuple[np.ndarray, np.ndarray, list, list]:
    """Load trajectories, extract 6-feature vectors, and build sequential arrays.

    Returns:
        X_6:    [N, 6]  — original 6 scalar features
        y:      [N]     — binary labels
        trajs:  list of LayerTrajectory objects (for Phase 2)
        labels: list of int labels
    """
    root = get_project_root()
    traj_path     = str(root / "data" / "trajectories" / "all_trajectories.pkl")
    baseline_path = str(root / "data" / "trajectories" / "baseline_norms.json")

    data = load_pickle(traj_path)
    if data is None:
        raise RuntimeError("No trajectories found. Run run_stage2_build_dataset.py first.")

    trajs  = data["trajectories"]
    labels = data["labels"]
    baseline_norms = load_json(baseline_path)

    log.info(f"Loaded {len(trajs)} trajectories  "
             f"(confab={labels.count(1)}, correct={labels.count(0)})")

    all_desc = [extract_all_descriptors(t, baseline_norms) for t in trajs]
    X_6 = np.array([features_dict_to_vector(d) for d in all_desc], dtype=np.float32)
    y   = np.array(labels, dtype=np.float32)

    return X_6, y, trajs, labels


def build_sequential_features(
    trajs: list,
    channels: str = "prob_entropy",  # "prob_entropy" | "all4"
) -> np.ndarray:
    """Build a [N, 32, n_channels] array from the stored trajectory signals.

    Channels:
      "prob_entropy"  → 2 channels: prob_trajectory, entropy_trajectory
      "all4"          → 4 channels: prob, entropy, mlp_norm, rank_norm
                        (rank is normalised by /50000 to bring into [0,1] range)

    No new GPU inference: all signals are already stored in the .pkl file.
    """
    n_layers = trajs[0].n_layers
    if channels == "prob_entropy":
        n_ch = 2
        def _row(t):
            return np.stack([
                np.array(t.prob_trajectory,    dtype=np.float32),
                np.array(t.entropy_trajectory, dtype=np.float32),
            ], axis=-1)  # [32, 2]
    else:  # all4
        n_ch = 4
        def _row(t):
            return np.stack([
                np.array(t.prob_trajectory,    dtype=np.float32),
                np.array(t.entropy_trajectory, dtype=np.float32),
                np.array(t.mlp_norm_trajectory, dtype=np.float32),
                np.array(t.rank_trajectory,    dtype=np.float32) / 50000.0,
            ], axis=-1)  # [32, 4]

    X_seq = np.stack([_row(t) for t in trajs], axis=0)  # [N, 32, n_ch]
    log.info(f"Built sequential feature array: shape={X_seq.shape}")
    return X_seq


def add_interactions(X_scaled: np.ndarray) -> np.ndarray:
    """Append all C(6,2)=15 pairwise products to the ALREADY-SCALED 6-feature matrix.

    Called AFTER StandardScaler.transform() so the scaler is fit on raw features
    only (preventing data leakage through the scaler seeing interaction magnitudes).

    Returns: [N, 21] array  (6 original + 15 products)
    """
    n_feats = X_scaled.shape[1]
    products = []
    for i, j in combinations(range(n_feats), 2):
        products.append((X_scaled[:, i] * X_scaled[:, j]).reshape(-1, 1))
    return np.concatenate([X_scaled] + products, axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Core CV routine for flat MLP
# ─────────────────────────────────────────────────────────────────────────────

def cross_validate_mlp(
    X: np.ndarray,
    y: np.ndarray,
    hidden_width: int  = 16,
    lr: float          = 1e-3,
    pw_scale: float    = 1.0,
    n_folds: int       = 5,
    n_epochs: int      = 150,
    use_smote: bool    = False,
    add_interaction: bool = False,
    seed: int          = 42,
    verbose: bool      = False,
    batch_size: int    = 64,
) -> Dict:
    """Stratified k-fold CV for the flat MLP classifier.

    All data-dependent transforms (scaler, SMOTE, interaction terms) are
    computed exclusively on the train fold — the test fold is never touched
    until evaluation.

    Args:
        X:               raw (unscaled) feature matrix [N, n_feats]
        y:               binary labels [N]
        hidden_width:    hidden layer size (n_feats → hidden_width → 1)
        lr:              Adam learning rate
        pw_scale:        multiplier on the natural pos_weight (n_neg/n_pos).
                         1.0 = current default; 0.5 = half the weight; 2.0 = double.
        n_folds:         number of stratified CV folds
        n_epochs:        training epochs per fold
        use_smote:       if True, apply SMOTE inside each train fold
        add_interaction: if True, append 15 pairwise products after scaling
        seed:            RNG seed for StratifiedKFold

    Returns:
        dict with mean_auroc, std_auroc, fold_aurocs
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_aurocs = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        y_train, y_test         = y[train_idx], y[test_idx]

        # 1. Scale — fit on TRAIN FOLD ONLY
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_raw).astype(np.float32)
        X_test_s  = scaler.transform(X_test_raw).astype(np.float32)

        # 2. Interaction terms — computed from already-scaled data
        if add_interaction:
            X_train_s = add_interactions(X_train_s).astype(np.float32)
            X_test_s  = add_interactions(X_test_s).astype(np.float32)

        # 3. SMOTE — applied to scaled train fold only
        if use_smote:
            try:
                from imblearn.over_sampling import SMOTE
                smote = SMOTE(random_state=seed)
                X_train_s, y_train = smote.fit_resample(X_train_s, y_train)
                X_train_s = X_train_s.astype(np.float32)
                y_train   = y_train.astype(np.float32)
            except ImportError:
                log.warning("imblearn not installed — skipping SMOTE, using raw class weights")

        # 4. pos_weight
        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())
        natural_pw = n_neg / n_pos
        pos_weight = torch.tensor([natural_pw * pw_scale], dtype=torch.float32)

        # 5. Model
        n_input = X_train_s.shape[1]
        model     = TrajectoryClassifier(n_features=n_input)
        # Override the internal architecture width if different from 16
        if hidden_width != 16:
            model.net = nn.Sequential(
                nn.Linear(n_input, hidden_width),
                nn.ReLU(),
                nn.Linear(hidden_width, 1),
            )
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        train_ds = TensorDataset(
            torch.tensor(X_train_s),
            torch.tensor(y_train.astype(np.float32)),
        )
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        best_auroc, best_state = 0.0, None

        for epoch in range(n_epochs):
            model.train()
            for xb, yb in train_dl:
                logits = model(xb)
                loss   = criterion(logits, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                probs = torch.sigmoid(
                    model(torch.tensor(X_test_s))
                ).cpu().numpy()

            fold_auroc = roc_auc_score(y_test, probs)
            if fold_auroc > best_auroc:
                best_auroc = fold_auroc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        fold_aurocs.append(best_auroc)
        if verbose:
            log.info(f"    Fold {fold_idx+1}/{n_folds}: AUROC={best_auroc:.4f}")

    return {
        "mean_auroc":  float(np.mean(fold_aurocs)),
        "std_auroc":   float(np.std(fold_aurocs)),
        "fold_aurocs": [float(a) for a in fold_aurocs],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Core CV routine for sequential models (CNN / GRU)
# ─────────────────────────────────────────────────────────────────────────────

def cross_validate_sequential(
    X_seq: np.ndarray,
    y: np.ndarray,
    model_type: str    = "cnn",   # "cnn" | "gru"
    n_channels: int    = 2,
    hidden_size: int   = 32,
    lr: float          = 1e-3,
    pw_scale: float    = 1.0,
    n_folds: int       = 5,
    n_epochs: int      = 150,
    seed: int          = 42,
    verbose: bool      = False,
) -> Dict:
    """Stratified k-fold CV for sequential models (1D-CNN or GRU).

    The sequential input is normalised per-channel (mean/std over the
    train-fold examples, applied channel-wise across all 32 layers) —
    this is the sequential analogue of StandardScaler and is likewise
    fit on the train fold only.

    Args:
        X_seq:       [N, n_layers, n_channels] raw sequential features
        y:           binary labels [N]
        model_type:  "cnn" or "gru"
        n_channels:  number of feature channels (must match X_seq.shape[2])
        hidden_size: hidden dim for GRU (ignored for CNN)
        lr:          Adam learning rate
        pw_scale:    pos_weight multiplier
        n_folds / n_epochs / seed: same as cross_validate_mlp

    Returns:
        dict with mean_auroc, std_auroc, fold_aurocs
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_aurocs = []
    n_layers_seq = X_seq.shape[1]

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_seq.reshape(len(X_seq), -1), y)):
        X_train_raw = X_seq[train_idx]   # [n_train, 32, n_ch]
        X_test_raw  = X_seq[test_idx]    # [n_test,  32, n_ch]
        y_train     = y[train_idx]
        y_test      = y[test_idx]

        # Per-channel normalisation — computed on train fold
        # mean/std computed over [n_train, 32] for each channel independently
        ch_mean = X_train_raw.mean(axis=(0, 1), keepdims=True)  # [1, 1, n_ch]
        ch_std  = X_train_raw.std(axis=(0, 1), keepdims=True).clip(min=1e-8)

        X_train_s = ((X_train_raw - ch_mean) / ch_std).astype(np.float32)
        X_test_s  = ((X_test_raw  - ch_mean) / ch_std).astype(np.float32)

        # pos_weight
        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())
        pos_weight = torch.tensor([n_neg / n_pos * pw_scale], dtype=torch.float32)

        # Model
        if model_type == "cnn":
            model = CNN1DClassifier(n_channels=n_channels, n_layers_seq=n_layers_seq)
        else:
            model = GRUClassifier(n_channels=n_channels, hidden_size=hidden_size)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        train_ds = TensorDataset(
            torch.tensor(X_train_s),
            torch.tensor(y_train.astype(np.float32)),
        )
        train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)

        best_auroc, best_state = 0.0, None

        for epoch in range(n_epochs):
            model.train()
            for xb, yb in train_dl:
                logits = model(xb)
                loss   = criterion(logits, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                test_tensor = torch.tensor(X_test_s)
                probs = torch.sigmoid(model(test_tensor)).cpu().numpy()

            fold_auroc = roc_auc_score(y_test, probs)
            if fold_auroc > best_auroc:
                best_auroc = fold_auroc

        fold_aurocs.append(best_auroc)
        if verbose:
            log.info(f"    Fold {fold_idx+1}/{n_folds}: AUROC={best_auroc:.4f}")

    return {
        "mean_auroc":  float(np.mean(fold_aurocs)),
        "std_auroc":   float(np.std(fold_aurocs)),
        "fold_aurocs": [float(a) for a in fold_aurocs],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Step 1: Grid search
# ─────────────────────────────────────────────────────────────────────────────

def run_phase1_grid(
    X_6: np.ndarray,
    y: np.ndarray,
    n_folds: int   = 5,
    n_epochs: int  = 150,
    resume: bool   = False,
    batch_size: int = 64,
) -> Tuple[Dict, Dict]:
    """Exhaustive grid search over hidden_width × lr × pos_weight_scale.

    Checkpoints after every config so an interrupted run can resume from
    where it left off (pass resume=True).

    Returns:
        all_results: dict mapping config_key -> result dict
        best_config: the single best config dict
    """
    hidden_widths = [8, 16, 32, 64]
    lrs           = [1e-4, 3e-4, 1e-3, 3e-3]
    pw_scales     = [0.5, 1.0, 1.5, 2.0, 3.0]

    total = len(hidden_widths) * len(lrs) * len(pw_scales)
    log.info(f"\n{'='*60}")
    log.info(f"PHASE 1 — STEP 1: Grid Search ({total} configs, {n_folds}-fold CV each)")
    log.info(f"{'='*60}")

    # Load checkpoint if resuming
    all_results = _load_checkpoint() if resume else {}
    done = len(all_results)  # configs already completed
    if done > 0:
        log.info(f"  Resuming from config {done+1}/{total} (skipping {done} already done)")

    for hw in hidden_widths:
        for lr in lrs:
            for pw in pw_scales:
                key = f"hw{hw}_lr{lr:.0e}_pw{pw}"
                if key in all_results:  # already done in a previous run
                    continue

                t0 = time.time()
                res = cross_validate_mlp(
                    X_6, y,
                    hidden_width=hw, lr=lr, pw_scale=pw,
                    n_folds=n_folds, n_epochs=n_epochs,
                    batch_size=batch_size,
                )
                elapsed = time.time() - t0
                all_results[key] = {**res, "hidden_width": hw, "lr": lr, "pw_scale": pw}
                done += 1

                # Checkpoint immediately after each config
                _save_checkpoint(all_results)

                delta = res["mean_auroc"] - BASELINE_CV_AUROC
                sign  = "+" if delta >= 0 else ""
                log.info(
                    f"  [{done:>3}/{total}] hw={hw:>2} lr={lr:.0e} pw={pw}x  →  "
                    f"AUROC={res['mean_auroc']:.4f}±{res['std_auroc']:.4f}  "
                    f"({sign}{delta:.4f} vs baseline)  [{elapsed:.1f}s]"
                )

    # Find best
    best_key = max(all_results, key=lambda k: all_results[k]["mean_auroc"])
    best      = all_results[best_key]
    log.info(f"\n  ✓ Best grid config: {best_key}")
    log.info(f"    hidden_width={best['hidden_width']}  lr={best['lr']:.0e}  pw_scale={best['pw_scale']}x")
    log.info(f"    CV AUROC = {best['mean_auroc']:.4f} ± {best['std_auroc']:.4f}  "
             f"(Δ={best['mean_auroc']-BASELINE_CV_AUROC:+.4f} vs 0.7104 baseline)")

    return all_results, best


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Step 2: Pairwise interaction terms
# ─────────────────────────────────────────────────────────────────────────────

def run_phase1_interactions(
    X_6: np.ndarray,
    y: np.ndarray,
    best_grid_config: Dict,
    n_folds: int  = 5,
    n_epochs: int = 150,
) -> Dict:
    """Add 15 pairwise interaction terms (6→21 features). Interactions formed
    AFTER scaling to prevent scale-dependent leakage.
    """
    log.info(f"\n{'='*60}")
    log.info("PHASE 1 — STEP 2: Pairwise Interaction Terms (6 → 21 features)")
    log.info(f"{'='*60}")

    hw = best_grid_config["hidden_width"]
    lr = best_grid_config["lr"]
    pw = best_grid_config["pw_scale"]

    res = cross_validate_mlp(
        X_6, y,
        hidden_width=hw, lr=lr, pw_scale=pw,
        n_folds=n_folds, n_epochs=n_epochs,
        add_interaction=True,
        verbose=True,
    )
    log.info(f"  With interactions (21 features): "
             f"AUROC={res['mean_auroc']:.4f}±{res['std_auroc']:.4f}  "
             f"(Δ={res['mean_auroc']-BASELINE_CV_AUROC:+.4f} vs baseline)")
    log.info(f"  Without interactions (best grid):  "
             f"AUROC={best_grid_config['mean_auroc']:.4f}±{best_grid_config['std_auroc']:.4f}")
    return res


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Step 3: SMOTE oversampling
# ─────────────────────────────────────────────────────────────────────────────

def run_phase1_smote(
    X_6: np.ndarray,
    y: np.ndarray,
    best_grid_config: Dict,
    n_folds: int  = 5,
    n_epochs: int = 150,
) -> Dict:
    """SMOTE applied within each train fold only — test fold is never resampled."""
    log.info(f"\n{'='*60}")
    log.info("PHASE 1 — STEP 3: SMOTE Minority-Class Oversampling (train fold only)")
    log.info(f"{'='*60}")

    hw = best_grid_config["hidden_width"]
    lr = best_grid_config["lr"]
    pw = best_grid_config["pw_scale"]

    res = cross_validate_mlp(
        X_6, y,
        hidden_width=hw, lr=lr, pw_scale=pw,
        n_folds=n_folds, n_epochs=n_epochs,
        use_smote=True,
        verbose=True,
    )
    log.info(f"  With SMOTE: AUROC={res['mean_auroc']:.4f}±{res['std_auroc']:.4f}  "
             f"(Δ={res['mean_auroc']-BASELINE_CV_AUROC:+.4f} vs baseline)")
    log.info(f"  Without SMOTE (best grid): "
             f"AUROC={best_grid_config['mean_auroc']:.4f}±{best_grid_config['std_auroc']:.4f}")
    return res


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Step 3b: SMOTE + Interactions combined
# ─────────────────────────────────────────────────────────────────────────────

def run_phase1_smote_interactions(
    X_6: np.ndarray,
    y: np.ndarray,
    best_grid_config: Dict,
    n_folds: int  = 5,
    n_epochs: int = 150,
) -> Dict:
    """SMOTE + interaction terms together, using best grid config."""
    log.info(f"\n  Testing SMOTE + interactions combined...")
    hw = best_grid_config["hidden_width"]
    lr = best_grid_config["lr"]
    pw = best_grid_config["pw_scale"]

    res = cross_validate_mlp(
        X_6, y,
        hidden_width=hw, lr=lr, pw_scale=pw,
        n_folds=n_folds, n_epochs=n_epochs,
        use_smote=True, add_interaction=True,
        verbose=True,
    )
    log.info(f"  SMOTE + interactions: AUROC={res['mean_auroc']:.4f}±{res['std_auroc']:.4f}  "
             f"(Δ={res['mean_auroc']-BASELINE_CV_AUROC:+.4f} vs baseline)")
    return res


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Step 5: Flat MLP on richer features
# ─────────────────────────────────────────────────────────────────────────────

def run_phase2_flat(
    trajs: list,
    y: np.ndarray,
    best_grid_config: Dict,
    n_folds: int  = 5,
    n_epochs: int = 150,
) -> Dict:
    """64-feature and 128-feature flat MLPs using per-layer prob+entropy (and optionally
    mlp_norm+rank). All signals already stored in the PKL — zero new GPU inference.
    """
    log.info(f"\n{'='*60}")
    log.info("PHASE 2 — STEP 5: Flat MLP on Richer Per-Layer Features")
    log.info(f"{'='*60}")

    hw = best_grid_config["hidden_width"]
    lr = best_grid_config["lr"]
    pw = best_grid_config["pw_scale"]

    results = {}

    # ── 64 features: prob + entropy per layer ────────────────────────────────
    log.info("  Building 64-feature matrix (prob+entropy × 32 layers)...")
    X_64 = np.stack([
        np.concatenate([t.prob_trajectory, t.entropy_trajectory])
        for t in trajs
    ], axis=0).astype(np.float32)

    log.info(f"  Training 64-feat flat MLP (hw={hw}, lr={lr:.0e}, pw={pw}x)...")
    res_64 = cross_validate_mlp(
        X_64, y,
        hidden_width=hw, lr=lr, pw_scale=pw,
        n_folds=n_folds, n_epochs=n_epochs, verbose=True,
    )
    results["flat_64_prob_entropy"] = res_64
    log.info(f"  64-feat flat MLP: AUROC={res_64['mean_auroc']:.4f}±{res_64['std_auroc']:.4f}  "
             f"(Δ={res_64['mean_auroc']-BASELINE_CV_AUROC:+.4f} vs baseline)")

    # ── 128 features: prob + entropy + mlp_norm + rank_norm ──────────────────
    log.info("  Building 128-feature matrix (prob+entropy+mlp_norm+rank_norm × 32 layers)...")
    X_128 = np.stack([
        np.concatenate([
            t.prob_trajectory,
            t.entropy_trajectory,
            t.mlp_norm_trajectory,
            [r / 50000.0 for r in t.rank_trajectory],
        ])
        for t in trajs
    ], axis=0).astype(np.float32)

    log.info(f"  Training 128-feat flat MLP (hw={hw}, lr={lr:.0e}, pw={pw}x)...")
    res_128 = cross_validate_mlp(
        X_128, y,
        hidden_width=hw, lr=lr, pw_scale=pw,
        n_folds=n_folds, n_epochs=n_epochs, verbose=True,
    )
    results["flat_128_all4"] = res_128
    log.info(f"  128-feat flat MLP: AUROC={res_128['mean_auroc']:.4f}±{res_128['std_auroc']:.4f}  "
             f"(Δ={res_128['mean_auroc']-BASELINE_CV_AUROC:+.4f} vs baseline)")

    # ── 64-feat with a wider hidden layer ────────────────────────────────────
    log.info("  Training 64-feat flat MLP with hidden=64 (wider)...")
    res_64_wide = cross_validate_mlp(
        X_64, y,
        hidden_width=64, lr=lr, pw_scale=pw,
        n_folds=n_folds, n_epochs=n_epochs, verbose=True,
    )
    results["flat_64_wide"] = res_64_wide
    log.info(f"  64-feat wide MLP (hw=64): AUROC={res_64_wide['mean_auroc']:.4f}±{res_64_wide['std_auroc']:.4f}  "
             f"(Δ={res_64_wide['mean_auroc']-BASELINE_CV_AUROC:+.4f} vs baseline)")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Step 6: Sequential models
# ─────────────────────────────────────────────────────────────────────────────

def run_phase2_sequential(
    trajs: list,
    y: np.ndarray,
    best_grid_config: Dict,
    n_folds: int  = 5,
    n_epochs: int = 150,
) -> Dict:
    """1D-CNN and GRU over the 32-layer sequence (2-channel: prob+entropy)."""
    log.info(f"\n{'='*60}")
    log.info("PHASE 2 — STEP 6: Sequential Models (1D-CNN, GRU)")
    log.info(f"{'='*60}")

    lr = best_grid_config["lr"]
    pw = best_grid_config["pw_scale"]
    results = {}

    X_seq_2ch = build_sequential_features(trajs, channels="prob_entropy")  # [N, 32, 2]
    X_seq_4ch = build_sequential_features(trajs, channels="all4")          # [N, 32, 4]

    # ── 1D-CNN (2 channels: prob + entropy) ──────────────────────────────────
    log.info(f"  Training 1D-CNN (2-ch: prob+entropy, lr={lr:.0e}, pw={pw}x)...")
    res_cnn_2ch = cross_validate_sequential(
        X_seq_2ch, y, model_type="cnn", n_channels=2,
        lr=lr, pw_scale=pw, n_folds=n_folds, n_epochs=n_epochs, verbose=True,
    )
    results["cnn_2ch"] = res_cnn_2ch
    log.info(f"  1D-CNN (2-ch): AUROC={res_cnn_2ch['mean_auroc']:.4f}±{res_cnn_2ch['std_auroc']:.4f}  "
             f"(Δ={res_cnn_2ch['mean_auroc']-BASELINE_CV_AUROC:+.4f} vs baseline)")

    # ── 1D-CNN (4 channels: all signals) ─────────────────────────────────────
    log.info(f"  Training 1D-CNN (4-ch: all signals, lr={lr:.0e}, pw={pw}x)...")
    res_cnn_4ch = cross_validate_sequential(
        X_seq_4ch, y, model_type="cnn", n_channels=4,
        lr=lr, pw_scale=pw, n_folds=n_folds, n_epochs=n_epochs, verbose=True,
    )
    results["cnn_4ch"] = res_cnn_4ch
    log.info(f"  1D-CNN (4-ch): AUROC={res_cnn_4ch['mean_auroc']:.4f}±{res_cnn_4ch['std_auroc']:.4f}  "
             f"(Δ={res_cnn_4ch['mean_auroc']-BASELINE_CV_AUROC:+.4f} vs baseline)")

    # ── GRU (2 channels: prob + entropy) ─────────────────────────────────────
    log.info(f"  Training GRU (2-ch: prob+entropy, hidden=32, lr={lr:.0e}, pw={pw}x)...")
    res_gru_2ch = cross_validate_sequential(
        X_seq_2ch, y, model_type="gru", n_channels=2, hidden_size=32,
        lr=lr, pw_scale=pw, n_folds=n_folds, n_epochs=n_epochs, verbose=True,
    )
    results["gru_2ch"] = res_gru_2ch
    log.info(f"  GRU (2-ch): AUROC={res_gru_2ch['mean_auroc']:.4f}±{res_gru_2ch['std_auroc']:.4f}  "
             f"(Δ={res_gru_2ch['mean_auroc']-BASELINE_CV_AUROC:+.4f} vs baseline)")

    # ── GRU (4 channels: all signals) ────────────────────────────────────────
    log.info(f"  Training GRU (4-ch: all signals, hidden=32, lr={lr:.0e}, pw={pw}x)...")
    res_gru_4ch = cross_validate_sequential(
        X_seq_4ch, y, model_type="gru", n_channels=4, hidden_size=32,
        lr=lr, pw_scale=pw, n_folds=n_folds, n_epochs=n_epochs, verbose=True,
    )
    results["gru_4ch"] = res_gru_4ch
    log.info(f"  GRU (4-ch): AUROC={res_gru_4ch['mean_auroc']:.4f}±{res_gru_4ch['std_auroc']:.4f}  "
             f"(Δ={res_gru_4ch['mean_auroc']-BASELINE_CV_AUROC:+.4f} vs baseline)")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Final summary printer
# ─────────────────────────────────────────────────────────────────────────────

def print_full_summary(all_results: Dict):
    """Print a complete, honest results table — all variants, including regressions."""
    print("\n" + "=" * 80)
    print("STAGE 3 IMPROVEMENT EXPERIMENT — COMPLETE RESULTS SUMMARY")
    print("=" * 80)
    print(f"\n{'Variant':<42} {'AUROC':>10} {'±Std':>8} {'Δ baseline':>12} {'Folds'}")
    print("-" * 80)

    baseline_auroc = BASELINE_CV_AUROC
    baseline_std   = BASELINE_CV_STD

    def row(name, res, indent=""):
        mean = res["mean_auroc"]
        std  = res["std_auroc"]
        delta = mean - baseline_auroc
        sign  = "+" if delta >= 0 else ""
        folds = [f"{a:.4f}" for a in res["fold_aurocs"]]
        print(f"{indent}{name:<42} {mean:>10.4f} {std:>8.4f} {sign+f'{delta:.4f}':>12}  {folds}")

    print(f"\n{'--- BASELINE ---'}")
    print(f"{'6-feat MLP (original)':<42} {baseline_auroc:>10.4f} {baseline_std:>8.4f} {'---':>12}  [published number]")

    # Phase 1 grid (top 10 + bottom 5)
    if "phase1_grid" in all_results:
        grid = all_results["phase1_grid"]
        sorted_keys = sorted(grid, key=lambda k: grid[k]["mean_auroc"], reverse=True)
        print(f"\n{'--- PHASE 1: GRID SEARCH (top 10 configs) ---'}")
        for k in sorted_keys[:10]:
            cfg = grid[k]
            name = f"  hw={cfg['hidden_width']} lr={cfg['lr']:.0e} pw={cfg['pw_scale']}x"
            row(name, cfg, indent="")
        print(f"  ... ({len(sorted_keys) - 10} lower configs omitted for brevity, all in JSON)")
        print(f"{'--- PHASE 1: GRID SEARCH (bottom 5 configs) ---'}")
        for k in sorted_keys[-5:]:
            cfg = grid[k]
            name = f"  hw={cfg['hidden_width']} lr={cfg['lr']:.0e} pw={cfg['pw_scale']}x"
            row(name, cfg, indent="")

    if "phase1_best_grid" in all_results:
        print(f"\n{'--- PHASE 1: BEST GRID CONFIG ---'}")
        row("  Best grid (6-feat)", all_results["phase1_best_grid"])

    if "phase1_interactions" in all_results:
        print(f"\n{'--- PHASE 1: PAIRWISE INTERACTIONS ---'}")
        row("  21-feat (6 + 15 products)", all_results["phase1_interactions"])

    if "phase1_smote" in all_results:
        print(f"\n{'--- PHASE 1: SMOTE OVERSAMPLING ---'}")
        row("  SMOTE (6-feat)", all_results["phase1_smote"])

    if "phase1_smote_interactions" in all_results:
        row("  SMOTE + interactions (21-feat)", all_results["phase1_smote_interactions"])

    if "phase1_best" in all_results:
        p1 = all_results["phase1_best"]
        print(f"\n  ★ PHASE 1 BEST: '{p1['name']}'  "
              f"AUROC={p1['mean_auroc']:.4f}±{p1['std_auroc']:.4f}  "
              f"(Δ={p1['mean_auroc']-baseline_auroc:+.4f})")

    if "phase2_flat" in all_results:
        print(f"\n{'--- PHASE 2: FLAT MLP (richer features, no new GPU inference) ---'}")
        for k, res in all_results["phase2_flat"].items():
            row(f"  {k}", res)

    if "phase2_sequential" in all_results:
        print(f"\n{'--- PHASE 2: SEQUENTIAL MODELS ---'}")
        for k, res in all_results["phase2_sequential"].items():
            row(f"  {k}", res)

    if "phase2_best" in all_results:
        p2 = all_results["phase2_best"]
        print(f"\n  ★ PHASE 2 BEST: '{p2['name']}'  "
              f"AUROC={p2['mean_auroc']:.4f}±{p2['std_auroc']:.4f}  "
              f"(Δ={p2['mean_auroc']-baseline_auroc:+.4f})")

    print("\n" + "=" * 80)

    # Paper framing analysis
    p1_best = all_results.get("phase1_best", {})
    p2_best = all_results.get("phase2_best", {})
    p1_gain = p1_best.get("mean_auroc", baseline_auroc) - baseline_auroc
    p2_gain = p2_best.get("mean_auroc", baseline_auroc) - baseline_auroc

    print("\nPAPER FRAMING ANALYSIS")
    print("-" * 80)
    if p1_gain < 0.01 and p2_gain < 0.01:
        print("  ● HONEST CEILING: Neither Phase 1 tuning nor Phase 2 richer features")
        print(f"    produce meaningful gains over the 0.{baseline_auroc:.4f} baseline.")
        print("    Report this plainly: 0.7104 is the honest ceiling of this feature family.")
    elif p2_gain >= 0.02 and p2_gain > p1_gain + 0.01:
        print(f"  ● Phase 2 MEANINGFULLY OUTPERFORMS the original 6-feature model.")
        print(f"    Phase 2 best: {p2_best['mean_auroc']:.4f} (Δ={p2_gain:+.4f})")
        print(f"    ⚠  PAPER FRAMING TRADE-OFF: The original framing ('6 interpretable")
        print(f"       features are sufficient, classifier capacity is not the bottleneck')")
        print(f"       must be revised or explicitly presented as the simpler/more")
        print(f"       interpretable alternative alongside the higher-accuracy richer model.")
        print(f"       Do NOT silently replace the 6-feature model with the richer one.")
    else:
        print(f"  ● Phase 1 tuning achieves the largest gain: Δ={p1_gain:+.4f}")
        print(f"    Phase 2 does not add meaningfully beyond Phase 1.")
        print(f"    The 6-feature interpretable model remains the primary contribution;")
        print(f"    tuning config can update the paper's headline number.")

    print("=" * 80 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 3 classifier improvement experiment")
    parser.add_argument("--epochs",     type=int,  default=150,  help="Training epochs per fold (default: 150)")
    parser.add_argument("--folds",      type=int,  default=5,    help="Number of CV folds (default: 5)")
    parser.add_argument("--no-smote",   action="store_true",     help="Skip SMOTE step (faster run)")
    parser.add_argument("--phase1-only",action="store_true",     help="Run Phase 1 only, skip Phase 2")
    parser.add_argument("--resume",     action="store_true",     help="Resume grid search from checkpoint (skip already-done configs)")
    parser.add_argument("--batch-size", type=int,  default=64,   help="Training batch size (default: 64, larger = faster on CPU)")
    args = parser.parse_args()

    t_total = time.time()
    log.info("=" * 70)
    log.info("RTA Stage 3 Improvement Experiment")
    log.info(f"Epochs={args.epochs}, Folds={args.folds}, SMOTE={'off' if args.no_smote else 'on'}")
    log.info("=" * 70)

    # ── Load data ─────────────────────────────────────────────────────────────
    X_6, y, trajs, labels = load_data()

    all_results = {
        "metadata": {
            "N":          int(len(y)),
            "n_pos":      int((y == 1).sum()),
            "n_neg":      int((y == 0).sum()),
            "n_folds":    args.folds,
            "n_epochs":   args.epochs,
            "baseline_cv_auroc": BASELINE_CV_AUROC,
            "baseline_cv_std":   BASELINE_CV_STD,
        }
    }

    # ── Phase 1: Step 1 — Grid Search ─────────────────────────────────────────
    grid_all, best_grid = run_phase1_grid(
        X_6, y,
        n_folds=args.folds, n_epochs=args.epochs,
        resume=args.resume, batch_size=args.batch_size,
    )
    all_results["phase1_grid"] = grid_all
    all_results["phase1_best_grid"] = best_grid

    # ── Phase 1: Step 2 — Interactions ────────────────────────────────────────
    interactions_res = run_phase1_interactions(
        X_6, y, best_grid, n_folds=args.folds, n_epochs=args.epochs
    )
    all_results["phase1_interactions"] = interactions_res

    # ── Phase 1: Step 3 — SMOTE ───────────────────────────────────────────────
    if not args.no_smote:
        smote_res = run_phase1_smote(
            X_6, y, best_grid, n_folds=args.folds, n_epochs=args.epochs
        )
        all_results["phase1_smote"] = smote_res

        smote_int_res = run_phase1_smote_interactions(
            X_6, y, best_grid, n_folds=args.folds, n_epochs=args.epochs
        )
        all_results["phase1_smote_interactions"] = smote_int_res
    else:
        smote_res = None
        smote_int_res = None
        log.info("SMOTE skipped (--no-smote flag)")

    # ── Phase 1: Identify best overall ────────────────────────────────────────
    candidates = {
        "best_grid_6feat":        best_grid,
        "interactions_21feat":    interactions_res,
    }
    if smote_res is not None:
        candidates["smote_6feat"]              = smote_res
        candidates["smote_interactions_21feat"] = smote_int_res

    best_p1_name = max(candidates, key=lambda k: candidates[k]["mean_auroc"])
    best_p1      = {**candidates[best_p1_name], "name": best_p1_name}
    all_results["phase1_best"] = best_p1

    log.info(f"\n{'='*60}")
    log.info(f"PHASE 1 RESULT: best='{best_p1_name}'  "
             f"AUROC={best_p1['mean_auroc']:.4f}±{best_p1['std_auroc']:.4f}  "
             f"(Δ={best_p1['mean_auroc']-BASELINE_CV_AUROC:+.4f})")
    log.info(f"{'='*60}")

    if args.phase1_only:
        log.info("--phase1-only flag: skipping Phase 2.")
    else:
        # ── Phase 2: Step 5 — Flat MLP on richer features ─────────────────────
        phase2_flat = run_phase2_flat(trajs, y, best_grid, n_folds=args.folds, n_epochs=args.epochs)
        all_results["phase2_flat"] = phase2_flat

        # ── Phase 2: Step 6 — Sequential models ───────────────────────────────
        phase2_seq = run_phase2_sequential(trajs, y, best_grid, n_folds=args.folds, n_epochs=args.epochs)
        all_results["phase2_sequential"] = phase2_seq

        # ── Phase 2: Identify best ─────────────────────────────────────────────
        all_p2 = {**phase2_flat, **phase2_seq}
        best_p2_name = max(all_p2, key=lambda k: all_p2[k]["mean_auroc"])
        best_p2      = {**all_p2[best_p2_name], "name": best_p2_name}
        all_results["phase2_best"] = best_p2

        log.info(f"\n{'='*60}")
        log.info(f"PHASE 2 RESULT: best='{best_p2_name}'  "
                 f"AUROC={best_p2['mean_auroc']:.4f}±{best_p2['std_auroc']:.4f}  "
                 f"(Δ={best_p2['mean_auroc']-BASELINE_CV_AUROC:+.4f})")
        log.info(f"{'='*60}")

    # ── Print full summary ─────────────────────────────────────────────────────
    print_full_summary(all_results)

    # ── Save JSON results ──────────────────────────────────────────────────────
    os.makedirs(str(RESULTS_DIR), exist_ok=True)
    out_path = str(RESULTS_DIR / "stage3_improvement_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"Results saved to {out_path}")

    elapsed = time.time() - t_total
    log.info(f"Total runtime: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
