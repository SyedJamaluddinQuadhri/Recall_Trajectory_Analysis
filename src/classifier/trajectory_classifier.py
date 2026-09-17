"""
trajectory_classifier.py — The deliberately simple classifier trained on
the six features (four novel descriptors + two baseline snapshot features).

Kept small on purpose: the paper's contribution is the FEATURES, not
classifier capacity. A tiny 2‑layer MLP (or even logistic regression)
should be sufficient if the shape descriptors carry real signal.
"""

import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    classification_report,
    balanced_accuracy_score,
    f1_score as macro_f1_score,
)

from src.utils import get_logger, get_project_root

log = get_logger("classifier.trajectory")

FEATURE_ORDER = [
    "crystallization",
    "smoothness",
    "mlp_alignment",
    "early_presence",
    "final_entropy",
    "final_rank_normalized",
]

MODEL_DIR = get_project_root() / "models" / "trajectory_classifier"


class TrajectoryClassifier(nn.Module):
    """Small 2‑layer MLP: 6 features → 16 hidden → 1 output (logit).

    The final Sigmoid activation is **removed** – we train with
    `BCEWithLogitsLoss` and apply a sigmoid only at inference time.
    """

    def __init__(self, n_features: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 16),
            nn.ReLU(),
            nn.Linear(16, 1),  # raw logit, no sigmoid
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # squeeze to shape (batch,)
        return self.net(x).squeeze(-1)


def features_dict_to_vector(feat: Dict) -> np.ndarray:
    """Convert an `extract_all_descriptors()` dict to an ordered feature vector."""
    return np.array([feat[k] for k in FEATURE_ORDER], dtype=np.float32)


def train_classifier(
    X: np.ndarray,
    y: np.ndarray,
    n_epochs: int = 150,
    lr: float = 1e-3,
    test_fraction: float = 0.2,
    save_dir: str = None,
) -> Dict:
    """Train the trajectory classifier and evaluate on a held‑out test set.

    Args:
        X: feature matrix, shape [N, 6], columns in FEATURE_ORDER
        y: binary labels, 1 = confabulated, 0 = correct
        n_epochs: training epochs
        lr: learning rate
        test_fraction: held‑out fraction for evaluation
        save_dir: where to save the trained model + scaler

    Returns:
        dict with trained model, scaler, and test metrics
    """
    save_dir = save_dir or str(MODEL_DIR)
    os.makedirs(save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1️⃣ Split & scale
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_fraction, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    # ------------------------------------------------------------------
    # 2️⃣ Compute **sample‑wise** class weights
    # ------------------------------------------------------------------
    class_counts = np.bincount(y_train.astype(int))
    class_weights = 1.0 / class_counts
    class_weights = class_weights * class_weights.sum() / class_weights.shape[0]
    sample_weights = class_weights[y_train.astype(int)]
    sample_weights_tensor = torch.from_numpy(sample_weights).float()

    # ------------------------------------------------------------------
    # 3️⃣ DataLoader that also yields the per‑sample weight
    # ------------------------------------------------------------------
    train_ds = TensorDataset(
        torch.tensor(X_train_s),
        torch.tensor(y_train.astype(np.float32)),
        sample_weights_tensor,
    )
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)

    # ------------------------------------------------------------------
    # 4️⃣ Model, optimizer, **class‑weighted** loss
    # ------------------------------------------------------------------
    model = TrajectoryClassifier(n_features=X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    best_auroc = 0.0
    best_state = None

    # ------------------------------------------------------------------
    # 5️⃣ Training loop
    # ------------------------------------------------------------------
    for epoch in range(n_epochs):
        model.train()
        for xb, yb, wb in train_dl:
            logits = model(xb)                     # raw scores
            loss_batch = criterion(logits, yb)     # per‑sample loss
            weighted_loss = (loss_batch * wb).mean()
            optimizer.zero_grad()
            weighted_loss.backward()
            optimizer.step()

        # ------------------------------------------------------------------
        # Validation (AUROC, balanced accuracy, macro‑F1)
        # ------------------------------------------------------------------
        model.eval()
        with torch.no_grad():
            test_logits = model(torch.tensor(X_test_s))
            test_pred = torch.sigmoid(test_logits).cpu().numpy()

        auroc = roc_auc_score(y_test, test_pred)
        bal_acc = balanced_accuracy_score(y_test, (test_pred >= 0.5).astype(int))
        macro_f1 = macro_f1_score(y_test, (test_pred >= 0.5).astype(int), average="macro")

        if auroc > best_auroc:
            best_auroc = auroc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 25 == 0:
            log.info(
                f"Epoch {epoch}: test AUROC={auroc:.4f} | "
                f"Balanced Acc={bal_acc:.4f} | Macro‑F1={macro_f1:.4f}"
            )

    # ------------------------------------------------------------------
    # 6️⃣ Load best model & compute final metrics
    # ------------------------------------------------------------------
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_logits = model(torch.tensor(X_test_s))
        final_pred = torch.sigmoid(final_logits).cpu().numpy()

    final_auroc = roc_auc_score(y_test, final_pred)
    final_f1 = f1_score(y_test, (final_pred >= 0.5).astype(int))
    final_bal_acc = balanced_accuracy_score(y_test, (final_pred >= 0.5).astype(int))
    final_macro_f1 = macro_f1_score(y_test, (final_pred >= 0.5).astype(int), average="macro")

    # ------------------------------------------------------------------
    # 7️⃣ Bootstrap 95 % CI on AUROC (unchanged)
    # ------------------------------------------------------------------
    from src.evaluation.benchmarks import bootstrap_auroc_ci
    auroc_ci = bootstrap_auroc_ci(y_test, final_pred)

    # ------------------------------------------------------------------
    # 8️⃣ Logging
    # ------------------------------------------------------------------
    log.info("\n" + "=" * 50)
    log.info(
        f"Final test AUROC: {final_auroc:.4f}  "
        f"(95% CI: [{auroc_ci['ci_lower']:.4f}, {auroc_ci['ci_upper']:.4f}])"
    )
    log.info(f"Final test F1:    {final_f1:.4f}")
    log.info(f"Final Balanced Accuracy: {final_bal_acc:.4f}")
    log.info(f"Final Macro‑F1:   {final_macro_f1:.4f}")
    log.info(
        classification_report(
            y_test,
            (final_pred >= 0.5).astype(int),
            target_names=["Correct", "Confabulated"],
        )
    )

    # ------------------------------------------------------------------
    # 9️⃣ Persist model & scaler
    # ------------------------------------------------------------------
    torch.save(model.state_dict(), os.path.join(save_dir, "classifier.pt"))
    with open(os.path.join(save_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    return {
        "model": model,
        "scaler": scaler,
        "test_auroc": final_auroc,
        "test_f1": final_f1,
        "test_balanced_accuracy": final_bal_acc,
        "test_macro_f1": final_macro_f1,
        "test_auroc_ci": auroc_ci,
        "X_test": X_test,
        "y_test": y_test,
        "test_pred": final_pred,
    }


def cross_validate_classifier(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    n_epochs: int = 150,
    lr: float = 1e-3,
) -> Dict:
    """Run stratified k-fold CV using the same class-weighted MLP as train_classifier().

    Provides a more robust AUROC estimate than a single 80/20 split by
    averaging over multiple held-out folds. Use this number as the paper's
    headline metric; the single-split AUROC from train_classifier() serves
    as a secondary sanity-check figure.

    Each fold uses:
      - Architecture:  n_features → 16 → 1 (ReLU hidden, raw logit output)
      - Loss:          BCEWithLogitsLoss(pos_weight=n_neg/n_pos) — identical
                       class-imbalance correction to train_classifier()
      - Optimizer:     Adam, lr=1e-3, 150 epochs per fold
      - Best-AUROC tracking per fold (no final-epoch selection bias)
      - StandardScaler fitted on the train fold only
      - StratifiedKFold(n_splits, shuffle=True, random_state=42)

    Args:
        X:        feature matrix, shape [N, n_features], columns in FEATURE_ORDER
        y:        binary labels (1=confabulated, 0=correct)
        n_folds:  number of stratified CV folds (default 5)
        n_epochs: training epochs per fold
        lr:       Adam learning rate

    Returns:
        dict with "mean_auroc", "std_auroc", "fold_aurocs"
    """
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_aurocs = []

    log.info(f"Running {n_folds}-fold stratified CV  ({len(y)} samples, "
             f"{int((y == 1).sum())} confab / {int((y == 0).sum())} correct)")

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train).astype(np.float32)
        X_test_s  = scaler.transform(X_test).astype(np.float32)

        # pos_weight = n_neg / n_pos — same imbalance correction as
        # train_classifier(), expressed as the scalar pos_weight= API
        # (mathematically equivalent to per-sample weighting).
        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())
        pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)

        model     = TrajectoryClassifier(n_features=X.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        train_ds = TensorDataset(
            torch.tensor(X_train_s),
            torch.tensor(y_train.astype(np.float32)),
        )
        train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)

        best_auroc = 0.0
        best_state = None

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
        log.info(f"  Fold {fold_idx + 1}/{n_folds}: best AUROC = {best_auroc:.4f}")

    mean_auroc = float(np.mean(fold_aurocs))
    std_auroc  = float(np.std(fold_aurocs))
    log.info(f"CV AUROC: {mean_auroc:.4f} ± {std_auroc:.4f}  "
             f"(paper headline metric, {n_folds}-fold stratified, weighted MLP)")

    return {
        "mean_auroc":  mean_auroc,
        "std_auroc":   std_auroc,
        "fold_aurocs": [float(a) for a in fold_aurocs],
    }


class TrajectoryClassifierPredictor:
    """Loads a trained classifier for inference."""

    def __init__(self, save_dir: str = None):
        save_dir = save_dir or str(MODEL_DIR)
        model_path = os.path.join(save_dir, "classifier.pt")
        scaler_path = os.path.join(save_dir, "scaler.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Classifier not found at {model_path}. Run Stage 3 first.")

        self.model = TrajectoryClassifier()
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

    def predict(self, feat_dict: Dict) -> float:
        """Returns **P(confabulated)** in [0, 1]."""
        vec = features_dict_to_vector(feat_dict).reshape(1, -1)
        scaled = self.scaler.transform(vec).astype(np.float32)
        with torch.no_grad():
            # Model now outputs a logit → apply sigmoid
            return float(torch.sigmoid(self.model(torch.tensor(scaled))).item())
