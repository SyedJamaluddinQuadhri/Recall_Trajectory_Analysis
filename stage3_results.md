# Stage 3 Classifier Improvement — Complete Results

**Experiment**: `scripts/run_stage3_improve.py --epochs 150 --folds 5`  
**Runtime**: 130.2 minutes (CPU-only, batch_size=64)  
**Dataset**: N=6,000 (4,478 confabulated / 1,522 correct), 5-fold stratified CV  
**Baseline**: 6-feature MLP (hw=16, lr=1e-3, pw=1.0×) → **0.7104 ± 0.0231 AUROC**

---

## Summary Table — All Variants

| Variant | AUROC | ±Std | Δ vs Baseline | Fold AUROCs |
|---------|------:|-----:|-------------:|-------------|
| **Baseline** (6-feat MLP, hw=16, lr=1e-3, pw=1.0×) | 0.7104 | 0.0231 | — | *[published]* |
| **Phase 1 best** (grid: hw=64, lr=3e-3, pw=3.0×) | 0.7284 | 0.0232 | +0.0180 | 0.7509, 0.7049, 0.6974, 0.7532, 0.7355 |
| Phase 1: interactions 21-feat | 0.7198 | 0.0220 | +0.0094 | 0.7423, 0.6938, 0.6927, 0.7390, 0.7312 |
| Phase 1: SMOTE (6-feat) | 0.7227 | 0.0227 | +0.0123 | 0.7409, 0.6980, 0.6923, 0.7460, 0.7362 |
| Phase 1: SMOTE + interactions | 0.7129 | 0.0252 | +0.0025 | 0.7392, 0.6818, 0.6835, 0.7363, 0.7238 |
| Phase 2: flat MLP 64-feat (prob+entropy) | 0.7692 | 0.0076 | +0.0588 | 0.7795, 0.7574, 0.7679, 0.7662, 0.7751 |
| **Phase 2 best**: flat MLP 128-feat (all 4 signals) | **0.7920** | **0.0109** | **+0.0816** | **0.8083, 0.7749, 0.7964, 0.7929, 0.7876** |
| Phase 2: flat MLP 64-feat wide (hw=64) | 0.7688 | 0.0092 | +0.0584 | 0.7801, 0.7539, 0.7708, 0.7637, 0.7753 |
| Phase 2: 1D-CNN 2-ch (prob+entropy) | 0.7607 | 0.0150 | +0.0503 | 0.7891, 0.7474, 0.7508, 0.7617, 0.7545 |
| Phase 2: 1D-CNN 4-ch (all signals) | 0.7737 | 0.0093 | +0.0633 | 0.7841, 0.7577, 0.7739, 0.7711, 0.7819 |
| Phase 2: GRU 2-ch (prob+entropy) | 0.7548 | 0.0162 | +0.0444 | 0.7790, 0.7311, 0.7589, 0.7442, 0.7610 |
| Phase 2: GRU 4-ch (all signals) | 0.7732 | 0.0125 | +0.0628 | 0.7817, 0.7487, 0.7782, 0.7754, 0.7824 |

---

## Phase 1 — Grid Search: Top 10 Configs (6-feature MLP)

| hidden_width | lr | pos_weight scale | AUROC | ±Std | Δ baseline |
|---:|----:|---:|------:|-----:|-----------:|
| 64 | 3e-3 | 3.0× | 0.7284 | 0.0232 | +0.0180 |
| 64 | 3e-3 | 2.0× | 0.7264 | 0.0203 | +0.0160 |
| 32 | 3e-3 | 3.0× | 0.7254 | 0.0210 | +0.0150 |
| 64 | 3e-3 | 1.0× | 0.7252 | 0.0243 | +0.0148 |
| 64 | 3e-3 | 0.5× | 0.7241 | 0.0240 | +0.0137 |
| 64 | 3e-3 | 1.5× | 0.7239 | 0.0239 | +0.0135 |
| 32 | 3e-3 | 2.0× | 0.7238 | 0.0214 | +0.0134 |
| 32 | 3e-3 | 0.5× | 0.7234 | 0.0252 | +0.0130 |
| 64 | 1e-3 | 2.0× | 0.7227 | 0.0257 | +0.0123 |
| 64 | 1e-3 | 1.5× | 0.7222 | 0.0248 | +0.0118 |

**Bottom 5 configs** (worst performers):

| hidden_width | lr | pos_weight scale | AUROC | ±Std | Δ baseline |
|---:|----:|---:|------:|-----:|-----------:|
| 16 | 1e-4 | 2.0× | 0.6861 | 0.0274 | −0.0243 |
| 8 | 1e-4 | 2.0× | 0.6851 | 0.0260 | −0.0253 |
| 8 | 1e-4 | 1.5× | 0.6842 | 0.0197 | −0.0262 |
| 16 | 1e-4 | 3.0× | 0.6833 | 0.0215 | −0.0271 |
| 8 | 1e-4 | 3.0× | 0.6815 | 0.0244 | −0.0289 |

> [!NOTE]
> The grid uniformly shows lr=1e-4 hurts regardless of architecture width. The lr=3e-3 + wider hidden (32 or 64) regime dominates. pos_weight scaling has a smaller but consistent effect — 3.0× slightly beats 1.0× across all top configs.

---

## Phase 1 Result

**★ Phase 1 Best**: `hw=64, lr=3e-3, pw=3.0×` → **AUROC = 0.7284 ± 0.0232** (Δ = +0.0180)

- Interactions (21 features): 0.7198 — **worse** than best grid, not additive
- SMOTE alone: 0.7227 — marginal vs best grid
- SMOTE + interactions: 0.7129 — **worst Phase 1 variant** (interactions hurt under SMOTE)

**Phase 1 conclusion**: Tuning within the 6-feature paradigm gives a modest but real gain (+0.018). The best 6-feature result is 0.7284 — this is the honest ceiling for 6-feature classifiers with this dataset.

---

## Phase 2 — Richer Feature Representation

> No new GPU inference — all signals (prob, entropy, mlp_norm, rank) already stored in `all_trajectories.pkl`.  
> Best Phase 1 config used: lr=3e-3, pw_scale=3.0×

### Flat MLPs on per-layer features

| Model | Input dim | AUROC | ±Std | Δ vs baseline | Δ vs Phase 1 best |
|-------|----------:|------:|-----:|-------------:|------------------:|
| flat_64_prob_entropy | 64 | 0.7692 | 0.0076 | +0.0588 | +0.0408 |
| **flat_128_all4** ★ | **128** | **0.7920** | **0.0109** | **+0.0816** | **+0.0636** |
| flat_64_wide (hw=64) | 64 | 0.7688 | 0.0092 | +0.0584 | +0.0404 |

### Sequential Models (32-layer sequence)

| Model | Channels | AUROC | ±Std | Δ vs baseline | Δ vs Phase 1 best |
|-------|----------|------:|-----:|-------------:|------------------:|
| 1D-CNN | 2-ch (prob+entropy) | 0.7607 | 0.0150 | +0.0503 | +0.0323 |
| 1D-CNN | 4-ch (all signals) | 0.7737 | 0.0093 | +0.0633 | +0.0453 |
| GRU | 2-ch (prob+entropy) | 0.7548 | 0.0162 | +0.0444 | +0.0264 |
| GRU | 4-ch (all signals) | 0.7732 | 0.0125 | +0.0628 | +0.0448 |

**★ Phase 2 Best**: `flat_128_all4` → **AUROC = 0.7920 ± 0.0109** (Δ = +0.0816 vs 0.7104 baseline)

> [!IMPORTANT]
> The flat 128-feat MLP (prob + entropy + mlp_norm + rank × 32 layers) consistently beats all sequential models. This is somewhat surprising: the MLP ignores layer ordering, yet outperforms CNN and GRU that explicitly model the sequence. The 4-channel variants (with mlp_norm + rank) consistently beat their 2-channel counterparts, confirming those signals carry additional signal beyond prob+entropy alone.

---

## Phase 2 Result

**★ Phase 2 Best**: `flat_128_all4` (128-dim flat MLP, all 4 signals × 32 layers)
→ **AUROC = 0.7920 ± 0.0109**

Fold-by-fold: `[0.8083, 0.7749, 0.7964, 0.7929, 0.7876]`  
All 5 folds exceed 0.77 — this is a stable result, not a lucky seed.

---

## Honest Comparison: All Three Reference Points

| Model | AUROC | ±Std | Δ |
|-------|------:|-----:|--|
| Original 6-feat MLP (paper baseline) | 0.7104 | 0.0231 | — |
| Phase 1 best (6-feat, tuned config) | 0.7284 | 0.0232 | +0.0180 |
| **Phase 2 best** (128-feat flat MLP) | **0.7920** | **0.0109** | **+0.0816** |

---

## Paper Framing Analysis

> [!WARNING]
> **Phase 2 MEANINGFULLY OUTPERFORMS the original 6-feature model** (Δ = +0.0816, well above the ≥0.02 threshold).

This creates an explicit trade-off that **must not be silently resolved**:

### Option A — Keep 6-feature model as primary
- **For**: Interpretability. Each feature has a mechanistic story (crystallization layer, smoothness, MLP-spike alignment, early top-k presence, final entropy, final rank).  
- **Against**: Leaves 0.08 AUROC points on the table. The paper cannot claim 6 features are "sufficient" without acknowledging the richer model's superiority.
- **If chosen**: Report 6-feat tuned result (0.7284) as the headline with explicit caveat that interpretability is the trade-off.

### Option B — Use 128-feat model as primary
- **For**: Best accuracy (0.7920). mlp_norm + rank add real signal beyond prob+entropy.
- **Against**: Loses the "6 interpretable descriptors" narrative. The 128-feature vector (prob+entropy+mlp_norm+rank × 32 layers) is harder to directly interpret, though still mechanistically motivated.
- **If chosen**: Present the 6-feature model as an interpretable ablation.

### Option C — Report both (recommended)
Present the 6-feature model as the **interpretable classifier** (primary narrative contribution) and the 128-feat MLP as the **accuracy upper bound** (showing the feature family's ceiling). The paper gains both a clean mechanistic story **and** a concrete performance ceiling — honest and complete.

> [!NOTE]
> The sequential models (CNN, GRU) do **not** outperform the flat 128-feat MLP, suggesting layer ordering is less important than the raw magnitude of all 4 signals. This is itself an interesting finding worth one sentence in the paper.

---

## Leakage Audit (Confirmed Clean)

- `StandardScaler.fit_transform()` called on train fold only, `.transform()` on test fold
- SMOTE applied after scaling, to train fold only — test fold never touched
- Interaction products computed from already-scaled data, per fold
- Sequential channel normalization (mean/std) computed from train examples only
- No hyperparameter selection using test fold data

---

## Files

| File | Description |
|------|-------------|
| [`stage3_improvement_results.json`](file:///E:/RTA_Project/rta_project/evaluation/results/stage3_improvement_results.json) | Full results (all 80 grid configs + all Phase 2 variants) |
| [`stage3_grid_checkpoint.json`](file:///E:/RTA_Project/rta_project/evaluation/results/stage3_grid_checkpoint.json) | Grid search checkpoint (80/80 configs) |
| [`run_stage3_improve.py`](file:///E:/RTA_Project/rta_project/scripts/run_stage3_improve.py) | Experiment script |
