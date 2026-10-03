# Stage 3 Classifier Improvement: Phase 1 (Tuning) + Phase 2 (Richer Features)

## Context

Current headline: **6 hand-designed scalar features → small MLP → 0.7104 ± 0.0231 AUROC** (5-fold CV,
class-weighted, leak-free).

**Dataset:** N=6,000; 4,478 confabulated (pos) / 1,522 correct (neg) — 2.94:1 class ratio.  
**Existing per-trajectory data:** `prob_trajectory[32]`, `entropy_trajectory[32]`,
`mlp_norm_trajectory[32]`, `rank_trajectory[32]` — all already stored in the `.pkl`; no new
GPU inference needed for Phase 2.

> [!IMPORTANT]
> All experiments will be implemented in **one self-contained script** (`scripts/run_stage3_improve.py`) that runs
> all phases sequentially, printing every variant's CV AUROC even if it regresses from baseline.
> No test-fold contamination: scaler, SMOTE, and feature selection all fit inside the training fold only.

---

## Proposed Changes

### Phase 1 — Cheap tuning of the existing 6-feature classifier

#### Step 1: Grid Search (hidden width × lr × pos_weight scale)

Grid:
- `hidden_width` ∈ {8, 16, 32, 64}  
- `lr` ∈ {1e-4, 3e-4, 1e-3, 3e-3}  
- `pw_scale` ∈ {0.5×, 1×, 1.5×, 2×, 3×} × natural class ratio (2.94)

Total: 4 × 4 × 5 = **80 grid points**, each evaluated via full 5-fold stratified CV.
Each fold trains for 150 epochs (same as baseline). Parallelism: sequential (no GPU, pure Python).

Expected compute: ~80 configs × 5 folds × ~3s/fold ≈ **~20 minutes** on CPU.

Report: best config name + its CV AUROC vs. the 0.7104 baseline.

#### Step 2: Pairwise Interaction Terms

Append all C(6,2)=15 products of feature pairs → 6+15=21-dimensional input.  
Scaler fit on train fold (prevents leakage — raw features are scaled before products are formed).

Options compared:
- 6 features (baseline)
- 21 features (6 + 15 pairwise interactions)

Use best grid-search config from Step 1 for the 21-feature run.

#### Step 3: SMOTE Minority-Class Oversampling

Within each training fold only (after scaler.fit_transform on the training fold), apply SMOTE
(`imblearn.over_sampling.SMOTE`) to balance the 3:1 ratio. Test fold never touched.

Compare baseline (no oversampling) vs. SMOTE, holding architecture/lr constant at best from Step 1.

#### Step 4: Report "Phase 1 best"

The single best-performing combination across Steps 1–3, with full config.

---

### Phase 2 — Richer representation from existing trajectory data

#### Step 5: 64-Feature Flat Vector (prob + entropy per layer)

For each of the 32 layers: tuned-lens top-1 probability AND entropy → 2 × 32 = **64 features**.  
These are already stored in every `LayerTrajectory` as `prob_trajectory[32]` and
`entropy_trajectory[32]` — **no new GPU inference needed.**

Feature extraction: `np.concatenate([traj.prob_trajectory, traj.entropy_trajectory])` per sample.

Two sub-experiments:
- **64-flat-MLP**: existing MLP architecture but with 64 inputs (best hidden_width from Step 1)
- **64 + 128 (prob + entropy + mlp_norm + rank)**: 4 × 32 = 128 features, same flat MLP

#### Step 6: Sequential Model (1D-CNN or GRU)

Treat the 32-layer sequence as a sequence, not a flat vector:
- Input shape: `[batch, 32, n_channels]` where n_channels = 2 (prob, entropy) or 4 (all four signals)
- **1D-CNN variant**: 2 conv layers (kernel=3, 16→32 filters) + global avg pool + linear → simple, interpretable
- **Small GRU variant**: 1-layer GRU (hidden=32) + final hidden state → linear

Both trained with the same pos_weight, same 5-fold CV, same no-leakage discipline.

#### Step 7: Report Phase 2 results

Report CV AUROC for each richer-representation variant, compared directly against:
- Baseline 0.7104 (original 6-feature MLP)
- Best Phase 1 result

If Phase 2 meaningfully outperforms Phase 1, flag the paper-framing trade-off explicitly.

---

## Implementation Plan: New Files

### [NEW] `scripts/run_stage3_improve.py`

Self-contained experiment runner. Contains:
1. `load_data()` — loads trajectories PKL, runs `extract_all_descriptors`, builds X_6, y
2. `build_feature_matrices()` — also builds X_21 (with interactions), X_64, X_128, X_seq (sequential)
3. `cross_validate_mlp()` — parameterized version of existing CV: takes `(X, y, hidden_width, lr, pw_scale, use_smote)`
4. `cross_validate_sequential()` — for CNN/GRU variants; takes `(X_seq, y, model_type)`
5. `run_phase1_grid()` — exhaustive grid over Step 1 params, reports all 80 results in a table
6. `run_phase1_interactions()` — Step 2
7. `run_phase1_smote()` — Step 3
8. `run_phase2_flat()` — Step 5 (64-feature and 128-feature flat MLPs)
9. `run_phase2_sequential()` — Step 6 (CNN and GRU)
10. `print_summary()` — prints every variant's AUROC honestly, including regressions
11. Saves a JSON results file to `evaluation/results/stage3_improvement_results.json`

### [NEW] `src/classifier/sequential_classifiers.py`

Contains:
- `CNN1DClassifier(n_channels, n_layers_seq)` — lightweight 1D-CNN for sequential input
- `GRUClassifier(n_channels, hidden_size)` — small GRU for sequential input

No changes to `trajectory_classifier.py` (existing 6-feature MLP stays untouched).

---

## Reporting Requirements

Every variant's CV AUROC is printed, even regressions. Final summary table includes:

| Variant | Features | CV AUROC | Δ vs. baseline |
|---------|----------|----------|----------------|
| Baseline (6-feat MLP) | 6 | 0.7104±0.0231 | — |
| Best grid config | 6 | TBD | TBD |
| + Interactions | 21 | TBD | TBD |
| + SMOTE | 6 | TBD | TBD |
| Phase 1 best | ? | TBD | TBD |
| Phase 2: 64-feat flat MLP | 64 | TBD | TBD |
| Phase 2: 128-feat flat MLP | 128 | TBD | TBD |
| Phase 2: 1D-CNN | 32×2 seq | TBD | TBD |
| Phase 2: GRU | 32×2 seq | TBD | TBD |

---

## Open Questions

> [!NOTE]
> The 64/128-feature Phase 2 experiments reuse `prob_trajectory` and `entropy_trajectory`
> that are **already stored** in the `.pkl` file — no GPU inference or tuned-lens re-run needed.
> This keeps Phase 2 fast (CPU-only, same runtime order as Phase 1).

> [!IMPORTANT]
> **Paper framing decision (if Phase 2 wins):** If the richer 64/128-feat or sequential model
> meaningfully outperforms the original 6-feature result, this creates a trade-off between
> *interpretability* (6 hand-designed features, each with a clear mechanistic story) and
> *accuracy* (richer representation). The plan is to flag this explicitly in the output and
> leave the framing decision to the user. The original 6-feature model is NOT silently replaced.

---

## Verification Plan

### Automated
- Script validates: scaler is fit only on train fold (verified by construction — each fold's scaler is a fresh `StandardScaler()`)
- SMOTE is applied only after `fit_transform(X_train)`, never to test fold
- AUROC is computed on the held-out test fold's raw features (scaled with the fold's own scaler)
- All 5 fold AUROCs printed individually — any anomaly (e.g. fold AUROC >0.98) flags potential leakage

### Manual
- Compare reported baseline CV AUROC (should be ~0.7104) against known number
- Verify grid-search results are plausible (no config should yield >0.85 on 6 features with N=6000)
- Check that Phase 2 sequential models aren't massively overfitting (look at fold std, not just mean)
