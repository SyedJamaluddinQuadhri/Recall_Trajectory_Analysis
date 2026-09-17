# RTA — Recall Trajectory Analysis
### Mechanistic Hallucination Detection via Layer-wise Factual Recall Signatures

A single-forward-pass hallucination detector that reads how a language model's
confidence evolves layer-by-layer during factual recall, and causally validates
the signal via activation patching.

---

## v3 — A Deeper Audit Found the Most Important Fix Yet

A second, deeper pass — tracing exact tensor positions with concrete
numbers rather than re-reading code — found two categories of issues
more serious than anything in v2.

**1. The causal patching experiment (Stage 4) was silently intervening
at the wrong position for any multi-token answer.** Concretely verified
with a worked example (10-token prompt, 3-token answer): the capture
step read position 0 of the correct example (roughly the "Q" token,
not where its recall event happens), and the patch step targeted
position 12 of the confabulated sequence (the slot predicting a
nonexistent 4th answer token, not the first real one — should have
been position 9). Both were fixed to use explicitly-computed positions
via the same `build_teacher_forced_input()` logic the rest of the
pipeline already relies on, and re-verified numerically after the fix.

**2. The tuned lens, as originally designed, likely could not fit in
4GB VRAM at all — the very first stage of the whole pipeline.**
Computed directly: the original full `nn.Linear(hidden_dim, hidden_dim)`
translator design needs ~0.84GB for weights plus ~1.68GB for Adam's
optimizer states — 2.52GB for the lens alone, which combined with the
2.1GB base model already exceeds the 4GB budget before a single
activation tensor is counted. The training loop's "sum all 32 layers'
losses, then call backward() once" pattern added a further ~3.36GB of
simultaneously-retained logits tensors on top of that.

Fixed with a low-rank translator design (LoRA-style down/up projection
with a zero-initialized up-projection, preserving the exact same
identity-at-init property as the original) that cuts lens parameters
by 10x, plus switching to a per-layer `backward()` call (mathematically
identical accumulated gradients, verified numerically, since each
layer's translator has disjoint parameters) instead of accumulating
all 32 layers into one graph. Combined estimated peak VRAM: ~2.5GB,
comfortably within budget. Both the identity-init property and the
gradient-equivalence claim were verified by actually running the code
(not just reasoning about it) — see the assertions this fix was tested
against during development.

**Also in this pass:** the entire base model is now explicitly frozen
at load time (`model_loader.py`) — Phi-2's final layer norm and LM head
were previously left trainable by default, wasting ~0.5GB on unused
gradients for the LM head alone during lens training. And
`project_trajectory_from_hidden_states()` — called on every single
trajectory extraction, across Stage 2's thousands of examples and every
patched forward pass in Stage 4 — is now correctly wrapped in
`torch.no_grad()`, since it's only ever used for inference and was
previously building a needless autograd graph on every call.

## v2 — What Changed From the First Implementation

A full review pass found and fixed several gaps. If you have the earlier
version of this project, replace these files before running anything —
the fixes are not cosmetic, they change what the pipeline actually measures.

| Area | Problem in v1 | Fix in v2 |
|---|---|---|
| Trajectory extraction | Only tracked the answer's FIRST token (`.split()[0]`) — silently wrong for any multi-token answer like "Marie Curie" | `extract_trajectory()` now teacher-forces the FULL answer span and computes a joint trajectory (geometric-mean probability, mean entropy, worst-case rank) across every answer token, still in exactly one forward pass |
| MLP-alignment feature | Standard deviation was a fabricated proxy (`mean * 0.15`) | `compute_baseline_norms()` now returns a REAL empirically-measured std per layer, computed the same way as the mean |
| Tuned lens trust | No way to check the lens was any good before relying on it | New Stage 1b (`run_stage1b_validate_lens.py`) checks final-layer agreement and monotonic layer-accuracy improvement, with a clear pass/fail and remediation instructions |
| Statistical rigor | Bare AUROC point estimates, single 80/20 split | Bootstrap 95% confidence intervals on every reported AUROC; ablation study uses 5-fold cross-validation (mean ± std) instead of one split |
| Competing baseline | Only compared RTA against its own ablated configs | A REPRODUCED Semantic Entropy Probes baseline (single-layer linear probe) now runs on the exact same labeled data, giving a genuine head-to-head AUROC comparison |
| Causal validation | Only one control (random layer) | Added a SECOND control: patching with norm-matched random noise instead of the correct example's real MLP output — isolates whether the *content* of the write matters, not just its magnitude |
| Code duplication | `activation_patching.py` reimplemented a simplified, single-token version of trajectory computation that could silently drift out of sync | Both files now share `build_teacher_forced_input()` and `project_trajectory_from_hidden_states()` — "before" and "after" trajectories in the causal experiment are guaranteed to use identical logic |

---

## What This Project Does

1. **Instruments** an open-weight LLM (Phi-2) to expose its internal, layer-by-layer
   confidence trajectory during generation — using a technique called the "tuned lens."
2. **Validates the lens** before trusting it for anything downstream (new in v2).
3. **Builds a labeled dataset** of closed-book factual questions where we know whether
   the model's answer is correct or confabulated.
4. **Extracts four shape descriptors** from each trajectory (crystallization layer,
   smoothness, MLP-spike alignment, early top-k presence) — now computed over the
   FULL answer span, not just one token.
5. **Trains a small classifier** on these descriptors, evaluated with cross-validation
   and bootstrap confidence intervals, and compared head-to-head against a reproduced
   Semantic Entropy Probes baseline.
6. **Causally validates** the strongest descriptor (MLP-spike alignment) using
   activation patching, with TWO controls: a random-layer control and a
   noise-vector control.
7. **Evaluates** everything: detection AUROC (with CIs), efficiency vs. baselines,
   and the causal intervention's success rate across all three groups.

---

## Hardware

Runs on **RTX 2050 (4GB VRAM)** using Phi-2 in 4-bit quantization.
Mistral-7B experiments (generalization check) run on Google Colab free tier (T4).

---

## Project Structure

```
rta_project/
├── src/
│   ├── instrumentation/     # Stage 1: hooks, tuned lens, trajectory extraction, lens validation
│   ├── dataset/              # Stage 2: ground-truth dataset construction
│   ├── features/              # Stage 3: the four shape descriptors
│   ├── classifier/            # Stage 3: trajectory classifier
│   ├── causal/                 # Stage 4: activation patching experiments (2 controls)
│   └── evaluation/            # Stage 5: benchmarks, ablations, bootstrap CIs, reproduced baseline
├── scripts/                   # One-command runners per stage
├── configs/                   # All hyperparameters in one place
├── data/                        # Datasets and cached trajectories
├── models/                     # Trained tuned-lens probes + classifier weights
└── notebooks/                  # Analysis and figure generation
```

## Run Order

```bash
python scripts/run_stage1_train_lens.py        # Train tuned-lens probes (~3 hrs)
python scripts/run_stage1b_validate_lens.py     # NEW — validate the lens before trusting it (~5 min)
python scripts/run_stage2_build_dataset.py      # Build labeled trajectories (~4-6 hrs)
python scripts/run_stage3_train_classifier.py   # Extract features + train classifier (CV + bootstrap CI)
python scripts/run_stage4_causal_patching.py --pilot   # Pilot run (20 pairs) FIRST
python scripts/run_stage4_causal_patching.py    # Full causal run (random-layer + noise controls)
python scripts/run_stage5_evaluate.py           # Full benchmark + ablation + reproduced baseline + efficiency table
```

**Do not skip Stage 1b.** If the tuned lens fails validation, every downstream
trajectory is potentially unreliable — the log output tells you exactly what
to rerun if it fails.

---

## Reading the Stage 5 Output

The final table now reports three things that weren't available in v1:

- Every ablation config's AUROC comes with a mean ± std across 5 folds, not
  a single point estimate.
- A "Head-to-head baseline" section directly compares Full RTA against a
  reproduced Semantic Entropy Probes implementation, trained and evaluated
  on the identical labeled examples.
- The causal validation section reports THREE rows — target layer, random-layer
  control, and noise-vector control. A successful result requires the target
  row to clearly beat BOTH controls, not just one.
