# RTA (Recall Trajectory Analysis) v3 — Setup & Run Guide

A complete, step-by-step walkthrough for the internal/white-box hallucination detection method — probing Phi-2's layer-by-layer confidence trajectory during factual recall, with a causal activation-patching validation stage.

**This is a fundamentally different project from HalluDet-Lite.** HalluDet-Lite treats the LLM as a black box and checks its output against external evidence. RTA opens the model itself and watches its internal computation happen, layer by layer, during generation. If you haven't yet, it's worth re-reading the "brief summary about both methods" explanation from earlier before running this — the mental model matters for interpreting what each stage's output actually means.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Environment Setup](#2-environment-setup)
3. [Installing Dependencies](#3-installing-dependencies)
4. [Verifying Your GPU Setup](#4-verifying-your-gpu-setup)
5. [Project Configuration](#5-project-configuration)
6. [Running Stage 1 — Tuned Lens Training](#6-running-stage-1--tuned-lens-training)
7. [Running Stage 1b — Lens Validation (Do Not Skip)](#7-running-stage-1b--lens-validation-do-not-skip)
8. [Running Stage 2 — Dataset & Trajectory Extraction](#8-running-stage-2--dataset--trajectory-extraction)
9. [Running Stage 3 — Classifier Training](#9-running-stage-3--classifier-training)
10. [Running Stage 4 — Causal Activation Patching](#10-running-stage-4--causal-activation-patching)
11. [Running Stage 5 — Full Evaluation](#11-running-stage-5--full-evaluation)
12. [Understanding What Each Stage's Output Means](#12-understanding-what-each-stages-output-means)
13. [Full Command Reference](#13-full-command-reference)
14. [Troubleshooting](#14-troubleshooting)
15. [Honest Expectations Before You Start](#15-honest-expectations-before-you-start)

---

## 1. Prerequisites

| Requirement | Minimum | Notes |
|---|---|---|
| OS | Windows 10/11, Linux, or macOS | GPU steps assume NVIDIA |
| Python | 3.10 or 3.11 | Same constraint as HalluDet-Lite |
| GPU | NVIDIA, 4GB+ VRAM | Tested design target: RTX 2050. The v3 low-rank tuned lens fix specifically exists to make Phi-2 fit in 4GB |
| Disk space | ~10GB free | Phi-2 weights (~1.5GB in 4-bit), PopQA/TriviaQA datasets, trajectory caches, checkpoints |
| RAM | 16GB recommended | Trajectory extraction over thousands of examples keeps intermediate data in memory |

**If you're setting this up alongside HalluDet-Lite,** use a **separate virtual environment** for each project — they have different (though overlapping) dependency sets, and keeping them isolated avoids version conflicts between the two.

---

## 2. Environment Setup

```bash
# Extract the project
unzip rta_project.zip
cd rta_project
```

**Create and activate a dedicated virtual environment:**

*Windows (Command Prompt):*
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

*Windows (PowerShell):*
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

*Linux / macOS:*
```bash
python3 -m venv venv
source venv/bin/activate
```

Confirm `(venv)` appears in your prompt before continuing.

---

## 3. Installing Dependencies

**Step 1 — PyTorch with CUDA** (same as HalluDet-Lite — check `nvidia-smi` for your CUDA version first):
```bash
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121
```

**Step 2 — Everything else:**
```bash
pip install -r requirements.txt
```

This is a **lighter dependency set than HalluDet-Lite** — RTA doesn't need FAISS, sentence-transformers, or Gradio, since it has no retrieval step and no demo app. It does need `bitsandbytes` for Phi-2's 4-bit quantization.

**No spaCy download needed for RTA** — unlike HalluDet-Lite, claim splitting isn't part of this pipeline; RTA operates directly on (prompt, answer) pairs from PopQA/TriviaQA.

---

## 4. Verifying Your GPU Setup

Identical check to HalluDet-Lite:
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

**This matters more here than for HalluDet-Lite.** RTA's whole hardware feasibility rests on the v3 low-rank tuned lens redesign — the original full-rank design was calculated to need ~4.62GB for the lens alone (before even loading Phi-2), which would not fit on a 4GB card at all. The low-rank fix brings this down to an estimated ~2.5GB peak. If your GPU has less than 4GB, or if you see out-of-memory errors specifically during Stage 1, see Section 14.

**Set the same memory fragmentation variable as HalluDet-Lite** (also auto-applied by `src/utils.py` on import, but worth setting explicitly):
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128    # Linux/Mac
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128       # Windows CMD
```

---

## 5. Project Configuration

All settings live in `configs/config.yaml`. Key ones to know about before your first run:

- `stage1_tuned_lens.rank` — the low-rank bottleneck dimension (default 128). This is the setting that keeps the lens's memory footprint small; don't increase it casually (see Section 6 for what increasing it costs).
- `stage2_dataset.max_samples_per_source` — how many PopQA/TriviaQA examples to pull (default 3,000 per source).
- `stage4_causal.n_matched_pairs` — how many correct/confabulated pairs the causal patching experiment tests (default 100).
- `stage4_causal.pilot_pairs` — the small sanity-check batch (default 20) — **always run this before the full 100+ pair run**.

---

## 6. Running Stage 1 — Tuned Lens Training

This trains a small set of per-layer "translator" modules that let you read out Phi-2's confidence at every layer, not just the final one — the foundation everything else depends on.

```bash
python scripts/run_stage1_train_lens.py
```

**Expected runtime:** ~3 hours on RTX 2050.

**A faster test run:**
```bash
python scripts/run_stage1_train_lens.py --sentences 500 --epochs 1
```

**If Stage 1b (next section) fails validation and asks for more capacity:**
```bash
python scripts/run_stage1_train_lens.py --rank 256
```
Be aware this roughly doubles the lens's memory footprint (from ~0.25GB to ~0.5GB combined weights+optimizer state) — still comfortably within budget on 4GB, but worth knowing if you're already tight on VRAM for other reasons.

**Output:** `models/tuned_lens_phi2.pt`.

---

## 7. Running Stage 1b — Lens Validation (Do Not Skip)

This is a dedicated validation stage that exists specifically because a badly-trained lens can silently produce trajectory data that *looks* plausible but has nothing to do with real factual recall. Run this before touching anything else.

```bash
python scripts/run_stage1b_validate_lens.py
```

**Expected runtime:** ~5 minutes.

**What a pass looks like:**
```
Final-layer agreement rate: 0.9812 (target: > 0.95)
First-third avg accuracy:   0.1204
Last-third avg accuracy:    0.6891
Monotonic improvement:      True (expect True)
✓ Tuned lens PASSED validation. Safe to proceed to Stage 2.
```

The two things being checked: does the lens's reading at the *final* layer nearly exactly match what the model actually outputs (it should, almost by construction), and does the lens's accuracy at predicting the next token generally *improve* as you move from early to late layers (it should, since later layers genuinely know more about the eventual output).

**If this fails,** the log tells you exactly what to try:
```bash
python scripts/run_stage1_train_lens.py --sentences 5000 --epochs 3
python scripts/run_stage1_train_lens.py --rank 256
```
Then re-run Stage 1b. **Do not proceed to Stage 2 with a failed lens** — every trajectory extracted afterward depends on this being trustworthy.

---

## 8. Running Stage 2 — Dataset & Trajectory Extraction

Builds a labeled dataset of closed-book factual questions (PopQA + TriviaQA), scores Phi-2's own answers as correct or confabulated via exact-match, and extracts the full layer-wise trajectory for each one.

```bash
python scripts/run_stage2_build_dataset.py
```

**Expected runtime:** 4–6 hours. Run overnight.

**A faster test run:**
```bash
python scripts/run_stage2_build_dataset.py --samples 500
```

**What happens under the hood:** for each example, Phi-2 generates a closed-book answer (no retrieved context — this is the key difference from HalluDet-Lite's RAV step, which explicitly fetches Wikipedia evidence). The answer is compared against the gold answer via normalized exact-match. Then the **full** answer span (not just its first token — this was a specific correctness fix made during development) is teacher-forced through the model, and a joint trajectory is computed across every layer.

**Output:** `data/processed/labeled_qa.json`, `data/trajectories/all_trajectories.pkl`, `data/trajectories/baseline_norms.json` (the last one holds the empirically-measured mean/std MLP-write norms per layer, needed by the MLP-alignment feature in Stage 3).

---

## 9. Running Stage 3 — Classifier Training

Extracts the four shape descriptors (crystallization layer, smoothness, MLP-spike alignment, early top-k presence) from every cached trajectory, and trains a small classifier.

```bash
python scripts/run_stage3_train_classifier.py
```

**Expected runtime:** ~10 minutes — this stage is fast; all the expensive work happened in Stage 2 and is cached.

**What you'll see:**
```
Final test AUROC: 0.8412 (95% CI: [0.8021, 0.8801])
```
Bootstrap confidence intervals are reported alongside every AUROC — a bare point estimate isn't considered sufficient evidence on its own in this pipeline.

**Output:** `models/trajectory_classifier/classifier.pt`, `scaler.pkl`.

---

## 10. Running Stage 4 — Causal Activation Patching

This is the stage that elevates RTA from "these features correlate with hallucination" to "we can causally intervene and change the model's behavior by manipulating this specific mechanism." **Always run the pilot first.**

```bash
# Pilot — 20 pairs, run this FIRST
python scripts/run_stage4_causal_patching.py --pilot
```

**Expected runtime:** ~15–20 minutes for the pilot.

**Read the pilot's output carefully before scaling up:**
```
Target-layer answer-change rate:       68.0%
Random-layer control change rate:      12.0%
Noise-vector control change rate:      8.0%
A successful validation shows target >> both controls.
```

A successful result shows the target-layer number clearly beating **both** controls — the random-layer control (rules out "patching anywhere works") and the noise-vector control (rules out "any large perturbation works, regardless of content"). If target doesn't clearly beat both, **stop here and diagnose** — this is the cheapest possible point to catch a problem with the core hypothesis, before committing hours to the full run.

**Once the pilot looks right, run the full experiment:**
```bash
python scripts/run_stage4_causal_patching.py --pairs 100
```

**Expected runtime:** ~1–2 hours for 100 pairs.

**Output:** `evaluation/results/causal_patching_results.json`.

---

## 11. Running Stage 5 — Full Evaluation

Produces the complete paper-ready results: the ablation study (final-layer-only → +crystallization → +smoothness → full RTA), a head-to-head comparison against a reproduced Semantic Entropy Probes baseline, an efficiency measurement, and the causal validation summary.

```bash
python scripts/run_stage5_evaluate.py
```

**Expected runtime:** ~15–30 minutes (fast — mostly reads cached data from Stages 2–4).

**What the final table looks like:**
```
RTA — FULL RESULTS SUMMARY
==============================================================================
Ablation Config            Mean AUROC       Std   Features
final_layer_only               0.7654    0.0221          2
+ crystallization              0.7891    0.0198          3
+ smoothness                   0.8102    0.0187          4
full_RTA                       0.8412    0.0165          6

Head-to-head baseline      Mean AUROC       Std
SEP (reproduced)               0.7788    0.0203
Full RTA (ours)                0.8412    0.0165

Method                        Time/query      VRAM   Fwd passes
RTA (ours)                        0.412s      2.3GB            1
SelfCheckGPT                        8-15s        5+          15+
...

Causal Validation Group    Answer changed %    Cryst. shift
target_layer                          64.0%          0.312
layer_control                         11.0%          0.041
noise_control                          7.0%          0.028
```

---

## 12. Understanding What Each Stage's Output Means

| Stage | Output file | What a healthy result looks like |
|---|---|---|
| 1 | `tuned_lens_phi2.pt` | Trains without OOM; low final training loss |
| 1b | `evaluation/results/lens_validation.json` | `final_layer_agreement_rate > 0.95`, `is_monotonic_enough: true` |
| 2 | `all_trajectories.pkl` | Roughly balanced correct/confabulated split (check the log's label distribution) |
| 3 | `classifier.pt` | AUROC meaningfully above 0.5, with a CI that doesn't include 0.5 |
| 4 | `causal_patching_results.json` | `target_layer` answer-change rate clearly higher than both `layer_control` and `noise_control` |
| 5 | `full_evaluation.json` | Full RTA config beats `final_layer_only` in the ablation; Full RTA competitive with or beating the reproduced SEP baseline |

---

## 13. Full Command Reference

```bash
# ── Every session ──────────────────────────────────────────────
source venv/bin/activate                          # Linux/Mac
venv\Scripts\activate.bat                         # Windows

# ── Verification ──────────────────────────────────────────────
python -c "import torch; print(torch.cuda.get_device_name(0))"

# ── Full pipeline, in strict order ────────────────────────────
python scripts/run_stage1_train_lens.py                # ~3 hrs
python scripts/run_stage1b_validate_lens.py            # ~5 min — DO NOT SKIP
python scripts/run_stage2_build_dataset.py             # ~4-6 hrs
python scripts/run_stage3_train_classifier.py          # ~10 min
python scripts/run_stage4_causal_patching.py --pilot   # ~15-20 min — ALWAYS FIRST
python scripts/run_stage4_causal_patching.py --pairs 100  # ~1-2 hrs
python scripts/run_stage5_evaluate.py                  # ~15-30 min

# ── Fast test runs ─────────────────────────────────────────────
python scripts/run_stage1_train_lens.py --sentences 500 --epochs 1
python scripts/run_stage2_build_dataset.py --samples 500
python scripts/run_stage4_causal_patching.py --pilot   # already the fast option

# ── If Stage 1b fails ──────────────────────────────────────────
python scripts/run_stage1_train_lens.py --sentences 5000 --epochs 3
python scripts/run_stage1_train_lens.py --rank 256
```

---

## 14. Troubleshooting

**`CUDA out of memory` during Stage 1**
This should be rare after the v3 low-rank fix, but if it happens: confirm you're not accidentally running an old full-rank checkpoint (delete `models/tuned_lens_phi2.pt` and retrain), and reduce `--sentences` and batch size in `configs/config.yaml`'s `stage1_tuned_lens` section.

**`CUDA out of memory` during Stage 2 or 4**
These stages load Phi-2 itself, which needs its own ~1.5–2GB in 4-bit. If you're also still holding the lens's training-time optimizer state in memory from a previous step in the same Python session, restart the script fresh rather than running stages back-to-back in one interactive session.

**Stage 1b fails and suggests more training — how much is "enough"?**
There's no universal number; start with `--sentences 5000 --epochs 3` and re-check. If it still fails after that, try `--rank 256` next, since a persistently poor last-third accuracy (rather than a persistently poor final-layer-agreement number) points at the lens needing more expressive capacity, not just more training time.

**Stage 4's pilot shows target barely beating the controls**
This is the most important "stop and think" moment in the whole project. It could mean: the target layer selection (`find_peak_confidence_layer`) isn't finding the right layer for these particular examples, the matched pairs aren't well-constructed (check `build_default_subject_list()` in `src/dataset/build_matched_pairs.py` — the starter list is small and may need expanding for more reliable relation-template coverage), or the core hypothesis needs revisiting for this model. Don't scale to the full 100+ pair run until this looks clearly right at the pilot scale.

**Everything runs, but Stage 5's Full RTA doesn't beat the reproduced SEP baseline**
This is a real, informative result, not necessarily a bug — report it honestly rather than tuning until it flips in your favor. It may mean the additional shape descriptors aren't adding as much as hoped for this particular model/dataset combination; that's worth investigating and discussing directly rather than hiding.

**A stage says a required file from a previous stage is missing**
Stages are strictly sequential — run them in the Section 13 order. There's no equivalent of HalluDet-Lite's `--train-only` shortcut for most RTA stages, since each one's output is a hard dependency for the next.

---

## 15. Honest Expectations Before You Start

A few things worth knowing before you commit the ~10 hours this full pipeline takes:

**This has more inherent scientific risk than HalluDet-Lite.** HalluDet-Lite's core techniques (NLI, retrieval) are well-established — you can be fairly confident they'll produce *some* reasonable signal. RTA's central bet — that trajectory shape and MLP-write alignment are genuinely diagnostic of confabulation — is an open empirical question being tested, not a guaranteed result. A weak or ambiguous Stage 4 pilot result is a real, valid scientific outcome, not necessarily something you did wrong.

**Budget real time for the Stage 4 pilot checkpoint specifically.** It's the single most informative moment in the whole pipeline — cheap to run (~20 minutes), and its result should genuinely influence whether you proceed to the expensive full run or go back to reconsider the target-layer selection or matched-pair construction first.

**None of this has been confirmed to run successfully on real hardware yet as of this guide.** Every fix described above (the low-rank lens, the corrected patch-position logic, the per-layer backward pass) was verified through isolated numeric simulation during development, not through a completed end-to-end run of the actual pipeline. Treat your first full run as the real test of all of it together, and don't be surprised if something in Stage 1 or 2 needs a small adjustment the moment it meets real data and real hardware for the first time.
