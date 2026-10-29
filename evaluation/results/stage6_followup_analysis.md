# Stage 6 Gated Correction — Follow-up Analysis

**Analysis date:** 2026-10-27  
**Inputs:** `evaluation/results/stage6_gated_correction.json` (1,200 examples, k=5, T=0.8) and `stage6_gated_correction_summary.json`  
**Constraint:** Read-only. No new generation, no retraining, no modification to any existing result file.  
**Outputs:** this file + `evaluation/results/stage6_followup_analysis.json`

---

## 0. Candidate Data Structure — Verification

| Field | Status |
|---|---|
| Candidate keys stored per-candidate | `answer`, `confab_prob` only |
| Per-candidate `is_correct` flag stored in file | **No** |
| Per-candidate correctness source | **Recomputed** using `exact_match(candidate.answer, gold_answers)` — same normalization as `src/dataset/build_factual_qa.py` |
| Recompute agrees with stored `originally_correct` flag | **1200 / 1200** (100%) |
| Examples with exactly 5 candidates | **1195 / 1200** |
| Examples with 4 candidates (generation failures) | **5** — `example_idx` in {5935, 5826, 3226, 4093, 3795} |

**Case:** Per-candidate correctness flags were **not** stored in the original file and were recomputed from `gold_answers` for all analyses below.

---

## 1. Oracle and Sample Accuracy

### 1a. Originally-Confabulated Examples (n = 896)

| Metric | Value | 95% Bootstrap CI |
|---|---|---|
| Oracle best-of-5 (at least 1 candidate correct) | **169 / 896 = 18.86%** | — |
| Mean per-candidate accuracy (T=0.8) | **5.52%** | [4.66%, 6.37%] |
| Greedy accuracy (baseline, T=0) | 0.00% (by definition) | — |

**Interpretation:** Even with oracle selection, only 18.86% of confabulated queries have a correct answer anywhere in the k=5 pool. For the remaining 727 confabulated examples (81.14%), stochastic resampling cannot help regardless of the selector — the model does not have the right fact in the pool at T=0.8.

### 1b. Originally-Correct Examples (n = 304)

| Metric | Value | 95% Bootstrap CI |
|---|---|---|
| All 5 candidates correct | **97 / 304 = 31.91%** | — |
| At least 1 candidate correct | **279 / 304 = 91.78%** | — |
| Mean per-candidate accuracy (T=0.8) | **65.13%** | [61.64%, 68.68%] |
| Greedy accuracy | 100.00% (by definition) | — |

**Interpretation:** Stochastic sampling massively degrades already-correct examples. Greedy decoding gives 100% accuracy; T=0.8 sampling drops mean per-candidate accuracy to 65.13%. In 25 / 304 = 8.22% of correct examples, all 5 sampled candidates are wrong.

### 1c. Greedy vs. T=0.8 Per-Candidate Accuracy (All 1,200 Examples)

| Method | Accuracy | 95% Bootstrap CI |
|---|---|---|
| Greedy (original answers) | **25.33%** (304 / 1,200) | — |
| T=0.8 mean per-candidate | **20.62%** | [18.79%, 22.51%] |

Stochastic sampling at T=0.8 reduces per-candidate accuracy by **−4.71 percentage points** relative to greedy. The 95% CI [18.79%, 22.51%] lies entirely below the greedy rate of 25.33%.

---

## 2. Selector Quality

### 2a. Mixed-Pool Examples

"Mixed" = at least one correct and at least one incorrect candidate in the k=5 pool.

| Metric | Value |
|---|---|
| Mixed-pool examples | **351 / 1,200** (29.25%) |
| Selector picks correct (lowest confab score) | **132 / 351 = 37.61%** |
| Random pick expected rate (within mixed examples) | **42.85%** |
| Selector vs. random gap | **−5.24 percentage points** |

**The selector performs worse than random on mixed-pool examples.** It actively steers toward incorrect answers in the setting where selection matters.

### 2b. Pairwise Within-Example Accuracy

For every (correct, incorrect) candidate pair within the same example, does the correct candidate receive the lower confabulation score?

| Metric | Value |
|---|---|
| Total (correct, incorrect) pairs | **1,691** |
| Pairs where correct has lower score | **777 / 1,691 = 45.95%** |
| Random expected | 50.00% |
| Gap | **−4.05 percentage points** |

The classifier assigns lower (more confident) scores to the correct sampled candidate **less often than chance**. In the within-example comparison, the selector is anti-correlated with candidate correctness.

### 2c. Candidate-Level AUROC (All 6,000 Candidates)

| Metric | Value |
|---|---|
| Total candidates | 5,995 (5 examples had only 4 candidates) |
| Incorrect candidates | 4,758 |
| Correct candidates | 1,237 |
| **Candidate-level AUROC** | **0.6961** |

The population-level AUROC is 0.696, confirming the classifier has genuine discriminative signal. The apparent contradiction with §2b (below-random pairwise accuracy) arises because the classifier was trained on greedy-decoded answers and its calibration does not transfer cleanly to T=0.8 sampled candidates from the same prompt.

---

## 3. Original-in-Pool Variant (Analytic)

Pool = greedy answer + k=5 sampled candidates (6 total). Selector picks the lowest confab-score candidate. Gate applied post-hoc at every tau.

| tau | Triggered | Confab Trig. | Correct Trig. | Fix n (rate) | Dist n (rate) | Acc After | Net Change | Published Net | Delta |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.0** | 1,200 | 896 | 304 | 53 (5.92%) | 124 (40.79%) | 19.42% | **−5.92%** | −6.25% | +0.33% |
| **0.3** | 1,086 | 857 | 229 | 49 (5.72%) | 108 (47.16%) | 20.42% | **−4.92%** | −5.08% | +0.17% |
| **0.4** | 948 | 781 | 167 | 44 (5.63%) | 94 (56.29%) | 21.17% | **−4.17%** | −4.25% | +0.08% |
| **0.5** | 757 | 642 | 115 | 35 (5.45%) | 72 (62.61%) | 22.25% | **−3.08%** | −3.17% | +0.08% |
| **0.6** | 481 | 418 | 63 | 20 (4.78%) | 40 (63.49%) | 23.67% | **−1.67%** | −1.75% | +0.08% |
| **0.7** | 194 | 175 | 19 | 8 (4.57%) | 14 (73.68%) | 24.83% | **−0.50%** | −0.50% | 0.00% |

Adding the greedy answer to the pool yields marginal improvements (0.08–0.33 pp) by occasionally re-selecting the original greedy answer instead of a worse sampled one. Fix rates are unchanged. **No threshold achieves net positive accuracy even with the original in the pool.**

---

## 4. Selective-Prediction View *(Exploratory)*

Using the RTA confabulation score as an abstention signal on the greedy answers (higher score = abstain). Compared against random abstention and the `final_rank_normalized` feature from the stored trajectory pkl (recomputed on CPU without new GPU inference).

### 4a. AURC (Area Under Risk-Coverage Curve — lower is better)

| Method | AURC |
|---|---|
| **RTA (−confab score as confidence)** | **0.58252** |
| Final-layer rank normalized | 0.70820 |
| Random abstention | 0.74668 |

### 4b. Accuracy at Fixed Coverage Levels

| Coverage | RTA | Final-Layer Rank | Random |
|:---:|:---:|:---:|:---:|
| 100% | 25.33% | 25.33% | 25.33% |
| 80% | 29.17% | 26.04% | 26.35% |
| 60% | 33.47% | 25.83% | 25.42% |
| 40% | 41.88% | 27.29% | 26.04% |
| 20% | **53.33%** | 30.00% | 24.17% |

At 20% coverage (answering only the 20% of queries the model is most confident about), RTA achieves 53.33% accuracy — more than double the full-coverage rate of 25.33%. This is the strongest positive result in the experiment.

---

## 5. Bootstrap Confidence Intervals (B = 2,000, resampling examples)

CIs written to `evaluation/results/stage6_followup_analysis.json` (not added to the original summary file).

### 5a. Per-Threshold: Fix Rate, Disturbance Rate, Net Accuracy Change

| tau | Fix Rate | Fix Rate 95% CI | Dist Rate | Dist Rate 95% CI | Net Acc Change | Net Acc 95% CI |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.00** | 5.92% | [4.41%, 7.61%] | 42.11% | [36.84%, 47.75%] | −6.25% | [−8.34%, −4.08%] |
| **0.30** | 5.72% | [4.21%, 7.29%] | 48.03% | [41.70%, 54.55%] | −5.08% | [−7.08%, −3.08%] |
| **0.40** | 5.63% | [4.04%, 7.27%] | 56.89% | [49.08%, 64.54%] | −4.25% | [−6.17%, −2.33%] |
| **0.50** | 5.45% | [3.75%, 7.24%] | 63.48% | [54.70%, 72.17%] | −3.17% | [−4.83%, −1.41%] |
| **0.60** | 4.78% | [2.89%, 6.87%] | 65.08% | [53.33%, 77.42%] | −1.75% | [−3.00%, −0.50%] |
| **0.70** | 4.57% | [1.72%, 7.82%] | 73.68% | [52.17%, 93.33%] | −0.50% | [−1.33%, +0.25%] |

Net accuracy is statistically significantly negative at every threshold **except tau=0.7**, where the CI straddles zero [−1.33%, +0.25%].

### 5b. Paired-Bootstrap

| Comparison | Point Estimate | 95% CI |
|---|---|---|
| Gated (tau=0.7) minus Ungated (tau=0.0) | **+5.79%** | [+3.75%, +7.83%] |
| Gated (tau=0.7) net change vs no correction (0%) | **−0.50%** | [−1.33%, +0.25%] |

The gate provides a statistically significant improvement over the ungated baseline. However, gated correction does **not** provide a statistically significant improvement over doing nothing (CI includes 0).

---

## 6. Breakdowns by Source and Popularity

### 6a. By Dataset Source (tau = 0.5)

| Source | n | Confab | Correct | Oracle Best-of-5 (n) | Oracle Rate | Baseline Acc | Acc After tau=0.5 | Net Change |
|---|---|---|---|---|---|---|---|---|
| **popqa** | 582 | 488 | 94 | 157 | 26.98% | 16.15% | 12.89% | **−3.26%** |
| **triviaqa** | 618 | 408 | 210 | 291 | 47.09% | 33.98% | 30.91% | **−3.07%** |

TriviaQA has a substantially higher oracle rate (47.09% vs 26.98%) and baseline accuracy (33.98% vs 16.15%), confirming Phi-2 has more knowledge of TriviaQA-style facts. Both sources degrade under gated correction at tau=0.5 by approximately the same magnitude (~3.1–3.3 pp).

### 6b. By PopQA Popularity Tercile (tau = 0.5)

Popularity tercile boundaries: Low (pop <= 217), Medium (217 < pop <= 1,244), High (pop > 1,244).

| Tier | n | Oracle Best-of-5 (n) | Oracle Rate | Baseline Acc | Net Change (tau=0.5) |
|---|---|---|---|---|---|
| **Low** (pop <= 217) | 195 | 43 | 22.05% | 12.82% | **−3.59%** |
| **Medium** (217–1,244) | 194 | 44 | 22.68% | 13.92% | **−6.70%** |
| **High** (pop > 1,244) | 193 | 70 | 36.27% | 21.76% | **+0.52%** |

Oracle rate increases with popularity (22% → 22% → 36%). High popularity is the only subgroup with a positive net accuracy change at tau=0.5 (+0.52%), consistent with those entities being better represented in Phi-2's parametric knowledge and thus having more viable candidates in the sampled pool.

---

## 7. What the Data Supports and What It Doesn't

### Supported by the data

1. **RTA detects confabulation.** Candidate-level AUROC = 0.696. The confab score ranks incorrect candidates above correct ones better than chance at the population level.

2. **Gating prevents catastrophic disturbance.** Ungated: net −6.25% (128 disturbed, 53 fixed). Gated tau=0.7: net −0.50% (14 disturbed, 8 fixed). Paired-bootstrap margin +5.79%, CI [+3.75%, +7.83%] is statistically significant.

3. **RTA is a strong abstention/selective-prediction signal.** At 20% coverage, accuracy reaches 53.33% vs 25.33% at full coverage. AURC 0.583 vs random 0.747 and final-layer 0.708.

4. **Oracle coverage is the binding constraint on fix rate.** 81.14% of confabulated examples have no correct candidate in the k=5 T=0.8 pool. No selector improvement can fix these without a different generation strategy.

5. **High-popularity entities benefit most from correction.** The high-popularity tercile is the only subgroup with a positive net accuracy change at tau=0.5, with an oracle rate (36.27%) meaningfully above the others.

6. **T=0.8 sampling actively degrades originally-correct answers.** Mean per-candidate accuracy on correct examples: 65.13% vs 100% greedy. 8.22% of correct examples have zero correct candidates in the pool.

### Not supported by the data

1. **The selector improves over random in the cases where selection matters.** Selector picks the correct candidate 37.6% of the time in mixed-pool examples; random would be correct 42.9% of the time. Pairwise within-example accuracy 45.95% vs 50.0% random. The classifier anti-selects correct sampled candidates.

2. **Gated correction improves accuracy over doing nothing.** At tau=0.7, net accuracy change CI is [−1.33%, +0.25%], straddling zero. No threshold achieves a statistically significant positive net improvement.

3. **Adding the greedy answer to the pool materially changes outcomes.** The original-in-pool variant reduces disturbance by 3–4 examples at low tau, but changes nothing at tau=0.7 and never crosses into positive territory.

4. **The fix rate is limited only by selector quality.** Even when oracle coverage exists (169 / 896 confabulated examples have a correct candidate), the selector fails to reliably pick it, confirming the problem is both oracle coverage and selector quality simultaneously.

5. **Final-layer rank is a competitive abstention signal.** At 20% coverage: 30.00% vs RTA's 53.33%. The trajectory features add substantial value beyond the final-layer token probability alone.
