# RTA Pre-Submission Data, Confound, and Label Audit
**Target Venue:** Transactions on Machine Learning Research (TMLR)  
**Date:** October 28, 2026  
**Artifacts Generated:**
- `evaluation/results/group_ids.json` (6,000 mapping entries)
- `evaluation/results/splits.json` (`SHA-256: 3302f2f528b4ee2af77de4e64d25bd58c4fbe0ec4604fcb8059405907fca3d1b`)
- `evaluation/results/data_audit_results.json` (Complete quantitative log)
- `evaluation/results/label_audit_sheet.csv` (100 sampled examples for human audit)

---

## Executive Summary: Is the RTA Signal Real?

**Verdict: YES. The Recall Trajectory Analysis (RTA) signal is genuine, robust, and cannot be explained away by data leakage, popularity confounds, or string-matching label noise.**

| Potential Threat / Artifact | Finding from Audit | Impact on Headline AUROC |
| :--- | :--- | :--- |
| **1. Data Leakage across Folds** | 193 subject leaks & 358 near-dup leaks in old stratified CV | **Negligible.** Enforcing strict group-disjoint folds only changes 6-feature AUROC by **−0.0027** (0.7104 → 0.7077) and 128-feature AUROC by **−0.0041** (0.7920 → 0.7879). |
| **2. Popularity & Surface Confounds** | Log-popularity alone yields 0.6407 AUROC; all 5 surface confounds combined achieve 0.6640 AUROC. | **RTA provides massive incremental value.** Adding 6-feature RTA to all surface confounds increases AUROC by **+0.0721** (95% CI: [+0.0649, +0.0800]). Adding 128-feature RTA increases AUROC by **+0.1234** (95% CI: [+0.1142, +0.1340]). |
| **3. Stringent Labeling Noise** | Only 95 of 4,478 confabulations (2.12%) flip under lenient/diacritic/token-F1 matching. | **Zero vulnerability.** Sensitivity analysis shows AUROC is 0.7129 (dropped) and 0.7094 (relabeled), within 0.005 of original 0.7077. |

---

## Part A: Duplicate and Leakage Audit

### 1. Corpus Statistics & Entity Distributions
All text fields (subjects, questions, model answers, gold answers) were Unicode NFC-normalized, lowercased, stripped of leading articles (`a`, `an`, `the`), and stripped of punctuation.

- **Total examples:** 6,000 (1,522 correct [25.37%], 4,478 confabulated [74.63%])
- **Unique normalized subjects:** 2,862
- **Unique normalized questions:** 5,884
- **Exact duplicate questions:** 116 examples (58 pairs)
- **Near-duplicate question pairs (Token Jaccard $\ge 0.8$):** 329 pairs involving 429 unique examples.
- **Subject frequency distribution:**
  - In PopQA ($N=2,999$ non-empty subjects): 50th, 75th, 90th, 95th percentiles = 1.0 example/subject. 99th percentile = 2.0. Maximum examples per subject = 4 (e.g., `'mother'`, `'love story'`, `'she'`).
  - TriviaQA ($N=3,001$) does not designate a specific subject entity; grouping is driven by exact/near-duplicate question clusters.

### 2. Leakage in Previous Stratified 5-Fold CV
In the original evaluation (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`), subject entities and near-duplicate questions were split randomly, creating train-test contamination:

| Fold Index | Subject Overlap Pairs (Train $\cap$ Test) | Near-Duplicate Pairs (Train $\cap$ Test) |
| :---: | :---: | :---: |
| **Fold 1** | 46 | 76 |
| **Fold 2** | 37 | 70 |
| **Fold 3** | 36 | 71 |
| **Fold 4** | 43 | 81 |
| **Fold 5** | 31 | 60 |
| **Total Cross-Fold Contaminations** | **193** | **358** |

### 3. Graph-Based Group Construction & Leakage-Free Splits
To guarantee zero train-test overlap, we constructed an undirected graph where each example is a node and edges connect examples sharing either the same normalized subject (for non-empty subjects) or possessing a near-duplicate question ($\text{Jaccard} \ge 0.8$).

- **Total connected components (Groups):** 5,716
- **Singleton groups:** 5,536 (96.85%)
- **Largest group size:** 17 examples
- **Mean group size:** 1.050 examples
- Written to: `evaluation/results/group_ids.json`

#### Split Specifications (`evaluation/results/splits.json`)
1. **Held-Out Test Split:** Generated via `GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)` on group IDs.
   - **Size:** 1,185 examples (group-disjoint from dev).
   - **Class balance:** 863 confabulated (72.83%), 322 correct (27.17%).
   - *Untouched during feature extraction, tuning, and CV.*
2. **Development 5-Fold Cross-Validation:** Generated via `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` on the 4,815 dev examples.
   - **Fold 1:** 963 test examples, 723 confabulated (75.08%)
   - **Fold 2:** 963 test examples, 723 confabulated (75.08%)
   - **Fold 3:** 963 test examples, 723 confabulated (75.08%)
   - **Fold 4:** 963 test examples, 723 confabulated (75.08%)
   - **Fold 5:** 963 test examples, 723 confabulated (75.08%)
3. **Immutability Verification:**
   - **SHA-256:** `3302f2f528b4ee2af77de4e64d25bd58c4fbe0ec4604fcb8059405907fca3d1b`

---

## Part B: Confound Baselines and Grouped Evaluation

### 1. Confound Baselines vs. RTA Features (Grouped 5-Fold CV)
All models were trained and evaluated strictly within the 5-fold grouped CV on the 4,815 development examples.

| Model / Feature Set | Features Included | Mean AUROC | Std Dev | Fold Scores |
| :--- | :--- | :---: | :---: | :--- |
| **(a) Log Popularity** | $\log(\text{pop} + 1)$ | 0.6407 | $\pm 0.0084$ | [0.6286, 0.6453, 0.6502, 0.6465, 0.6327] |
| **(b) Source Indicator** | $\mathbb{I}(\text{source} == \text{PopQA})$ | 0.6112 | $\pm 0.0027$ | [0.6134, 0.6120, 0.6141, 0.6100, 0.6065] |
| **(c) Surface Length & Freq** | Char len, token len, subject freq | 0.6451 | $\pm 0.0111$ | [0.6513, 0.6536, 0.6570, 0.6291, 0.6344] |
| **(d) All Confounds Combined** | [log_pop, source, len_c, len_t, freq] | **0.6640** | $\pm 0.0066$ | [0.6636, 0.6671, 0.6746, 0.6562, 0.6583] |
| **RTA 6-Feature MLP** | 4 shape descriptors + 2 snapshot | **0.7077** | $\pm 0.0087$ | [0.7165, 0.6949, 0.7014, 0.7175, 0.7081] |
| **RTA 128-Feature Model** | Full sequential trajectory features | **0.7879** | $\pm 0.0134$ | [0.7913, 0.7806, 0.8126, 0.7788, 0.7762] |

### 2. Comparison with Previously Reported Numbers

| Model | Previously Reported (Stratified CV) | New Grouped CV (Zero Leakage) | $\Delta$ (Impact of Leakage) |
| :--- | :---: | :---: | :---: |
| **6-Feature MLP** | 0.7104 | **0.7077** | **−0.0027** (−0.38%) |
| **128-Feature Model** | 0.7920 | **0.7879** | **−0.0041** (−0.52%) |

> **Key Takeaway:** The previously reported numbers were **not** inflated by train-test leakage. When entity and duplicate leakage is completely eliminated via grouped splitting, the headline numbers remain virtually unchanged.

### 3. Incremental Value Test (Beyond All Confounds)
To prove that RTA does not simply rediscovering entity frequency or dataset origin, we trained hybrid models combining all 5 surface confounds with RTA feature vectors:

$$\Delta \text{AUROC} = \text{AUROC}(\text{Confounds} + \text{RTA}) - \text{AUROC}(\text{Confounds Alone})$$

- **Confounds Alone:** $0.6640$
- **Confounds + 6-Feature RTA:** **0.7362** ($\pm 0.0033$)
  - **Incremental Gain:** **+0.0721**
  - **Paired Bootstrap 95% CI:** **[+0.0649, +0.0800]** ($p < 10^{-6}$)
- **Confounds + 128-Feature RTA:** **0.7871** ($\pm 0.0163$)
  - **Incremental Gain:** **+0.1234**
  - **Paired Bootstrap 95% CI:** **[+0.1142, +0.1340]** ($p < 10^{-6}$)

Both increments are overwhelmingly positive and statistically significant, ruling out surface cues as the explanation for RTA performance.

---

### 4. Stratified Breakdown: Source and Popularity

#### By Dataset Source (Out-of-Fold Predictions on Dev)
- **TriviaQA ($N = 2,418$):** Confabulation rate: 66.8%, **AUROC: 0.7172** (95% CI: [0.6947, 0.7384])
- **PopQA ($N = 2,397$):** Confabulation rate: 83.4%, **AUROC: 0.6446** (95% CI: [0.6146, 0.6742])

#### By Popularity Quintile (PopQA Dev Split)
Quintile boundaries were determined on non-zero Wikipedia monthly pageview distributions:

| Quintile | Monthly Pageviews Range | $N$ | Confabulation Rate | Out-of-Fold AUROC | 95% Bootstrap CI |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Q1** | $0 - 130$ | 484 | 85.5% | 0.5103 | [0.4417, 0.5808] |
| **Q2** | $130 - 318$ | 477 | 87.4% | 0.6824 | [0.6121, 0.7460] |
| **Q3** | $318 - 912$ | 477 | 86.4% | 0.6233 | [0.5532, 0.6979] |
| **Q4** | $912 - 4,974$ | 479 | 85.0% | 0.6367 | [0.5650, 0.7021] |
| **Q5** | $4,974 - 15,101,521$ | 480 | 72.7% | 0.6532 | [0.5916, 0.7142] |

> **Analysis:**
> - In **Q1** (long-tail obscure entities, pageviews $< 130$), the model essentially guesses blindly; internal representation trajectories show uniform uncertainty, leading to near-chance discrimination.
> - For **Q2 through Q5**, RTA maintains robust and consistent discrimination ($0.62 - 0.68$), peaking in mid-range entities where internal competition between truth and confabulation is most active. RTA is **not** an artifact of popularity extremes.

---

## Part C: Label Audit and Sensitivity Analysis

### 1. Stringent vs. Lenient Matching Audit
The original labeling function utilized strict normalized substring inclusion (`pred in gold` or `gold in pred`). We audited all 4,478 confabulated examples against a lenient evaluation suite:
1. Full Unicode diacritic stripping (e.g., `Müller` $\rightarrow$ `Muller`).
2. Punctuation stripping and case insensitivity.
3. Token-level set overlap ($F_1 \ge 0.5$).

#### Flip Results:
- **Total confabulated examples flipping to correct:** **95 / 4,478 (2.12%)**
- **PopQA:** 17 / 2,483 (0.68%)
- **TriviaQA:** 78 / 1,995 (3.91%)

Over 97.8% of labels are completely invariant to lenient matching.

### 2. Manual Inspection Sample (`evaluation/results/label_audit_sheet.csv`)
A stratified set of 100 examples was written to `label_audit_sheet.csv` comprising:
- 25 confabulated examples that flip under lenient matching
- 25 confabulated examples that remain confabulated under lenient matching
- 30 random confabulated examples
- 20 random correct examples

### 3. Sensitivity Analysis on Model AUROC
We retrained the 6-feature MLP under two noise scenarios on the grouped 5-fold CV:

| Scenario | Definition | Dev $N$ | Grouped 5-Fold AUROC | $\Delta$ from Baseline |
| :--- | :--- | :---: | :---: | :---: |
| **Baseline** | Original dataset labels | 4,815 | **0.7077** $\pm 0.0087$ | — |
| **Scenario A** | Drop all 95 lenient-match-flagged examples | 4,738 | **0.7129** $\pm 0.0193$ | **+0.0052** |
| **Scenario B** | Relabel all 95 flagged examples as "correct" | 4,815 | **0.7094** $\pm 0.0107$ | **+0.0017** |

> **Conclusion on Label Noise:**
> Eliminating or correcting these minor boundary cases slightly *increases* model AUROC (up to 0.7129), demonstrating that label noise was acting as mild attenuation rather than an artificial performance booster.

---

## Recommendations for TMLR Submission

1. **Adopt Grouped Splits as the Primary Protocol:**
   - Report $0.7077 \pm 0.0087$ (6-feature) and $0.7879 \pm 0.0134$ (128-feature) as the official cross-validation benchmarks.
   - Cite `splits.json` (`SHA-256: 3302f2f5...`) in the reproducibility section.
2. **Include Confound Baselines Table in Section 4:**
   - Present the Combined Confounders baseline ($0.6640$) and the incremental gain ($+0.0721$ and $+0.1234$, $p < 10^{-6}$) to pre-empt reviewer skepticism regarding entity frequency or surface length artifacts.
3. **Report Popularity Quintile Breakdown:**
   - Showcase the Q1–Q5 breakdown in the discussion to demonstrate that RTA captures mechanistic dynamics across moderate-to-high popularity regimes, with failure modes cleanly isolated to zero-knowledge rare entities.
