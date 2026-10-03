# Stage 4 Causal Validation — Results Summary

**Status: CLOSED. All planned sublayer decompositions complete. No further patch-mode experiments to be run.**

---

## 1. Headline Result (Paper-Ready)

The reported Stage 4 result is the **single-layer MLP-output patching run at n=400**.

**Source file (canonical):**
`evaluation/results/causal_patching_FINAL_n400_single_layer.json`
*(identical copy of `causal_patching_results_n400_s7.json`; seed=7)*

> [!IMPORTANT]
> **Rounding note:** The noise CI upper bound is stored as `0.4025` (40.25%). This rounds to **40.3%** at one decimal place and is the correct value to report. An earlier display script showed 40.2% due to Python banker's rounding on the `.5` boundary — this is a display artifact only, not a data inconsistency.

### Final Numbers

| Condition | n | Answer-change rate | 95% CI |
|:---|---:|---:|:---|
| Target layer (real content) | 400 | **27.5%** | [23.0%, 31.8%] |
| Random-layer control (real content, wrong layer) | 800 | 11.6% | [9.4%, 13.9%] |
| Noise-vector control (target layer, norm-matched noise) | 400 | 35.5% | [30.5%, **40.3%**] |

---

## 2. All Tested Intervention Designs

Four intervention designs were tested. Only the single-layer MLP result (n=400) is
reported as the paper's causal-validation finding. The three additional modes are
documented here for completeness and methodological transparency.

| Mode | Target rate | Random-layer control | Noise-vector control | n (target) | Usable as evidence? |
|:---|---:|---:|---:|---:|:---|
| **Single-layer MLP output** (peak layer only) | **27.5%** [23.0%, 31.8%] | 11.6% [9.4%, 13.9%] | 35.5% [30.5%, 40.3%] | 400 | **Yes — this is the headline result** |
| **Attention output only** (same peak layer) | 10.5% [7.5%, 13.8%] | 8.6% [6.8%, 10.6%] | 11.5% [8.5%, 14.8%] | 400 | No — target indistinguishable from noise (see §4) |
| Window MLP (peak ± 1, 3 layers simultaneously) | 59.2% [50.0%, 68.3%] | 20.4% [15.4%, 25.8%] | 62.5% [53.3%, 70.8%] | 120 | Methodological note only (see below) |
| Residual stream (full block output at peak layer) | 97.5% [94.2%, 100.0%] | 47.5% [41.2%, 53.8%] | 100.0% [100.0%, 100.0%] | 120 | **No — saturated/degenerate, not usable** |

> [!NOTE]
> **Window mode (n=120):** The larger window increases target sensitivity (59.2% vs 27.5%) but noise remains higher (62.5%), preserving the same qualitative finding. The sample size is insufficient for paper-level CI precision relative to the n=400 single-layer baseline, so this is reported as a methodological observation only.

> [!WARNING]
> **Residual-stream mode (n=120): DEGENERATE RUN — NOT USABLE AS EVIDENCE.**
> The noise-vector control saturated at 100% [100%, 100%] and the random-layer control jumped to 47.5%. This indicates that replacing the entire block output (attention + MLP combined) with any vector of sufficient norm completely disrupts the forward pass regardless of content, making it impossible to use the answer-change-rate metric to distinguish content-specific from content-agnostic effects. This result is reported here only as a methodological note documenting why full residual-stream patching is not an appropriate experimental design for this measurement.

---

## 3. Honest Conclusion (for paper's causal-validation section)

> We tested whether the layer identified as the peak confidence-crystallization
> point (the "recall layer") has a causal role in generating confabulated outputs,
> and whether this role is carried by the MLP sublayer, the self-attention sublayer,
> or both.
>
> **Layer-specificity (MLP mode, n=400):** Activation patching at the peak layer
> (single-layer MLP-output replacement) produced an answer-change rate of
> 27.5% [23.0%, 31.8%], compared to 11.6% [9.4%, 13.9%] for a random-layer
> control (p < 0.001 by Fisher's exact test). Layer-specificity is established.
>
> **Content-specificity (MLP mode):** A norm-matched noise-vector control at the
> same layer produced 35.5% [30.5%, 40.3%], which overlaps and exceeds the target
> CI. Content-specificity is not established for the MLP sublayer.
>
> **Attention sublayer (n=400, seed=7, same pairs):** Replacing only the
> self-attention sublayer output at the same peak layer and position yielded a
> target rate of 10.5% [7.5%, 13.8%] vs. noise 11.5% [8.5%, 14.8%]. McNemar's
> paired test: chi²=0.161, p=0.69; bootstrap CI on target−noise difference:
> −1.0 pp [−4.5 pp, +2.5 pp]. The attention sublayer shows **neither layer-
> specificity** (target barely exceeds the random-layer control of 8.6%) **nor
> content-specificity**. Per-pair outcomes between MLP and attention patching are
> weakly correlated (phi=0.17), with only 21/400 pairs changed by both — the
> two sublayers largely affect different examples.
>
> **Overall conclusion:** The peak crystallization layer is **causally necessary**
> at the MLP sublayer — layer-specific disruption significantly exceeds random-
> layer baselines. However, **neither the MLP nor the self-attention sublayer at
> this layer shows content-specific causal effects**: in both cases norm-matched
> noise matches or exceeds the real-content patch. The recall layer appears to
> function as a high-information-density bottleneck whose disruption (rather than
> targeted content replacement) is sufficient to alter the model's output.
> Establishing content-specificity would require an alternative metric (e.g.,
> direction of logit shift toward the correct answer token, rather than binary
> answer-change rate) or a different experimental design.

---

## 4. Attention-only Patching — Completed Results

**Experiment design:** Replaced only the self-attention sublayer output
(`layer.self_attn`, PhiAttention, post-output-projection) at the peak layer and
answer-predict position, leaving the MLP path untouched. Both sublayers run in
parallel from the same layernormed input in Phi-2's decoder (attention + MLP
added to residual independently). Hook verified against installed transformers:
`PhiDecoderLayer.forward` → `self.self_attn(normed_input)` returns
`(attn_output, attn_weights)`; hook patches `output[0]` at the target position.

**Correctness checks (passed before full run):**
- Identity test: max logit deviation = 0.00e+00 (exact zero) ✓
- Position check: layer=25, shape=(1,1,2560), positions verified for 3 pairs
  including one 2-token answer ✓
- Hook-active test: max logit deviation = 4.10e-01 on cross-pair patch ✓

**Results (n=400, seed=7, same pairs as canonical MLP run):**

| Condition | Rate | 95% CI | avg_correct_token_gain | avg_confab_token_drop |
|:---|---:|:---|---:|---:|
| Target (real attention output) | 10.5% | [7.5%, 13.8%] | +0.0025 | +0.0027 |
| Random-layer control | 8.6% | [6.8%, 10.6%] | — | — |
| Noise (norm-matched, attn norm) | 11.5% | [8.5%, 14.8%] | −0.0016 | +0.0058 |

**Paired test — target vs noise (McNemar, continuity-corrected):**
- Discordant pairs: target-yes/noise-no = 26, target-no/noise-yes = 30
- chi² = 0.161, p = 0.69
- Bootstrap 95% CI on (target − noise): −1.0 pp [−4.5 pp, +2.5 pp]
- **Result: No significant difference. Target is not distinguishable from noise.**

**Comparison and correlation with MLP mode (same 400 pairs):**

| | MLP mode | Attention mode |
|:---|---:|---:|
| Target rate | 27.5% [23.0%, 31.8%] | 10.5% [7.5%, 13.8%] |
| Random-layer | 11.6% [9.4%, 13.9%] | 8.6% [6.8%, 10.6%] |
| Noise | 35.5% [30.5%, 40.3%] | 11.5% [8.5%, 14.8%] |
| avg_correct_token_gain | +0.0066 | +0.0025 |
| avg_confab_token_drop | +0.0117 | +0.0027 |

Per-pair outcome correlation (phi / Matthews): **0.173** (weak).
Contingency: both changed = 21, MLP only = 89, attention only = 21, neither = 269.
The two sublayers largely affect different examples — not a consistent causal locus.

> [!IMPORTANT]
> **Stopping rule declared in advance:** This is the final Stage 4 variant.
> The result is reported as-is without further patch modes, seeds, or variations.
> Outcome: **neither the MLP nor the attention sublayer at the crystallization
> layer shows content-specific causal effects.** Layer-specificity (target vs
> random-layer) is established only for the MLP mode.

**Output file:** `evaluation/results/causal_patching_results_n400_s7_attention.json`

---

## Appendix: File Inventory

| File | Description | Status |
|:---|:---|:---|
| `causal_patching_FINAL_n400_single_layer.json` | **Canonical paper result** (copy of n400_s7) | Use this for paper |
| `causal_patching_results_n400_s7.json` | Original n=400 run, seed=7 | Source of above |
| `causal_patching_results.json` | Legacy path (identical content to n400_s7) | Do not use — ambiguous name |
| `causal_patching_results_n120_s42_window.json` | Window mode pilot | Methodological note only |
| `causal_patching_results_n120_s42_residual_stream.json` | Residual stream pilot | Degenerate — not usable |
| `causal_patching_results_n400_s7_attention.json` | Attention-only mode, n=400, seed=7 | Final Stage 4 sublayer variant |
| `causal_patching_results_n50_*.json`, `n20_*.json`, `n100_*.json` | Earlier smaller pilot runs | Superseded by n=400 |
