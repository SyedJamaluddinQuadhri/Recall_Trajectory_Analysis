"""
analyse_attention_results.py — scratch copy
Run: python scripts/analyse_attention_results.py (from project root)
"""

import sys, os, json, math, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import get_project_root, load_json

root = get_project_root()

ATTN_PATH = str(root / "evaluation" / "results" / "causal_patching_results_n400_s7_attention.json")
MLP_PATH  = str(root / "evaluation" / "results" / "causal_patching_FINAL_n400_single_layer.json")


def bootstrap_proportion_ci(outcomes, n_boot=10000, ci=0.95, seed=42):
    rng = random.Random(seed)
    n = len(outcomes)
    if n == 0:
        return 0.0, 0.0, 0.0
    obs = sum(outcomes) / n
    boot = sorted(sum(rng.choices(outcomes, k=n)) / n for _ in range(n_boot))
    lo = boot[int((1 - ci) / 2 * n_boot)]
    hi = boot[int((1 + ci) / 2 * n_boot)]
    return obs, lo, hi


def bootstrap_diff_ci(a, b, n_boot=10000, ci=0.95, seed=43):
    rng = random.Random(seed)
    n = min(len(a), len(b))
    pairs = list(zip(a[:n], b[:n]))
    obs_diff = sum(x - y for x, y in pairs) / n
    boot = sorted(sum(x - y for x, y in rng.choices(pairs, k=n)) / n for _ in range(n_boot))
    return obs_diff, boot[int((1 - ci) / 2 * n_boot)], boot[int((1 + ci) / 2 * n_boot)]


def mcnemar_test(a, b):
    n = min(len(a), len(b))
    b01 = sum(1 for i in range(n) if a[i] == 0 and b[i] == 1)
    b10 = sum(1 for i in range(n) if a[i] == 1 and b[i] == 0)
    if b01 + b10 == 0:
        return 0.0, 1.0, b01, b10
    chi2 = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
    try:
        from scipy.stats import chi2 as chi2_dist
        p = chi2_dist.sf(chi2, df=1)
    except ImportError:
        p = math.erfc(math.sqrt(chi2 / 2))
    return chi2, p, b01, b10


def pearson_phi(a, b):
    n = min(len(a), len(b))
    a = a[:n]; b = b[:n]
    n11 = sum(1 for x, y in zip(a, b) if x == 1 and y == 1)
    n10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    n01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    n00 = sum(1 for x, y in zip(a, b) if x == 0 and y == 0)
    denom = math.sqrt((n11+n10)*(n01+n00)*(n11+n01)*(n10+n00))
    if denom == 0:
        return float('nan')
    return (n11*n00 - n10*n01) / denom


def avg_field(results, field):
    vals = [r.get(field, 0.0) for r in results]
    return sum(vals) / len(vals) if vals else float('nan')


print("Loading attention results:", ATTN_PATH)
attn_data = load_json(ATTN_PATH)
print("Loading MLP results:      ", MLP_PATH)
mlp_data  = load_json(MLP_PATH)

attn_target = [int(r["answer_changed"]) for r in attn_data["target_results"]]
attn_noise  = [int(r["answer_changed"]) for r in attn_data["noise_control_results"]]
attn_layer  = [int(r["answer_changed"]) for r in attn_data["layer_control_results"]]

mlp_target  = [int(r["answer_changed"]) for r in mlp_data["target_results"]]
mlp_noise   = [int(r["answer_changed"]) for r in mlp_data["noise_control_results"]]
mlp_layer   = [int(r["answer_changed"]) for r in mlp_data["layer_control_results"]]

at_p, at_lo, at_hi = bootstrap_proportion_ci(attn_target)
an_p, an_lo, an_hi = bootstrap_proportion_ci(attn_noise)
al_p, al_lo, al_hi = bootstrap_proportion_ci(attn_layer)

mt_p, mt_lo, mt_hi = bootstrap_proportion_ci(mlp_target)
mn_p, mn_lo, mn_hi = bootstrap_proportion_ci(mlp_noise)
ml_p, ml_lo, ml_hi = bootstrap_proportion_ci(mlp_layer)

attn_cg = avg_field(attn_data["target_results"], "correct_token_gain")
attn_cd = avg_field(attn_data["target_results"], "confab_token_drop")
mlp_cg  = avg_field(mlp_data["target_results"],  "correct_token_gain")
mlp_cd  = avg_field(mlp_data["target_results"],  "confab_token_drop")

chi2, p_val, b01, b10 = mcnemar_test(attn_target, attn_noise)
diff_obs, diff_lo, diff_hi = bootstrap_diff_ci(attn_target, attn_noise)

phi = pearson_phi(mlp_target, attn_target)
n_same_1  = sum(1 for x, y in zip(mlp_target, attn_target) if x == 1 and y == 1)
n_mlp_only  = sum(1 for x, y in zip(mlp_target, attn_target) if x == 1 and y == 0)
n_attn_only = sum(1 for x, y in zip(mlp_target, attn_target) if x == 0 and y == 1)
n_neither   = sum(1 for x, y in zip(mlp_target, attn_target) if x == 0 and y == 0)

flags = []
if len(attn_target) != len(mlp_target):
    flags.append(f"INCONSISTENCY: attention n={len(attn_target)}, MLP n={len(mlp_target)}")
if abs(mt_p - 0.275) > 0.04:
    flags.append(
        f"INCONSISTENCY: MLP target rate {mt_p:.3f} deviates from expected 27.5% by >4pp. "
        f"Verify the loaded file is the canonical run."
    )

W = 70
print("\n" + "=" * W)
print("STAGE 4 — ATTENTION-ONLY PATCHING RESULTS")
print("=" * W)
print(f"\nAttention mode (n={len(attn_target)} pairs, seed=7):")
print(f"  Target:  {at_p*100:5.1f}%  [{at_lo*100:.1f}%, {at_hi*100:.1f}%]"
      f"  avg_cg={attn_cg:+.4f}  avg_cd={attn_cd:+.4f}")
print(f"  Random:  {al_p*100:5.1f}%  [{al_lo*100:.1f}%, {al_hi*100:.1f}%]"
      f"  (n={len(attn_layer)} layer-control results)")
print(f"  Noise:   {an_p*100:5.1f}%  [{an_lo*100:.1f}%, {an_hi*100:.1f}%]")

def fmt(p, lo, hi):
    return f"{p*100:5.1f}% [{lo*100:.1f}%,{hi*100:.1f}%]"

print(f"\n{'':-<{W}}")
print(f"{'Condition':<16} {'MLP mode':>22} {'Attention mode':>22}")
print(f"{'':-<{W}}")
print(f"{'Target':<16} {fmt(mt_p,mt_lo,mt_hi):>22} {fmt(at_p,at_lo,at_hi):>22}")
print(f"{'Random-layer':<16} {fmt(ml_p,ml_lo,ml_hi):>22} {fmt(al_p,al_lo,al_hi):>22}")
print(f"{'Noise':<16} {fmt(mn_p,mn_lo,mn_hi):>22} {fmt(an_p,an_lo,an_hi):>22}")
print(f"{'avg_cg':<16} {mlp_cg:+.4f}{'':>15} {attn_cg:+.4f}")
print(f"{'avg_cd':<16} {mlp_cd:+.4f}{'':>15} {attn_cd:+.4f}")
print(f"{'':-<{W}}")

print(f"\nPaired McNemar — ATTENTION target vs noise (continuity-corrected):")
print(f"  Discordant: target-yes/noise-no={b10}, target-no/noise-yes={b01}")
print(f"  chi2={chi2:.3f}, p={p_val:.4f}")
print(f"  Bootstrap 95% CI on (target-noise): {diff_obs*100:+.1f}pp "
      f"[{diff_lo*100:+.1f}pp, {diff_hi*100:+.1f}pp]")
if p_val < 0.05 and diff_lo > 0:
    verdict = "Target significantly beats noise (p<0.05, CI excludes 0)."
elif p_val < 0.05 and diff_hi < 0:
    verdict = "Noise significantly beats target (p<0.05)."
else:
    verdict = "No significant difference between target and noise (paired)."
print(f"  --> {verdict}")

print(f"\nPer-pair correlation — MLP target vs Attention target:")
print(f"  Phi (Matthews): {phi:.3f}")
print(f"  Both changed:   {n_same_1:4d}  |  MLP only: {n_mlp_only:4d}")
print(f"  Attn only:      {n_attn_only:4d}  |  Neither:  {n_neither:4d}")

if flags:
    print(f"\n{'!'*20} INCONSISTENCY FLAGS {'!'*20}")
    for flag in flags:
        print("  ", flag)
else:
    print("\n[No inconsistencies detected.]")
print("=" * W)
