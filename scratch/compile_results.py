import json
import numpy as np

runs = [
    ('n=20', 42), ('n=20', 7), ('n=20', 123),
    ('n=50', 42), ('n=50', 7), ('n=50', 123)
]

print(f"| Config | Seed | Target Change % [95% CI] | Random Control % [95% CI] | Noise Control % [95% CI] | Target Confab Drop | Target Correct Gain | Noise Correct Gain |")
print("|---|---|---|---|---|---|---|---|")
all_target_rates = []
all_random_rates = []
all_noise_rates = []

for cfg, s in runs:
    n_val = 20 if cfg == 'n=20' else 50
    p = f"evaluation/results/causal_patching_results_n{n_val}_s{s}.json"
    d = json.load(open(p, encoding="utf-8"))
    t = d["summary"]["target_layer"]
    l = d["summary"]["layer_control"]
    n = d["summary"]["noise_control"]
    
    t_str = f"{t['answer_changed_rate']*100:.1f}% [{t['ci_lower']*100:.1f}%, {t['ci_upper']*100:.1f}%]"
    l_str = f"{l['answer_changed_rate']*100:.1f}% [{l['ci_lower']*100:.1f}%, {l['ci_upper']*100:.1f}%]"
    n_str = f"{n['answer_changed_rate']*100:.1f}% [{n['ci_lower']*100:.1f}%, {n['ci_upper']*100:.1f}%]"
    
    print(f"| {cfg} | {s} | {t_str} | {l_str} | {n_str} | {t['avg_confab_token_drop']:+.4f} | {t['avg_correct_token_gain']:+.4f} | {n['avg_correct_token_gain']:+.4f} |")

print("\n--- Detailed Metric Inspection (Target vs Noise vs Random) ---")
for n_val in [20, 50]:
    t_rates = []
    l_rates = []
    n_rates = []
    t_gains = []
    n_gains = []
    t_drops = []
    n_drops = []
    for s in [42, 7, 123]:
        p = f"evaluation/results/causal_patching_results_n{n_val}_s{s}.json"
        d = json.load(open(p, encoding="utf-8"))
        t_rates.append(d["summary"]["target_layer"]["answer_changed_rate"])
        l_rates.append(d["summary"]["layer_control"]["answer_changed_rate"])
        n_rates.append(d["summary"]["noise_control"]["answer_changed_rate"])
        t_gains.append(d["summary"]["target_layer"]["avg_correct_token_gain"])
        n_gains.append(d["summary"]["noise_control"]["avg_correct_token_gain"])
        t_drops.append(d["summary"]["target_layer"]["avg_confab_token_drop"])
        n_drops.append(d["summary"]["noise_control"]["avg_confab_token_drop"])
    print(f"\nMean across 3 seeds for n={n_val}:")
    print(f"  Target Change Rate:  {np.mean(t_rates)*100:.1f}% +/- {np.std(t_rates)*100:.1f}%")
    print(f"  Random Control Rate: {np.mean(l_rates)*100:.1f}% +/- {np.std(l_rates)*100:.1f}%")
    print(f"  Noise Control Rate:  {np.mean(n_rates)*100:.1f}% +/- {np.std(n_rates)*100:.1f}%")
    print(f"  Target Correct Gain: {np.mean(t_gains):+.4f} vs Noise: {np.mean(n_gains):+.4f}")
    print(f"  Target Confab Drop:  {np.mean(t_drops):+.4f} vs Noise: {np.mean(n_drops):+.4f}")
