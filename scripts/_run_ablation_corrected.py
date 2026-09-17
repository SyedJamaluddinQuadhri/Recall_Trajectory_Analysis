import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from src.utils import get_logger, get_project_root, load_json, load_pickle
from src.evaluation.benchmarks import run_ablation_study

log = get_logger("scratch.ablation")
root = get_project_root()
traj_path = str(root / "data" / "trajectories" / "all_trajectories.pkl")
baseline_path = str(root / "data" / "trajectories" / "baseline_norms.json")
output_path = str(root / "evaluation" / "results" / "ablation_corrected.json")

data = load_pickle(traj_path)
baseline_norms = load_json(baseline_path)

log.info(f"Loaded {len(data['trajectories'])} trajectories")
log.info(f"Label dist: correct={data['labels'].count(0)}, confab={data['labels'].count(1)}")
log.info("Running corrected ablation (class-weighted MLP, 5-fold StratifiedKFold)...")

ablation = run_ablation_study(data["trajectories"], data["labels"], baseline_norms, n_folds=5)

with open(output_path, "w") as f:
    json.dump({"ablation_corrected": ablation}, f, indent=2)

configs_order = ["final_layer_only", "+ crystallization", "+ smoothness", "full_RTA"]
print("\n" + "=" * 70)
print("CORRECTED ABLATION (class-weighted MLP, 5-fold StratifiedKFold)")
print("=" * 70)
for name in configs_order:
    m = ablation[name]
    folds_str = ", ".join(f"{a:.4f}" for a in m["fold_aurocs"])
    print(f"  {name:<22}  {m['mean_auroc']:.4f} +/- {m['std_auroc']:.4f}   n_feat={m['n_features']}   folds=[{folds_str}]")

aurocs = [ablation[n]["mean_auroc"] for n in configs_order]
is_monotonic = all(aurocs[i] <= aurocs[i+1] for i in range(len(aurocs)-1))
print()
print(f"Monotonic trend: {'HOLDS' if is_monotonic else 'BROKEN - check values above'}")
print("=" * 70)
