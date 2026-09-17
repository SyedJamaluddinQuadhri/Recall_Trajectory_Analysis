"""
run_stage5_evaluate.py — Stage 5: Full evaluation suite.

Runs the ablation study, measures efficiency, and prints the complete
paper-ready results table combining detection quality, efficiency
comparison, and causal validation summary.

Runtime: ~15 minutes
Output:  evaluation/results/full_evaluation.json

Usage:
    python scripts/run_stage5_evaluate.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import get_logger, get_project_root, save_json, load_json, load_pickle
from src.instrumentation.model_loader import load_phi2, get_model_structure
from src.instrumentation.tuned_lens import load_tuned_lens
from src.evaluation.benchmarks import (
    run_ablation_study, measure_efficiency, print_full_results_table,
    run_semantic_entropy_probe_baseline,
)

log = get_logger("script.stage5")


def main():
    log.info("=" * 60)
    log.info("RTA Stage 5 — Full Evaluation Suite")
    log.info("=" * 60)

    root = get_project_root()
    traj_path = str(root / "data" / "trajectories" / "all_trajectories.pkl")
    baseline_path = str(root / "data" / "trajectories" / "baseline_norms.json")
    causal_path = str(root / "evaluation" / "results" / "causal_patching_results.json")
    output_path = str(root / "evaluation" / "results" / "full_evaluation.json")

    all_results = {}

    # ── Ablation study ─────────────────────────────────────────────────────────
    data = load_pickle(traj_path)
    baseline_norms = load_json(baseline_path)

    if data:
        log.info("Running ablation study...")
        ablation = run_ablation_study(data["trajectories"], data["labels"], baseline_norms)
        all_results["ablation"] = ablation

        log.info("Running reproduced Semantic Entropy Probes baseline (head-to-head, same data)...")
        sep_baseline = run_semantic_entropy_probe_baseline(data["trajectories"], data["labels"])
        all_results["semantic_entropy_probe_baseline"] = sep_baseline
    else:
        log.warning("No trajectories found — skipping ablation and baseline. Run Stage 2-3 first.")

    # ── Efficiency measurement ─────────────────────────────────────────────────
    log.info("Measuring efficiency...")
    model, tokenizer = load_phi2()
    structure = get_model_structure(model)
    lens = load_tuned_lens(structure["hidden_dim"], structure["n_layers"])

    # v2 FIX: sample_answers are pre-known here, matching the realistic
    # deployment scenario (analyzing an EXISTING response). Passing these
    # explicitly avoids measure_efficiency() silently including a hidden
    # model.generate() call, which would misrepresent the single-pass claim.
    sample_prompts = [
        "Q: What is the capital of France?\nA:",
        "Q: Who wrote Romeo and Juliet?\nA:",
        "Q: What year did World War 2 end?\nA:",
        "Q: What is the chemical symbol for gold?\nA:",
        "Q: Who painted the Mona Lisa?\nA:",
    ]
    sample_answers = ["Paris", "William Shakespeare", "1945", "Au", "Leonardo da Vinci"]
    efficiency = measure_efficiency(model, tokenizer, lens, structure, sample_prompts, sample_answers)
    all_results["efficiency"] = efficiency

    # ── Causal validation summary (if available) ──────────────────────────────
    if os.path.exists(causal_path):
        causal_data = load_json(causal_path)
        all_results["causal"] = causal_data["summary"]
    else:
        log.warning("No causal patching results found — run Stage 4 first.")

    # ── Save and print ─────────────────────────────────────────────────────────
    save_json(all_results, output_path)
    print_full_results_table(all_results)

    log.info("✓ Stage 5 complete! Full results saved to evaluation/results/full_evaluation.json")


if __name__ == "__main__":
    main()
