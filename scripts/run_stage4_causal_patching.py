"""
run_stage4_causal_patching.py — Stage 4: Build matched pairs and run
the causal activation-patching validation experiment.

v5 CHANGE — adds --patch-mode CLI argument to select among three patching
strategies (all sharing the same matched-pairs pool and seeding so results
are directly comparable):

  single_layer  (default, existing baseline): patches only the single
                peak MLP layer — the v4 validated baseline (n=400 result
                already known: target=27.5%, random=11.6%, noise=35.5%).

  window        (new): patches MLP outputs at [peak-1, peak, peak+1]
                simultaneously. Tests whether spreading the patch over
                adjacent layers closes the target vs. noise gap.

  residual_stream (new): patches the full block output (attention + MLP)
                at the peak layer. Tests whether the causally relevant
                content is in the combined residual, not just the MLP write.

v3 FIXES (kept):
  - Passes the ALREADY-KNOWN answers from build_matched_pairs.py
    ("correct_model_answer" / "confab_model_answer") explicitly into
    every call, instead of leaving them to be regenerated. This avoids
    both a redundant model.generate() call AND a subtle mismatch risk
    (build_matched_pairs.py labels pairs using max_new_tokens=12, while
    extract_trajectory's own fallback generation uses max_new_tokens=8
    by default — regenerating could silently produce different text
    than what was used to label the pair as "confabulated" in the
    first place).
  - Uses the shared find_peak_confidence_layer() helper (now the single
    source of truth for "which layer is the recall event," used
    identically here and inside mlp_spike_alignment()).

Runtime: ~1-2 hours for ~100-200 matched pairs
Output:  evaluation/results/causal_patching_results_n{N}_s{seed}_{mode}.json
         (for single_layer mode, also writes the legacy causal_patching_results.json)

Usage:
    # Existing baseline (unchanged)
    python scripts/run_stage4_causal_patching.py --pairs 50 --patch-mode single_layer

    # New modes — exploratory pilot at n=120
    python scripts/run_stage4_causal_patching.py --pairs 120 --seed 42 --patch-mode window
    python scripts/run_stage4_causal_patching.py --pairs 120 --seed 42 --patch-mode residual_stream
"""

import argparse
import sys, os
import random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import get_logger, get_project_root, save_json, load_json
from src.instrumentation.model_loader import load_phi2, get_model_structure
from src.instrumentation.tuned_lens import load_tuned_lens
from src.instrumentation.trajectory_extractor import extract_trajectory
from src.features.shape_descriptors import find_peak_confidence_layer
from src.dataset.build_matched_pairs import build_matched_pairs, build_default_subject_list
from src.causal.activation_patching import (
    # Mode 1 — single-layer MLP (existing baseline)
    run_patching_experiment,
    run_specificity_check,
    run_noise_control_check,
    # Mode 2 — window MLP (new)
    run_window_patching_experiment,
    run_window_specificity_check,
    run_window_noise_control,
    # Mode 3 — residual stream (new)
    run_residual_stream_patching_experiment,
    run_residual_specificity_check,
    run_residual_noise_control,
)
from src.evaluation.benchmarks import summarize_causal_experiments

log = get_logger("script.stage4")


def main():
    parser = argparse.ArgumentParser(description="Stage 4: Causal activation patching validation")
    parser.add_argument("--pairs", type=int, default=100, help="Number of matched pairs to test")
    parser.add_argument("--pilot", action="store_true", help="Quick pilot with 20 pairs first")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for matched pair sampling and noise draws")
    parser.add_argument(
        "--patch-mode",
        choices=["single_layer", "window", "residual_stream"],
        default="single_layer",
        dest="patch_mode",
        help=(
            "Patching strategy: "
            "single_layer = existing baseline (peak MLP layer only); "
            "window = peak MLP layer ± 1 (3 layers simultaneously); "
            "residual_stream = full block output (attention + MLP) at peak layer"
        ),
    )
    args = parser.parse_args()

    n_pairs = 20 if args.pilot else args.pairs

    # Global seeding so all controls (including random-layer selection) are strictly reproducible
    random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)

    log.info("=" * 60)
    log.info("RTA Stage 4 — Causal Activation Patching Validation")
    log.info(f"Patch mode: {args.patch_mode}")
    log.info(f"Testing {n_pairs} matched pairs (seed={args.seed})")
    log.info("=" * 60)

    root = get_project_root()
    pairs_path    = str(root / "data" / "processed" / "matched_pairs.json")
    baseline_path = str(root / "data" / "trajectories" / "baseline_norms.json")

    # Mode-specific output filename — always includes mode so files never clash
    seeded_results_path = str(
        root / "evaluation" / "results"
        / f"causal_patching_results_n{n_pairs}_s{args.seed}_{args.patch_mode}.json"
    )
    # Legacy path — only written for single_layer to preserve backward compat
    legacy_results_path = str(root / "evaluation" / "results" / "causal_patching_results.json")

    # ── Load model once (shared between pair-building and patching) ───────────
    model, tokenizer = load_phi2()
    structure = get_model_structure(model)
    lens = load_tuned_lens(structure["hidden_dim"], structure["n_layers"])

    # ── Build or load matched pairs ───────────────────────────────────────────
    if os.path.exists(pairs_path):
        all_matched_pairs = load_json(pairs_path)
    else:
        subjects = build_default_subject_list()
        # Pass the already-loaded model in — avoids a second load_phi2() call
        # inside build_matched_pairs, which would double VRAM usage.
        all_matched_pairs = build_matched_pairs(
            subjects, max_pairs=n_pairs,
            save_path=pairs_path,
            model=model, tokenizer=tokenizer,
        )

    rng = random.Random(args.seed)
    if len(all_matched_pairs) > n_pairs:
        matched_pairs = rng.sample(all_matched_pairs, n_pairs)
    else:
        matched_pairs = all_matched_pairs[:n_pairs]
    log.info(
        f"Sampled {len(matched_pairs)} matched pairs from pool of "
        f"{len(all_matched_pairs)} using seed {args.seed}"
    )

    # Sanity check: warn (don't crash) if loading a STALE cache built before
    # build_matched_pairs.py started storing model_answer fields — those
    # pairs will fall back to slower, regeneration-based extraction.
    if matched_pairs and "correct_model_answer" not in matched_pairs[0]:
        log.warning(
            "Cached matched_pairs.json is missing 'correct_model_answer' / "
            "'confab_model_answer' fields (built with an older version of "
            "build_matched_pairs.py). Delete data/processed/matched_pairs.json "
            "and rerun to get the faster, consistency-guaranteed path."
        )

    baseline_norms = load_json(baseline_path)

    target_results       = []
    layer_control_results = []
    noise_control_results = []

    for i, pair in enumerate(matched_pairs):
        correct_answer = pair.get("correct_model_answer")
        confab_answer  = pair.get("confab_model_answer")

        # Get the correct example's OWN trajectory (using its already-known
        # answer, avoiding regeneration) to find the target layer.
        correct_traj = extract_trajectory(
            model, tokenizer, lens, structure, pair["correct_prompt"],
            answer_text=correct_answer,
        )
        target_layer = find_peak_confidence_layer(correct_traj.prob_trajectory)

        # ── Dispatch to the selected patching mode ────────────────────────────
        if args.patch_mode == "single_layer":
            result = run_patching_experiment(
                model, tokenizer, lens, structure,
                pair["correct_prompt"], pair["confab_prompt"],
                target_layer, baseline_norms,
                correct_answer_text=correct_answer, confab_answer_text=confab_answer,
            )
            controls = run_specificity_check(
                model, tokenizer, lens, structure,
                pair["correct_prompt"], pair["confab_prompt"],
                target_layer, baseline_norms,
                correct_answer_text=correct_answer, confab_answer_text=confab_answer,
                n_random_layers=2,
            )
            noise_result = run_noise_control_check(
                model, tokenizer, lens, structure,
                pair["correct_prompt"], pair["confab_prompt"],
                target_layer, baseline_norms,
                correct_answer_text=correct_answer, confab_answer_text=confab_answer,
                seed=args.seed * 10000 + i,
            )

        elif args.patch_mode == "window":
            result = run_window_patching_experiment(
                model, tokenizer, lens, structure,
                pair["correct_prompt"], pair["confab_prompt"],
                target_layer, baseline_norms,
                correct_answer_text=correct_answer, confab_answer_text=confab_answer,
            )
            controls = run_window_specificity_check(
                model, tokenizer, lens, structure,
                pair["correct_prompt"], pair["confab_prompt"],
                target_layer, baseline_norms,
                correct_answer_text=correct_answer, confab_answer_text=confab_answer,
                n_random_layers=2,
            )
            noise_result = run_window_noise_control(
                model, tokenizer, lens, structure,
                pair["correct_prompt"], pair["confab_prompt"],
                target_layer, baseline_norms,
                correct_answer_text=correct_answer, confab_answer_text=confab_answer,
                seed=args.seed * 10000 + i,
            )

        elif args.patch_mode == "residual_stream":
            result = run_residual_stream_patching_experiment(
                model, tokenizer, lens, structure,
                pair["correct_prompt"], pair["confab_prompt"],
                target_layer, baseline_norms,
                correct_answer_text=correct_answer, confab_answer_text=confab_answer,
            )
            controls = run_residual_specificity_check(
                model, tokenizer, lens, structure,
                pair["correct_prompt"], pair["confab_prompt"],
                target_layer, baseline_norms,
                correct_answer_text=correct_answer, confab_answer_text=confab_answer,
                n_random_layers=2,
            )
            noise_result = run_residual_noise_control(
                model, tokenizer, lens, structure,
                pair["correct_prompt"], pair["confab_prompt"],
                target_layer, baseline_norms,
                correct_answer_text=correct_answer, confab_answer_text=confab_answer,
                seed=args.seed * 10000 + i,
            )

        result["is_control"] = False
        target_results.append(result)
        layer_control_results.extend(controls)
        noise_control_results.append(noise_result)

        if i % 10 == 0:
            log.info(f"Processed {i}/{len(matched_pairs)} pairs")

    # ── Summarize (reuses existing summarize_causal_experiments unchanged) ─────
    summary = summarize_causal_experiments(
        target_results, layer_control_results, noise_control_results
    )

    results_data = {
        "patch_mode": args.patch_mode,
        "n_pairs": n_pairs,
        "seed": args.seed,
        "summary": summary,
        "target_results":        target_results,
        "layer_control_results": layer_control_results,
        "noise_control_results": noise_control_results,
    }

    save_json(results_data, seeded_results_path)
    log.info(f"Saved results to {seeded_results_path}")

    # Also write legacy path for backward compat (single_layer mode only)
    if args.patch_mode == "single_layer":
        save_json(results_data, legacy_results_path)

    # ── Print summary ─────────────────────────────────────────────────────────
    t_ci = f"[{summary['target_layer']['ci_lower']*100:.1f}%, {summary['target_layer']['ci_upper']*100:.1f}%]"
    l_ci = f"[{summary['layer_control']['ci_lower']*100:.1f}%, {summary['layer_control']['ci_upper']*100:.1f}%]"
    n_ci = f"[{summary['noise_control']['ci_lower']*100:.1f}%, {summary['noise_control']['ci_upper']*100:.1f}%]"

    log.info(f"\n✓ Stage 4 complete!  [patch_mode={args.patch_mode}, n={n_pairs}, seed={args.seed}]")
    log.info(
        f"Target-layer answer-change rate:       "
        f"{summary['target_layer']['answer_changed_rate']*100:.1f}% "
        f"(95% CI: {t_ci}, "
        f"confab drop: {summary['target_layer']['avg_confab_token_drop']:+.4f}, "
        f"mlp-align shift: {summary['target_layer']['avg_mlp_alignment_shift']:+.3f})"
    )
    log.info(
        f"Random-layer control change rate:      "
        f"{summary['layer_control']['answer_changed_rate']*100:.1f}% "
        f"(95% CI: {l_ci}, "
        f"confab drop: {summary['layer_control']['avg_confab_token_drop']:+.4f}, "
        f"mlp-align shift: {summary['layer_control']['avg_mlp_alignment_shift']:+.3f})"
    )
    log.info(
        f"Noise-vector control change rate:      "
        f"{summary['noise_control']['answer_changed_rate']*100:.1f}% "
        f"(95% CI: {n_ci}, "
        f"confab drop: {summary['noise_control']['avg_confab_token_drop']:+.4f}, "
        f"mlp-align shift: {summary['noise_control']['avg_mlp_alignment_shift']:+.3f})"
    )
    log.info("A successful validation shows target >> both controls.")
    log.info("Next step: python scripts/run_stage5_evaluate.py")


if __name__ == "__main__":
    main()
