import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.utils import get_logger, load_json, save_json, get_project_root
from src.instrumentation.model_loader import load_phi2, get_model_structure
from src.instrumentation.tuned_lens import load_tuned_lens
from src.instrumentation.trajectory_extractor import extract_trajectory
from src.features.shape_descriptors import find_peak_confidence_layer
from src.dataset.build_matched_pairs import build_matched_pairs, build_default_subject_list
from src.causal.activation_patching import (
    run_patching_experiment, run_noise_control_check
)

def main():
    root = get_project_root()
    baseline_path = str(root / "data" / "trajectories" / "baseline_norms.json")
    baseline_norms = load_json(baseline_path)

    model, tokenizer = load_phi2()
    structure = get_model_structure(model)
    lens = load_tuned_lens(structure["hidden_dim"], structure["n_layers"])

    print("Building up to 100 matched pairs from full subject list...")
    subjects = build_default_subject_list()
    # Build without the 20 cap
    matched_pairs = build_matched_pairs(
        subjects, max_pairs=100, save_path=None, model=model, tokenizer=tokenizer
    )
    n = len(matched_pairs)
    print(f"Total matched pairs available: {n}")

    # Step 4: Spot check the target layers for the first 10 pairs
    print("\n" + "=" * 60)
    print("STEP 4: Spot-check target layers (peak confidence jump)")
    print("=" * 60)
    target_layers = []
    for i in range(min(15, n)):
        pair = matched_pairs[i]
        c_ans = pair.get("correct_model_answer")
        traj = extract_trajectory(model, tokenizer, lens, structure, pair["correct_prompt"], answer_text=c_ans)
        tl = find_peak_confidence_layer(traj.prob_trajectory)
        target_layers.append(tl)
        # Check prob jumps around tl
        p_prev = traj.prob_trajectory[tl-1] if tl > 0 else 0.0
        p_at = traj.prob_trajectory[tl]
        p_next = traj.prob_trajectory[tl+1] if tl < len(traj.prob_trajectory)-1 else p_at
        p_final = traj.prob_trajectory[-1]
        print(f"Pair {i:2d} ({pair['relation']:<11}): {pair['correct_subject']:<20} -> L{tl:2d} | P[tl-1]={p_prev:.3f}, P[tl]={p_at:.3f}, P[tl+1]={p_next:.3f}, P[31]={p_final:.3f}")

    # Now run target vs noise comparison across all n pairs
    print("\n" + "=" * 60)
    print(f"STEP 1: Target vs Noise comparison at n={n} pairs")
    print("=" * 60)

    target_results = []
    noise_results = []

    for i, pair in enumerate(matched_pairs):
        c_ans = pair.get("correct_model_answer")
        f_ans = pair.get("confab_model_answer")

        correct_traj = extract_trajectory(
            model, tokenizer, lens, structure, pair["correct_prompt"], answer_text=c_ans
        )
        target_layer = find_peak_confidence_layer(correct_traj.prob_trajectory)

        # Target patch
        res_t = run_patching_experiment(
            model, tokenizer, lens, structure,
            pair["correct_prompt"], pair["confab_prompt"],
            target_layer, baseline_norms,
            correct_answer_text=c_ans, confab_answer_text=f_ans,
            control_mode="captured"
        )
        target_results.append(res_t)

        # Noise patch
        res_n = run_noise_control_check(
            model, tokenizer, lens, structure,
            pair["correct_prompt"], pair["confab_prompt"],
            target_layer, baseline_norms,
            correct_answer_text=c_ans, confab_answer_text=f_ans
        )
        noise_results.append(res_n)

        if (i + 1) % 10 == 0 or (i + 1) == n:
            print(f"Processed {i+1}/{n} pairs...")

    # Summary
    def summarize(res_list):
        k = len(res_list)
        chg = sum(1 for r in res_list if r["answer_changed"])
        flips = sum(1 for r in res_list if r["argmax_flipped"])
        drop = np.mean([r["confab_token_drop"] for r in res_list])
        gain = np.mean([r["correct_token_gain"] for r in res_list])
        align = np.mean([r["mlp_alignment_shift"] for r in res_list])
        return {
            "n": k,
            "answer_changed_rate": chg / k,
            "argmax_flip_rate": flips / k,
            "avg_confab_token_drop": float(drop),
            "avg_correct_token_gain": float(gain),
            "avg_mlp_align_shift": float(align),
        }

    s_target = summarize(target_results)
    s_noise = summarize(noise_results)

    print("\n" + "=" * 60)
    print("TARGET VS NOISE SUMMARY AT HIGHER N:")
    print("=" * 60)
    print(f"{'Metric':<25} {'Target (Real)':>15} {'Noise Control':>15} {'Difference (T - N)':>20}")
    print("-" * 75)
    print(f"{'Answer Changed Rate':<25} {s_target['answer_changed_rate']*100:>14.1f}% {s_noise['answer_changed_rate']*100:>14.1f}% {(s_target['answer_changed_rate'] - s_noise['answer_changed_rate'])*100:>19.1f}%")
    print(f"{'Argmax Flip Rate':<25} {s_target['argmax_flip_rate']*100:>14.1f}% {s_noise['argmax_flip_rate']*100:>14.1f}% {(s_target['argmax_flip_rate'] - s_noise['argmax_flip_rate'])*100:>19.1f}%")
    print(f"{'Avg Confab Drop':<25} {s_target['avg_confab_token_drop']:>15.4f} {s_noise['avg_confab_token_drop']:>15.4f} {s_target['avg_confab_token_drop'] - s_noise['avg_confab_token_drop']:>20.4f}")
    print(f"{'Avg Correct Token Gain':<25} {s_target['avg_correct_token_gain']:>15.4f} {s_noise['avg_correct_token_gain']:>15.4f} {s_target['avg_correct_token_gain'] - s_noise['avg_correct_token_gain']:>20.4f}")
    print(f"{'Avg MLP-Align Shift':<25} {s_target['avg_mlp_align_shift']:>15.4f} {s_noise['avg_mlp_align_shift']:>15.4f} {s_target['avg_mlp_align_shift'] - s_noise['avg_mlp_align_shift']:>20.4f}")

if __name__ == "__main__":
    main()
