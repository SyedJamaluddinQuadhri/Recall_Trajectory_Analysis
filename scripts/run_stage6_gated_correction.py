"""
run_stage6_gated_correction.py — Stage 6: Gated Correction Experiment.

DESIGN DECISION — PRIMARY MODEL:
    The paper-primary classifier is the 6-feature MLP (crystallization,
    smoothness, mlp_alignment, early_presence, final_entropy,
    final_rank_normalized).  Its trained weights live at
    models/trajectory_classifier/classifier.pt and are loaded directly via
    TrajectoryClassifierPredictor — no retraining, no CV fold selection.
    The 128-feat model (0.7920 AUROC) is the higher-accuracy alternative
    documented in stage3_results.md but requires CV-fold retraining and has
    no single persisted checkpoint; only the 6-feat checkpoint exists on disk.
    We therefore use the 6-feat MLP as the gate and scorer for all experiments
    in this script.

CORRECTION STRATEGY — RESAMPLE + PICK-BEST:
    For each query that exceeds threshold tau:
      1. Re-generate the answer k=5 times at temperature=0.8 from the same
         prompt (sampling, not greedy).
      2. Run the full Stage 1->3 pipeline (trajectory extraction -> 6 features
         -> classifier score) on each candidate.
      3. Select the candidate with the lowest confabulation probability.

TEMPERATURE CHOICE — 0.8:
    We fix temperature at 0.8 (rather than sweeping it).  This choice is
    justified as follows:
      - 0.7 is close enough to greedy that the k samples are nearly
        deterministic and add little diversity.
      - 1.0 produces genuinely random token distributions that can degrade
        factual grounding in short-answer QA.
      - 0.8 is the established sampling temperature for open-ended factual QA
        in the LLM literature (used by, e.g., Kadavath et al. 2022) and
        empirically balances answer diversity with factual coherence.

TRAIN/TEST SPLIT DISCIPLINE:
    We reproduce the EXACT same 80/20 stratified split used by
    train_classifier() (sklearn train_test_split, stratify=y, random_state=42)
    and report ALL metrics on the held-out 20% test fold only.

MEASUREMENTS (per threshold tau in {0.3, 0.4, 0.5, 0.6, 0.7}):
    - Confabulation fix rate
    - Correct-answer disturbance rate (core asymmetry number)
    - Trigger rate
    - Net accuracy change
    - Ungated baseline (tau=0.0): always correct, to isolate gate value

EFFICIENCY REPORTING:
    Per-query latency and peak VRAM for single-pass vs. correction pipeline.
    VRAM checks confirm sequential (not batched) k-resampling does not grow.

OUTPUT:
    evaluation/results/stage6_gated_correction.json          - per-example log
    evaluation/results/stage6_gated_correction_summary.json  - aggregate table

Usage:
    python scripts/run_stage6_gated_correction.py
    python scripts/run_stage6_gated_correction.py --k 5 --temperature 0.8
    python scripts/run_stage6_gated_correction.py --max-examples 200  # quick test
    python scripts/run_stage6_gated_correction.py --skip-efficiency
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Force UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import torch
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import get_logger, get_project_root, load_json, load_pickle, save_json
from src.instrumentation.model_loader import load_phi2, get_model_structure
from src.instrumentation.tuned_lens import load_tuned_lens
from src.instrumentation.trajectory_extractor import extract_trajectory
from src.features.shape_descriptors import extract_all_descriptors
from src.classifier.trajectory_classifier import TrajectoryClassifierPredictor

log = get_logger("script.stage6_gated_correction")

ROOT = get_project_root()
RESULTS_DIR = ROOT / "evaluation" / "results"

# Threshold sweep — includes 0.0 (ungated baseline) for comparison
THRESHOLDS = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7]


# ---------------------------------------------------------------------------
# Exact-match (must stay in sync with build_factual_qa.py)
# ---------------------------------------------------------------------------

def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation and articles for exact-match comparison."""
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(predicted: str, gold_answers: List[str]) -> bool:
    """Check if predicted answer matches any gold alias (bidirectional substring)."""
    pred_norm = normalize_answer(predicted)
    for gold in gold_answers:
        gold_norm = normalize_answer(gold)
        if gold_norm and (gold_norm in pred_norm or pred_norm in gold_norm):
            return True
    return False


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def load_pipeline(device: str):
    """Load Phi-2, tuned lens, and model structure."""
    log.info("Loading Phi-2 + tuned lens...")
    model, tokenizer = load_phi2(device=device)
    structure = get_model_structure(model)
    lens = load_tuned_lens(structure["hidden_dim"], structure["n_layers"], device=device)
    log.info("Pipeline loaded.")
    return model, tokenizer, lens, structure


def score_answer(
    model, tokenizer, lens, structure,
    prompt: str,
    answer_text: str,
    predictor: TrajectoryClassifierPredictor,
    baseline_norms: dict,
    max_answer_tokens: int = 8,
) -> Tuple[float, dict]:
    """
    Run the full Stage 1->3 pipeline on one (prompt, answer) pair.
    Returns (confab_prob, features_dict).
    """
    traj = extract_trajectory(
        model, tokenizer, lens, structure,
        prompt=prompt, answer_text=answer_text,
        max_answer_tokens=max_answer_tokens,
    )
    feats = extract_all_descriptors(traj, baseline_norms)
    prob  = predictor.predict(feats)
    return prob, feats


def generate_candidate_answers(
    model,
    tokenizer,
    prompt: str,
    k: int = 5,
    temperature: float = 0.8,
    max_new_tokens: int = 12,
    device: str = "cpu",
) -> List[str]:
    """
    Generate k candidate answers via temperature sampling.

    Candidates are generated SEQUENTIALLY — one at a time — so VRAM stays at
    the baseline single-sample footprint throughout (no batch accumulation).
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    candidates = []
    with torch.no_grad():
        for _ in range(k):
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,
            )
            new_tokens = out[0][inputs["input_ids"].shape[1]:]
            answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
            answer = answer.split("\n")[0].strip()
            if answer:
                candidates.append(answer)
    return candidates


def pick_best_candidate(
    candidates: List[str],
    prompt: str,
    model, tokenizer, lens, structure,
    predictor: TrajectoryClassifierPredictor,
    baseline_norms: dict,
    max_answer_tokens: int = 8,
) -> Tuple[str, float, List[dict]]:
    """
    Score each candidate and return the one with lowest confab probability.
    Returns (best_answer, best_prob, all_candidate_scores).
    """
    best_answer = None
    best_prob   = float("inf")
    cand_scores = []

    for cand in candidates:
        prob, feats = score_answer(
            model, tokenizer, lens, structure,
            prompt, cand, predictor, baseline_norms, max_answer_tokens,
        )
        cand_scores.append({"answer": cand, "confab_prob": prob})
        if prob < best_prob:
            best_prob   = prob
            best_answer = cand

    return best_answer, best_prob, cand_scores


# ---------------------------------------------------------------------------
# VRAM helpers
# ---------------------------------------------------------------------------

def _vram_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e6
    return 0.0


def _vram_current_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e6
    return 0.0


def _reset_vram_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# Test fold reproduction
# ---------------------------------------------------------------------------

def load_test_fold_indices(n_total: int, y: np.ndarray) -> np.ndarray:
    """
    Reproduce the EXACT 20% test fold from train_classifier().
    Uses train_test_split(test_size=0.2, stratify=y, random_state=42).
    """
    idx = np.arange(n_total)
    _, test_idx = train_test_split(idx, test_size=0.2, stratify=y, random_state=42)
    return test_idx


# ---------------------------------------------------------------------------
# Aggregate metrics for a single threshold (no GPU needed)
# ---------------------------------------------------------------------------

def compute_threshold_metrics(per_example: List[dict], tau: float) -> dict:
    """
    Compute all aggregate metrics for a given threshold from the pre-computed
    per-example results.  Pure Python — no model inference.

    The per-example list must contain 'corrected_answer' and 'corrected_is_correct'
    (the pick-best answer/correctness for EVERY example, regardless of threshold)
    so we can apply any threshold post-hoc without re-running inference.
    """
    n_total = len(per_example)
    n_originally_correct = sum(1 for e in per_example if e["originally_correct"])
    n_originally_confab  = n_total - n_originally_correct

    triggered     = [e for e in per_example if e["original_confab_prob"] >= tau]
    not_triggered = [e for e in per_example if e["original_confab_prob"] <  tau]
    n_triggered   = len(triggered)
    trigger_rate  = n_triggered / n_total if n_total > 0 else 0.0

    # Fix rate: among triggered confabulated, fraction corrected
    triggered_confab = [e for e in triggered if not e["originally_correct"]]
    if triggered_confab:
        fixed    = [e for e in triggered_confab if e["corrected_is_correct"]]
        fix_rate = len(fixed) / len(triggered_confab)
        n_fixed  = len(fixed)
    else:
        fix_rate = 0.0
        n_fixed  = 0

    # Disturbance rate: among triggered correct, fraction now wrong
    triggered_correct = [e for e in triggered if e["originally_correct"]]
    if triggered_correct:
        disturbed    = [e for e in triggered_correct if not e["corrected_is_correct"]]
        disturb_rate = len(disturbed) / len(triggered_correct)
        n_disturbed  = len(disturbed)
    else:
        disturb_rate = 0.0
        n_disturbed  = 0

    # Net accuracy
    acc_before = n_originally_correct / n_total if n_total > 0 else 0.0
    n_correct_after = 0
    for e in per_example:
        if e["original_confab_prob"] < tau:
            n_correct_after += int(e["originally_correct"])
        else:
            n_correct_after += int(e["corrected_is_correct"])
    acc_after = n_correct_after / n_total if n_total > 0 else 0.0

    return {
        "tau":                    tau,
        "n_total":                n_total,
        "n_originally_correct":  n_originally_correct,
        "n_originally_confab":   n_originally_confab,
        "n_triggered":            n_triggered,
        "trigger_rate":           trigger_rate,
        "n_triggered_confab":     len(triggered_confab),
        "n_triggered_correct":    len(triggered_correct),
        "fix_rate":               fix_rate,
        "n_fixed":                n_fixed,
        "disturbance_rate":       disturb_rate,
        "n_disturbed":            n_disturbed,
        "accuracy_before":        acc_before,
        "accuracy_after":         acc_after,
        "net_accuracy_change":    acc_after - acc_before,
    }


# ---------------------------------------------------------------------------
# Efficiency measurement
# ---------------------------------------------------------------------------

def measure_single_pass_efficiency(
    model, tokenizer, lens, structure,
    predictor, baseline_norms,
    sample_prompts, sample_answers, device,
) -> dict:
    """Latency + peak VRAM for Stage 1->3 single-pass detection (no correction)."""
    latencies = []
    _reset_vram_peak()
    for prompt, answer in zip(sample_prompts, sample_answers):
        t0 = time.perf_counter()
        score_answer(model, tokenizer, lens, structure,
                     prompt, answer, predictor, baseline_norms)
        latencies.append(time.perf_counter() - t0)
    return {
        "mean_latency_s": float(np.mean(latencies)),
        "std_latency_s":  float(np.std(latencies)),
        "peak_vram_mb":   _vram_mb(),
        "n_samples":      len(latencies),
    }


def measure_correction_efficiency(
    model, tokenizer, lens, structure,
    predictor, baseline_norms,
    sample_prompts, sample_answers,
    k, temperature, device,
) -> dict:
    """
    Latency + peak VRAM for the full correction pipeline (k-resample + pick-best).
    Also checks that VRAM does not grow between samples (confirming sequential
    sampling does not accumulate allocations).
    """
    latencies = []
    vram_checkpoints = []
    _reset_vram_peak()
    for prompt, answer in zip(sample_prompts, sample_answers):
        t0 = time.perf_counter()
        candidates = generate_candidate_answers(
            model, tokenizer, prompt, k=k,
            temperature=temperature, device=device,
        )
        if not candidates:
            candidates = [answer]
        pick_best_candidate(
            candidates, prompt, model, tokenizer, lens, structure,
            predictor, baseline_norms,
        )
        latencies.append(time.perf_counter() - t0)
        vram_checkpoints.append(_vram_current_mb())

    peak_vram_mb = _vram_mb()
    vram_growth  = 0.0
    if len(vram_checkpoints) >= 2:
        diffs = [abs(vram_checkpoints[i] - vram_checkpoints[i-1])
                 for i in range(1, len(vram_checkpoints))]
        vram_growth = float(max(diffs))

    return {
        "mean_latency_s":      float(np.mean(latencies)),
        "std_latency_s":       float(np.std(latencies)),
        "peak_vram_mb":        peak_vram_mb,
        "max_vram_growth_mb":  vram_growth,
        "vram_growth_flagged": vram_growth > 50.0,
        "n_samples":           len(latencies),
        "k":                   k,
        "temperature":         temperature,
    }


# ---------------------------------------------------------------------------
# Threshold recommendation
# ---------------------------------------------------------------------------

def select_recommended_threshold(threshold_results: List[dict]) -> dict:
    """
    Criterion: best net accuracy gain among thresholds with disturbance_rate < 0.10.
    Rationale: we prefer gates that do not significantly disturb correct answers
    (< 10% disturbance) while still improving overall accuracy.
    If no threshold meets the disturbance constraint, fall back to best net gain.
    The ungated (tau=0.0) entry is excluded from selection.
    """
    candidates = [r for r in threshold_results if r["tau"] > 0.0]
    if not candidates:
        return {"recommended_tau": None, "selection_criterion": "no_candidates"}

    low_disturbance = [r for r in candidates if r["disturbance_rate"] < 0.10]
    if low_disturbance:
        best      = max(low_disturbance, key=lambda r: r["net_accuracy_change"])
        criterion = "best_net_accuracy_gain_with_disturbance_rate_lt_10pct"
    else:
        best      = max(candidates, key=lambda r: r["net_accuracy_change"])
        criterion = "best_net_accuracy_gain_fallback_no_low_disturbance_threshold"

    return {
        "recommended_tau":                    best["tau"],
        "fix_rate_at_tau_star":               best["fix_rate"],
        "disturbance_rate_at_tau_star":       best["disturbance_rate"],
        "net_accuracy_change_at_tau_star":    best["net_accuracy_change"],
        "trigger_rate_at_tau_star":           best["trigger_rate"],
        "selection_criterion":                criterion,
    }


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(
    threshold_results: List[dict],
    recommendation: dict,
    efficiency: dict,
    ungated_metrics: dict,
    k: int,
    temperature: float,
):
    sep = "=" * 90
    print(f"\n{sep}")
    print("STAGE 6 - GATED CORRECTION EXPERIMENT: DETECTION-CORRECTION ASYMMETRY")
    print(sep)
    print(f"  Classifier: 6-feature MLP (paper-primary, AUROC baseline 0.7104 / tuned 0.7284)")
    print(f"  Correction: resample k={k} at temperature={temperature}, pick lowest confab prob")
    print(f"  Dataset:    held-out 20% test fold (random_state=42, same as classifier training)\n")

    # Threshold sweep table
    hdr = (f"{'tau':>5}  {'Trigger%':>9}  {'Fix%':>7}  {'Disturb%':>9}  "
           f"{'Acc Before':>10}  {'Acc After':>9}  {'Net DAcc':>9}")
    print(hdr)
    print("-" * 90)
    for r in threshold_results:
        tag = "  <- UNGATED (tau=0)" if r["tau"] == 0.0 else ""
        print(
            f"{r['tau']:>5.1f}  "
            f"{r['trigger_rate']*100:>8.1f}%  "
            f"{r['fix_rate']*100:>6.1f}%  "
            f"{r['disturbance_rate']*100:>8.1f}%  "
            f"{r['accuracy_before']*100:>9.2f}%  "
            f"{r['accuracy_after']*100:>8.2f}%  "
            f"{r['net_accuracy_change']*100:>+8.2f}%"
            f"{tag}"
        )

    print("\nCOLUMN NOTES:")
    print("  tau      = confabulation probability threshold for triggering correction")
    print("  Trigger% = fraction of test examples sent to k-resample at this tau")
    print("  Fix%     = of triggered confabulated examples, fraction that became correct")
    print("  Disturb% = of triggered correct examples, fraction that became incorrect")
    print("  Net DAcc = overall accuracy after correction minus accuracy before")
    print("  UNGATED  = tau=0.0, i.e. ALL examples corrected regardless of score")

    # Fix-rate vs Disturbance-rate curve
    print(f"\n{'─'*70}")
    print("FIX-RATE vs. DISTURBANCE-RATE CURVE (precision-style, tau sweep):")
    print(f"  {'tau':>5}  {'Fix%':>7}  {'Disturb%':>9}")
    for r in threshold_results:
        if r["tau"] > 0.0:
            print(f"  {r['tau']:>5.1f}  {r['fix_rate']*100:>6.1f}%  {r['disturbance_rate']*100:>8.1f}%")

    # Recommendation
    print(f"\n{'─'*70}")
    print("RECOMMENDED OPERATING THRESHOLD:")
    tau_star = recommendation.get("recommended_tau")
    if tau_star is not None:
        print(f"  tau* = {tau_star}")
        print(f"  Fix rate:         {recommendation['fix_rate_at_tau_star']*100:.1f}%")
        print(f"  Disturbance rate: {recommendation['disturbance_rate_at_tau_star']*100:.1f}%")
        print(f"  Net accuracy D:   {recommendation['net_accuracy_change_at_tau_star']*100:+.2f}%")
        print(f"  Trigger rate:     {recommendation['trigger_rate_at_tau_star']*100:.1f}%")
        print(f"  Selection criterion: {recommendation['selection_criterion']}")
    else:
        print("  No threshold recommended.")

    # Ungated comparison
    print(f"\n{'─'*70}")
    print("UNGATED BASELINE COMPARISON (tau=0.0, always correct):")
    print(f"  Fix rate:         {ungated_metrics['fix_rate']*100:.1f}%")
    print(f"  Disturbance rate: {ungated_metrics['disturbance_rate']*100:.1f}%")
    print(f"  Net accuracy D:   {ungated_metrics['net_accuracy_change']*100:+.2f}%")

    if tau_star is not None:
        gate_value = (
            recommendation["net_accuracy_change_at_tau_star"]
            - ungated_metrics["net_accuracy_change"]
        )
        print(f"  Gate value (gated minus ungated net D): {gate_value*100:+.2f}%")
        if gate_value > 0.005:
            print("  FINDING: The gate PROVIDES measurable benefit over indiscriminate correction.")
        elif gate_value < -0.005:
            print("  FINDING: The gate DOES NOT protect enough -- ungated performs better.")
            print("  Report this honestly: RTA score is not precise enough to serve as a correction gate at this resolution.")
        else:
            print("  FINDING: Gated and ungated produce comparable results -- gate has minimal effect.")

    # Efficiency
    print(f"\n{'─'*70}")
    print("EFFICIENCY:")
    sp = efficiency.get("single_pass_baseline", {})
    ce = efficiency.get("correction_efficiency", {})
    if sp:
        print(f"  Single-pass (no correction):  {sp['mean_latency_s']:.3f}s +/- {sp['std_latency_s']:.3f}s  |  peak VRAM {sp['peak_vram_mb']:.0f} MB")
    if ce:
        print(f"  Correction pipeline (k={ce['k']}):    {ce['mean_latency_s']:.3f}s +/- {ce['std_latency_s']:.3f}s  |  peak VRAM {ce['peak_vram_mb']:.0f} MB")
        grow = ce.get("max_vram_growth_mb", 0)
        if ce.get("vram_growth_flagged"):
            print(f"  WARNING: VRAM growth between samples = {grow:.0f} MB -- investigate sequential sampling")
        else:
            print(f"  VRAM growth between samples = {grow:.1f} MB (within expected noise -- sequential sampling confirmed OK)")
    if "estimated_triggered_query_latency_s" in efficiency:
        print(f"  Triggered query (gate hit):   ~{efficiency['estimated_triggered_query_latency_s']:.2f}s")
        print(f"  Non-triggered query:          ~{efficiency['estimated_nontriggered_query_latency_s']:.3f}s")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage 6: Gated Correction Experiment")
    parser.add_argument("--k",                  type=int,   default=5,   help="Resample candidates (default: 5)")
    parser.add_argument("--temperature",         type=float, default=0.8, help="Sampling temperature (default: 0.8)")
    parser.add_argument("--max-examples",        type=int,   default=None,help="Cap test examples for quick testing")
    parser.add_argument("--device",              type=str,   default=None,help="Device: cuda / cpu (auto-detect)")
    parser.add_argument("--efficiency-samples",  type=int,   default=5,   help="Samples for efficiency measurement (default: 5)")
    parser.add_argument("--skip-efficiency",     action="store_true",     help="Skip efficiency measurement")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    log.info("=" * 70)
    log.info("RTA Stage 6 -- Gated Correction Experiment")
    log.info(f"  k={args.k}  temperature={args.temperature}  device={device}")
    log.info("  Classifier: 6-feature MLP (paper-primary, loaded from checkpoint)")
    log.info("  Correction gating: threshold sweep tau in " + str(THRESHOLDS))
    log.info("=" * 70)

    # -- Load data ----------------------------------------------------------
    traj_path     = str(ROOT / "data" / "trajectories" / "all_trajectories.pkl")
    baseline_path = str(ROOT / "data" / "trajectories" / "baseline_norms.json")
    labeled_path  = str(ROOT / "data" / "processed" / "labeled_qa.json")

    log.info("Loading pre-computed trajectories and labeled dataset...")
    traj_data      = load_pickle(traj_path)
    baseline_norms = load_json(baseline_path)
    labeled_qa     = load_json(labeled_path)

    if traj_data is None:
        raise RuntimeError(
            "No trajectories found at data/trajectories/all_trajectories.pkl. "
            "Run scripts/run_stage2_build_dataset.py first."
        )

    all_trajs  = traj_data["trajectories"]
    all_labels = traj_data["labels"]   # 1 = confabulated, 0 = correct
    n_total    = len(all_trajs)

    log.info(f"  Loaded {n_total} examples  "
             f"(confab={sum(all_labels)}, correct={n_total - sum(all_labels)})")

    if len(labeled_qa) != n_total:
        raise RuntimeError(
            f"labeled_qa.json has {len(labeled_qa)} entries but "
            f"all_trajectories.pkl has {n_total}. They must correspond 1:1."
        )

    # -- Reproduce EXACT test fold ------------------------------------------
    y_all    = np.array(all_labels, dtype=np.float32)
    test_idx = load_test_fold_indices(n_total, y_all)

    test_trajs   = [all_trajs[i]  for i in test_idx]
    test_labels  = [all_labels[i] for i in test_idx]
    test_labeled = [labeled_qa[i] for i in test_idx]
    n_test       = len(test_idx)

    log.info(f"  Test fold: {n_test} examples  "
             f"(confab={sum(test_labels)}, correct={n_test - sum(test_labels)})")

    if args.max_examples is not None:
        cap          = args.max_examples
        test_trajs   = test_trajs[:cap]
        test_labels  = test_labels[:cap]
        test_labeled = test_labeled[:cap]
        n_test       = len(test_trajs)
        log.info(f"  (Capped to {n_test} examples via --max-examples)")

    # -- Load pipeline + classifier -----------------------------------------
    model, tokenizer, lens, structure = load_pipeline(device)
    predictor = TrajectoryClassifierPredictor()
    log.info("  6-feature MLP classifier loaded from checkpoint.")

    # -- PASS 1: Score every test example using pre-computed trajectories ----
    # We use the pre-stored trajectories for the original (greedy) answers.
    # No new model.generate() or forward pass needed for original scoring.
    log.info("Pass 1: Scoring ALL test examples (Stage 2 features -> Stage 3 classifier)...")
    log.info("  (Using pre-computed trajectories -- no new GPU inference for original answers)")

    original_scores = []
    for i, traj in enumerate(test_trajs):
        feats = extract_all_descriptors(traj, baseline_norms)
        prob  = predictor.predict(feats)
        original_scores.append(prob)
        if i > 0 and i % 200 == 0:
            log.info(f"  Scored {i}/{n_test}")

    log.info(
        f"  Done. Scores: min={min(original_scores):.3f} "
        f"mean={sum(original_scores)/len(original_scores):.3f} "
        f"max={max(original_scores):.3f}"
    )

    # -- PASS 2: Generate k candidates + score for EVERY test example --------
    # We generate candidates for ALL examples upfront regardless of tau.
    # Post-hoc threshold sweep is then free (no GPU needed).
    log.info(
        f"Pass 2: Generating k={args.k} candidates + scoring for ALL {n_test} test examples..."
    )
    log.info(
        f"  temperature={args.temperature}, max_new_tokens=12, sequential (no batching)"
    )
    log.info(
        "  This is the most expensive pass.  VRAM will be monitored per example."
    )

    per_example_results = []
    vram_peak_list      = []
    t_pass2_start       = time.time()

    for i, (traj, label, qa_item, orig_prob) in enumerate(
        zip(test_trajs, test_labels, test_labeled, original_scores)
    ):
        t_ex = time.perf_counter()
        _reset_vram_peak()

        prompt             = f"Q: {qa_item['question']}\nA:"
        original_answer    = qa_item["model_answer"]
        gold_answers       = qa_item["gold_answers"]
        originally_correct = (label == 0)

        # Generate k candidate answers at temperature=0.8
        candidates = generate_candidate_answers(
            model, tokenizer, prompt,
            k=args.k, temperature=args.temperature,
            max_new_tokens=12, device=device,
        )
        if not candidates:
            # Fallback: use original answer (can happen if model returns empty)
            candidates = [original_answer]

        # Score each candidate and pick the lowest-confab-prob one
        best_answer, best_prob, cand_scores = pick_best_candidate(
            candidates, prompt,
            model, tokenizer, lens, structure,
            predictor, baseline_norms, max_answer_tokens=8,
        )

        corrected_is_correct = exact_match(best_answer, gold_answers)
        latency              = time.perf_counter() - t_ex
        peak_vram            = _vram_mb()
        vram_peak_list.append(peak_vram)

        per_example_results.append({
            "example_idx":           int(test_idx[i]),
            "question":              qa_item["question"],
            "gold_answers":          gold_answers,
            "source":                qa_item.get("source", ""),
            # Original greedy answer
            "original_answer":       original_answer,
            "original_confab_prob":  float(orig_prob),
            "originally_correct":    bool(originally_correct),
            # Pick-best corrected answer
            "corrected_answer":      best_answer,
            "corrected_confab_prob": float(best_prob),
            "corrected_is_correct":  bool(corrected_is_correct),
            # All k candidates with scores
            "candidates": [
                {"answer": c["answer"], "confab_prob": float(c["confab_prob"])}
                for c in cand_scores
            ],
            # Per-example efficiency
            "latency_s":     latency,
            "peak_vram_mb": peak_vram,
        })

        if i % 50 == 0 or i == n_test - 1:
            elapsed = time.time() - t_pass2_start
            rate    = (i + 1) / max(elapsed, 1e-9)
            eta_min = (n_test - i - 1) / max(rate, 1e-9) / 60
            log.info(
                f"  [{i+1}/{n_test}] "
                f"orig_prob={orig_prob:.3f} best_prob={best_prob:.3f} "
                f"was_correct={originally_correct} "
                f"corrected_correct={corrected_is_correct} "
                f"vram={peak_vram:.0f}MB  eta={eta_min:.1f}min"
            )

    t_pass2_elapsed = time.time() - t_pass2_start
    log.info(f"  Pass 2 complete in {t_pass2_elapsed/60:.1f} minutes")
    log.info(
        f"  VRAM per example: mean={np.mean(vram_peak_list):.1f}MB "
        f"max={max(vram_peak_list):.1f}MB"
    )

    # -- Threshold sweep (no GPU) -------------------------------------------
    log.info("Sweeping thresholds...")
    threshold_results = []
    for tau in THRESHOLDS:
        metrics = compute_threshold_metrics(per_example_results, tau)
        threshold_results.append(metrics)
        log.info(
            f"  tau={tau:.1f}: trigger={metrics['trigger_rate']*100:.1f}% "
            f"fix={metrics['fix_rate']*100:.1f}% "
            f"disturb={metrics['disturbance_rate']*100:.1f}% "
            f"net_acc={metrics['net_accuracy_change']*100:+.2f}%"
        )

    ungated_metrics = next(r for r in threshold_results if r["tau"] == 0.0)
    recommendation  = select_recommended_threshold(threshold_results)
    log.info(
        f"  Recommended tau* = {recommendation.get('recommended_tau')} "
        f"({recommendation['selection_criterion']})"
    )

    # -- Efficiency measurement ---------------------------------------------
    efficiency = {}
    if not args.skip_efficiency:
        log.info("Measuring efficiency...")
        n_eff = min(args.efficiency_samples, n_test)
        eff_prompts = [f"Q: {test_labeled[i]['question']}\nA:" for i in range(n_eff)]
        eff_answers = [test_labeled[i]["model_answer"]          for i in range(n_eff)]

        log.info(f"  Single-pass efficiency ({n_eff} samples)...")
        sp_eff = measure_single_pass_efficiency(
            model, tokenizer, lens, structure,
            predictor, baseline_norms, eff_prompts, eff_answers, device,
        )
        efficiency["single_pass_baseline"] = sp_eff
        log.info(
            f"  Single-pass: {sp_eff['mean_latency_s']:.3f}s  "
            f"VRAM={sp_eff['peak_vram_mb']:.0f}MB"
        )

        log.info(f"  Correction pipeline efficiency ({n_eff} samples, k={args.k})...")
        corr_eff = measure_correction_efficiency(
            model, tokenizer, lens, structure,
            predictor, baseline_norms, eff_prompts, eff_answers,
            k=args.k, temperature=args.temperature, device=device,
        )
        efficiency["correction_efficiency"] = corr_eff
        log.info(
            f"  Correction: {corr_eff['mean_latency_s']:.3f}s  "
            f"VRAM={corr_eff['peak_vram_mb']:.0f}MB  "
            f"growth={corr_eff['max_vram_growth_mb']:.1f}MB  "
            f"flagged={corr_eff['vram_growth_flagged']}"
        )

        efficiency["estimated_triggered_query_latency_s"] = (
            sp_eff["mean_latency_s"] + corr_eff["mean_latency_s"]
        )
        efficiency["estimated_nontriggered_query_latency_s"] = sp_eff["mean_latency_s"]

        budget_mb = 3000.0
        if corr_eff["peak_vram_mb"] > budget_mb:
            log.warning(
                f"Peak VRAM during correction ({corr_eff['peak_vram_mb']:.0f} MB) "
                f"EXCEEDS the 3 GB budget!"
            )
        else:
            log.info(
                f"  VRAM budget check: {corr_eff['peak_vram_mb']:.0f} MB < {budget_mb:.0f} MB -- OK"
            )
    else:
        log.info("  --skip-efficiency: efficiency measurement skipped")

    # -- Print summary ------------------------------------------------------
    print_summary(
        threshold_results, recommendation, efficiency, ungated_metrics,
        k=args.k, temperature=args.temperature,
    )

    # -- Save full per-example log ------------------------------------------
    os.makedirs(str(RESULTS_DIR), exist_ok=True)
    full_path = str(RESULTS_DIR / "stage6_gated_correction.json")
    save_json({
        "metadata": {
            "k":                 args.k,
            "temperature":       args.temperature,
            "thresholds_swept":  THRESHOLDS,
            "n_test":            n_test,
            "device":            device,
            "classifier":        "6-feature MLP (models/trajectory_classifier/classifier.pt)",
            "split_discipline":  "train_test_split(test_size=0.2, stratify=y, random_state=42)",
            "note":              (
                "Pass 1 uses pre-stored trajectories for original-answer scoring. "
                "Pass 2 runs model.generate(temperature=0.8) k times per example "
                "then extract_trajectory + Stage 3 classifier on each candidate."
            ),
        },
        "per_example": per_example_results,
    }, full_path)
    log.info(f"Full per-example log -> {full_path}")

    # -- Save aggregate summary --------------------------------------------
    summary_path = str(RESULTS_DIR / "stage6_gated_correction_summary.json")
    save_json({
        "metadata": {
            "k":              args.k,
            "temperature":    args.temperature,
            "n_test":         n_test,
            "classifier":     "6-feature MLP",
            "thresholds_swept": THRESHOLDS,
        },
        "threshold_results": threshold_results,
        "ungated_metrics":   ungated_metrics,
        "recommendation":    recommendation,
        "efficiency":        efficiency,
    }, summary_path)
    log.info(f"Aggregate summary -> {summary_path}")

    log.info("Stage 6 complete.")


if __name__ == "__main__":
    main()
