"""
shape_descriptors.py — The four novel shape descriptors computed from a
LayerTrajectory. These are the entire signal the classifier uses.

Each function takes a LayerTrajectory (from instrumentation/trajectory_extractor.py)
and/or a baseline reference (mean AND std of MLP norms over many examples)
and returns a single float feature.

v2 CHANGES FROM THE FIRST IMPLEMENTATION:
  - mlp_spike_alignment() now uses a REAL empirically-measured standard
    deviation per layer instead of a fabricated `mean * 0.15` proxy.
  - early_topk_presence() now reads traj.answer_token_ids[0] (the first
    answer token) instead of a stale single-token attribute that no
    longer exists on the multi-token-aware LayerTrajectory.
  - compute_baseline_norms() now returns BOTH mean and std per layer.
"""

from typing import List, Dict
import numpy as np

from src.instrumentation.trajectory_extractor import LayerTrajectory
from src.utils import get_logger

log = get_logger("features.shape_descriptors")


def crystallization_layer(
    prob_trajectory: List[float],
    threshold: float = 0.5,
    normalize: bool = True,
) -> float:
    """
    Descriptor 1 — Crystallization layer.

    The earliest layer l* at which P_l(answer) exceeds `threshold` AND
    stays above it for all subsequent layers.

    A LOW value = the model became confident early and stayed confident
                  (signature of genuine recall).
    A HIGH value (or the max, "never crystallizes") = confidence only
                  stabilized very late, or never stabilized at all
                  (signature of confabulation).

    Args:
        prob_trajectory: P_l(answer) for l = 1..L
        threshold: confidence level considered "crystallized"
        normalize: if True, return l*/L in [0, 1]; else return raw layer index

    Returns:
        float — normalized (or raw) crystallization layer index.
        Returns 1.0 (normalized) or L (raw) if trajectory never crystallizes,
        i.e. treated as "as late as possible."
    """
    L = len(prob_trajectory)
    for l in range(L):
        if all(p >= threshold for p in prob_trajectory[l:]):
            return (l / L) if normalize else float(l)
    return 1.0 if normalize else float(L)


def trajectory_smoothness(prob_trajectory: List[float]) -> float:
    """
    Descriptor 2 — Trajectory smoothness (sharp single jump vs. noisy climb).

    Computed as: (largest single consecutive-layer jump) / (total variation).

    A HIGH value (close to 1.0) = almost all confidence change happened in
                ONE big jump — signature of a discrete retrieval event.
    A LOW value = confidence crept up via many small steps — signature of
                gradual language-model smoothing (confabulation).

    Args:
        prob_trajectory: P_l(answer) for l = 1..L

    Returns:
        float in [0, 1]. Returns 0.0 if trajectory is flat (no variation).
    """
    diffs = [
        prob_trajectory[i + 1] - prob_trajectory[i]
        for i in range(len(prob_trajectory) - 1)
    ]
    abs_diffs = [abs(d) for d in diffs]
    total_variation = sum(abs_diffs)

    if total_variation < 1e-8:
        return 0.0

    max_jump = max(abs_diffs)
    return max_jump / total_variation


def find_peak_confidence_layer(prob_trajectory: List[float]) -> int:
    """
    Shared helper: find the layer index where the confidence trajectory
    shows its single largest increase from the previous layer.

    Used both by mlp_spike_alignment() below (to find the layer whose
    MLP activity to check) and by the causal patching pipeline
    (scripts/run_stage4_causal_patching.py — to pick which layer to
    intervene on). Previously this exact computation was duplicated in
    both places; extracting it here means both call sites are
    guaranteed to agree on "the recall event layer" by construction.

    Args:
        prob_trajectory: P_l(answer) for l = 1..L

    Returns:
        int — the layer index (0-based, into the diffs array — i.e. the
        layer AFTER which the jump occurs) of the largest single
        increase. Returns 0 if the trajectory has fewer than 2 layers.
    """
    if len(prob_trajectory) < 2:
        return 0
    diffs = [
        prob_trajectory[i + 1] - prob_trajectory[i]
        for i in range(len(prob_trajectory) - 1)
    ]
    return int(np.argmax(diffs))


def mlp_spike_alignment(
    prob_trajectory: List[float],
    mlp_norms: List[float],
    baseline_mean: List[float],
    baseline_std: List[float],
) -> float:
    """
    Descriptor 3 — MLP-spike alignment (the core novel signal).

    Finds the layer of maximum confidence INCREASE, then checks whether
    that SAME layer shows an unusually large MLP write (relative to a
    REAL empirically-measured baseline distribution of MLP norms at that
    layer, computed over many other examples).

    A HIGH z-score = the confidence jump coincides with an unusual MLP
                write event — this is the mechanistic signature of a
                key-value memory lookup firing (genuine recall).
    A LOW or NEGATIVE z-score = confidence rose without any corresponding
                write event standing out — suggests the rise came from
                generic smoothing rather than a specific retrieval
                (confabulation signature).

    Args:
        prob_trajectory: P_l(answer) for l = 1..L
        mlp_norms:       M_l for l = 1..L (this example)
        baseline_mean:   mean M_l per layer, from compute_baseline_norms()
        baseline_std:    REAL standard deviation of M_l per layer, from
                         compute_baseline_norms() — this replaces the
                         earlier fabricated `mean * 0.15` proxy.

    Returns:
        float — z-score of the MLP norm at the peak-confidence-jump layer.
    """
    if len(prob_trajectory) < 2:
        return 0.0

    peak_layer = find_peak_confidence_layer(prob_trajectory)

    observed = mlp_norms[peak_layer]
    mean_at_layer = baseline_mean[peak_layer]
    std_at_layer = max(baseline_std[peak_layer], 1e-6)  # floor to avoid divide-by-zero

    z_score = (observed - mean_at_layer) / std_at_layer
    return float(z_score)


def early_topk_presence(
    top10_tokens_per_layer: List[List[int]],
    answer_token_ids: List[int],
    early_fraction: float = 0.33,
) -> float:
    """
    Descriptor 4 — Early top-k presence.

    Checks whether the eventual FIRST answer token appears ANYWHERE in
    the top-10 predictions during the EARLY portion of the network
    (first `early_fraction` of layers), even if it wasn't the top
    prediction yet.

    We check the FIRST answer token specifically (not all tokens),
    because top10_tokens_per_layer is only recorded for that position
    (see trajectory_extractor.py) — the first token of a multi-token
    answer is generally the most informative content token to check
    for an early "trace."

    A value of 1.0 = the answer token had an early trace in the model's
                predictions (genuine recall signature).
    A value of 0.0 = the answer token appeared from nowhere, only in
                later layers (confabulation signature).

    Args:
        top10_tokens_per_layer: list of top-10 token ID lists, one per layer
        answer_token_ids: the FULL list of answer token ids (only [0] is checked)
        early_fraction: what fraction of early layers counts as "early"

    Returns:
        float — 1.0 if present in any early layer's top-10, else 0.0
    """
    if not answer_token_ids or not top10_tokens_per_layer:
        return 0.0

    first_token_id = answer_token_ids[0]
    n_layers = len(top10_tokens_per_layer)
    n_early = max(1, int(n_layers * early_fraction))

    for l in range(n_early):
        if first_token_id in top10_tokens_per_layer[l]:
            return 1.0
    return 0.0


def compute_baseline_norms(mlp_norm_trajectories: List[List[float]]) -> Dict[str, List[float]]:
    """
    Compute the MEAN AND STANDARD DEVIATION of MLP norm per layer over
    MANY examples — this is the reference distribution needed by
    mlp_spike_alignment().

    v2 CHANGE: previously only the mean was computed, and the standard
    deviation was approximated as `mean * 0.15` (a fabricated proxy).
    This function now returns the REAL empirical standard deviation,
    computed the same way as the mean, over the same reference sample.

    Call this ONCE over a large sample of trajectories (e.g. 1000+
    examples, mixing correct and confabulated), then reuse the result
    for every subsequent mlp_spike_alignment() computation.

    Args:
        mlp_norm_trajectories: list of M_l sequences, one per example

    Returns:
        dict with "mean": list of per-layer means, "std": list of per-layer
        standard deviations (both length n_layers)
    """
    arr = np.array(mlp_norm_trajectories)  # [n_examples, n_layers]
    return {
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
    }


def extract_all_descriptors(
    traj: LayerTrajectory,
    baseline_norms: Dict[str, List[float]],
) -> dict:
    """
    Convenience function: compute all four shape descriptors plus two
    baseline (snapshot) features for comparison, from a single trajectory.

    Args:
        traj: a LayerTrajectory (multi-token aware, from trajectory_extractor.py)
        baseline_norms: dict with "mean" and "std" keys, from compute_baseline_norms()

    Returns:
        dict with keys: crystallization, smoothness, mlp_alignment,
        early_presence, final_entropy, final_rank_normalized
    """
    return {
        "crystallization":        crystallization_layer(traj.prob_trajectory),
        "smoothness":              trajectory_smoothness(traj.prob_trajectory),
        "mlp_alignment":           mlp_spike_alignment(
                                       traj.prob_trajectory, traj.mlp_norm_trajectory,
                                       baseline_norms["mean"], baseline_norms["std"],
                                   ),
        "early_presence":          early_topk_presence(
                                       traj.top10_tokens_per_layer, traj.answer_token_ids
                                   ),
        # Baseline (snapshot) features — for the ablation study comparison
        "final_entropy":           traj.entropy_trajectory[-1] if traj.entropy_trajectory else 0.0,
        "final_rank_normalized":   traj.rank_trajectory[-1] / 50000.0 if traj.rank_trajectory else 1.0,
    }
