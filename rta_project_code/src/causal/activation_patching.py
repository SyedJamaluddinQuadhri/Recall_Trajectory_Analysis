"""
activation_patching.py — Causal validation of the MLP-spike-alignment
signal via activation patching (ROME-style intervention).

THE EXPERIMENT:
1. Run a CORRECT example (model knows this fact) — capture the MLP
   sublayer's output at the layer flagged as the "recall event" layer,
   AT THE SPECIFIC POSITION that predicts the correct example's own
   first answer token (i.e., the last token of correct_prompt).
2. Run a CONFABULATED example (model gets this wrong, same relation
   template, different subject) — but at that SAME layer, physically
   replace the model's own MLP output, AT THE POSITION THAT PREDICTS
   THE CONFABULATED EXAMPLE'S OWN FIRST ANSWER TOKEN, with the captured
   value from the correct example.
3. Measure two things:
   (a) Does the trajectory shape shift toward the "genuine recall"
       signature (earlier crystallization, higher MLP-alignment)?
   (b) Does the model's prediction AT THAT SPECIFIC POSITION shift
       toward the correct example's answer token?
4. Specificity checks (TWO controls):
   (a) Random-layer control: same position logic, different layer.
   (b) Noise-vector control: same position, norm-matched random noise
       instead of the correct example's real MLP output.

v4 CORRECTNESS FIX (found during a deep audit — this is the most
important fix in the whole project):
Earlier versions patched using a hardcoded `token_positions=[-1]`
(the LAST position of the full prompt+answer sequence) and captured
the correct example's MLP output from its ENTIRE sequence, then read
off position 0 of that capture as the "source." For any multi-token
answer, BOTH of these were wrong:
  - The patch target `-1` lands on the position AFTER the whole
    answer, not the position that predicts the answer's FIRST token
    (verified concretely: for a 10-token prompt + 3-token answer,
    -1 resolves to position 12, which predicts a 4th, nonexistent
    answer token — not position 9, which predicts the actual first
    answer token).
  - The capture source read `source[:, 0, :]` — position 0 of
    correct_prompt (typically something like the "Q" token) — not
    the LAST position of correct_prompt, which is where that
    example's own recall event actually happens.
Both bugs together meant the causal experiment was silently patching
noise-adjacent content into an unrelated position, for any answer
longer than one token. This version fixes both: capture is now
restricted to exactly one meaningful position (the last position of
correct_prompt), and the patch target is computed the same way
build_teacher_forced_input() computes predict_positions, so "before"
and "after" are guaranteed to refer to the same, correct position.

v3 CHANGE (kept): this file reuses build_teacher_forced_input() and
project_trajectory_from_hidden_states() from trajectory_extractor.py
instead of duplicating trajectory logic.

v2 CHANGE (kept): answer texts are now passed in explicitly wherever
already known (from the matched-pairs dataset), instead of calling
extract_trajectory() with answer_text=None, which would silently
trigger a fresh model.generate() call — wasteful, and risks a subtle
mismatch against the max_new_tokens length used when the pair was
originally labeled.
"""

from typing import Dict, Optional, List, Tuple
import random

import torch

from src.utils import get_logger
from src.instrumentation.trajectory_extractor import (
    extract_trajectory, MLPNormCapture,
    build_teacher_forced_input, project_trajectory_from_hidden_states,
    tokenize_answer, LayerTrajectory,
)
from src.features.shape_descriptors import extract_all_descriptors

log = get_logger("causal.activation_patching")


class ActivationPatcher:
    """
    Manages capturing an MLP output from ONE SPECIFIC token position in
    one forward pass, and injecting it (or a noise-matched substitute)
    into ONE SPECIFIC token position of another forward pass.

    v4 FIX: both capture and patch now operate on a single, explicitly
    computed position — never a blind "last position of whatever
    sequence happens to be running," which was the source of the
    position-mismatch bug for multi-token answers.
    """

    def __init__(self, model, structure: dict):
        self.model = model
        self.layers = structure["layers"]
        self.captured_output: Optional[torch.Tensor] = None  # shape [1, 1, hidden_dim]
        self._capture_hook_handle = None
        self._patch_hook_handle = None

    def capture_at_position(self, layer_idx: int, position: int):
        """
        Register a hook that captures this layer's MLP output at EXACTLY
        the given token position (not the whole sequence).

        Args:
            layer_idx: which transformer block's MLP to capture
            position:  the exact position to capture (e.g. -1 for "last
                       position of whatever sequence this forward pass
                       receives" — safe here specifically because the
                       forward pass this is used with is always just
                       `correct_prompt` alone, no answer appended, so
                       -1 genuinely IS the position predicting the
                       correct answer's first token)
        """
        def hook(module, input, output):
            seq_len = output.shape[1]
            p = position if position >= 0 else seq_len + position
            self.captured_output = output[:, p:p + 1, :].detach().clone()
        self._capture_hook_handle = self.layers[layer_idx].mlp.register_forward_hook(hook)

    def stop_capture(self):
        if self._capture_hook_handle:
            self._capture_hook_handle.remove()
            self._capture_hook_handle = None

    def patch_at_position(
        self, layer_idx: int, position: int,
        replacement: Optional[torch.Tensor] = None,
    ):
        """
        Register a hook that REPLACES this layer's MLP output at EXACTLY
        the given token position.

        Args:
            layer_idx: which transformer block's MLP to patch
            position:  the exact position to overwrite (must be computed
                       via build_teacher_forced_input()'s predict_positions
                       for the sequence this hook will run against — NOT
                       a blind -1, since the forward pass here includes
                       the answer tokens appended after the prompt)
            replacement: tensor of shape [1, 1, hidden_dim] to substitute
                       in. If None, uses whatever was captured via
                       capture_at_position(). Pass a noise tensor here
                       for the noise-vector control.
        """
        source = replacement if replacement is not None else self.captured_output
        if source is None:
            raise ValueError("No replacement tensor available — call capture_at_position() first or pass one explicitly.")

        def hook(module, input, output):
            patched = output.clone()
            seq_len = patched.shape[1]
            p = position if position >= 0 else seq_len + position
            if 0 <= p < seq_len:
                patched[:, p, :] = source[:, 0, :]
            return patched

        self._patch_hook_handle = self.layers[layer_idx].mlp.register_forward_hook(hook)

    def stop_patch(self):
        if self._patch_hook_handle:
            self._patch_hook_handle.remove()
            self._patch_hook_handle = None


def _make_noise_like(reference: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
    """
    Build a random noise tensor with the SAME per-position L2 norm as
    `reference` (shape [1, 1, hidden_dim]), but random direction. Used
    for the noise-vector control: isolates whether the CONTENT of the
    correct example's MLP write matters, versus merely its magnitude.
    """
    if seed is not None:
        torch.manual_seed(seed)
    noise = torch.randn_like(reference)
    ref_norm = reference.norm(dim=-1, keepdim=True)
    noise_norm = noise.norm(dim=-1, keepdim=True)
    return noise * (ref_norm / (noise_norm + 1e-8))


def _get_answer_token_ids(tokenizer, known_answer_text: Optional[str], max_answer_tokens: int) -> List[int]:
    """Tokenize a known answer text, with a safe fallback."""
    if not known_answer_text:
        return []
    ids = tokenize_answer(tokenizer, known_answer_text)[:max_answer_tokens]
    return ids if ids else []


def _compute_trajectory_with_optional_patch(
    model, tokenizer, lens, structure,
    prompt: str, answer_token_ids: List[int],
) -> Tuple["LayerTrajectory", int, "torch.Tensor"]:
    """
    Run ONE teacher-forced forward pass and return:
      - the LayerTrajectory
      - the argmax token ID at predict_positions[0]
      - the FULL softmax probability vector at predict_positions[0]
        (shape [vocab_size] on CPU)

    Returning the full distribution lets the caller query the probability
    of ANY token (correct-answer token, confab token, etc.) without
    running a second forward pass. This eliminates the redundant pass
    that previously re-ran the identical forward pass to recover
    prob_confab_token_before.

    Any patch hooks registered on `model` before this call are live.
    """
    import torch.nn.functional as F
    device = next(model.parameters()).device
    input_ids, predict_positions = build_teacher_forced_input(
        tokenizer, prompt, answer_token_ids, device
    )

    mlp_capture = MLPNormCapture(model, structure, token_positions=predict_positions)
    mlp_capture.attach()
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True)
    mlp_capture.detach()

    prob_traj, entropy_traj, rank_traj, top10, per_token_probs = \
        project_trajectory_from_hidden_states(
            out.hidden_states, lens, structure, predict_positions, answer_token_ids
        )

    traj = LayerTrajectory(
        prompt=prompt, answer_text=tokenizer.decode(answer_token_ids),
        answer_token_ids=answer_token_ids, n_layers=structure["n_layers"],
        n_answer_tokens=len(answer_token_ids),
    )
    traj.mlp_norm_trajectory = mlp_capture.get_norms()
    traj.prob_trajectory      = prob_traj
    traj.entropy_trajectory   = entropy_traj
    traj.rank_trajectory      = rank_traj
    traj.top10_tokens_per_layer = top10
    traj.per_token_prob_trajectories = per_token_probs

    first_predict_pos = predict_positions[0]
    logits_at_pos     = out.logits[0, first_predict_pos]           # [vocab_size]
    argmax_token_id   = logits_at_pos.argmax().item()
    full_probs        = F.softmax(logits_at_pos, dim=-1).cpu()     # [vocab_size]

    return traj, argmax_token_id, full_probs


def run_patching_experiment(
    model,
    tokenizer,
    lens,
    structure: dict,
    correct_prompt: str,
    confab_prompt: str,
    target_layer: int,
    baseline_norms: Dict[str, List[float]],
    correct_answer_text: Optional[str] = None,
    confab_answer_text: Optional[str] = None,
    control_mode: str = "captured",  # "captured" (real patch) or "noise" (control)
    max_answer_tokens: int = 8,
    noise_seed: Optional[int] = None,
) -> Dict:
    """
    Full causal patching experiment for one matched pair.

    Args:
        model, tokenizer, lens, structure: standard RTA instrumentation objects
        correct_prompt: prompt where the model answers correctly
        confab_prompt:  prompt (same relation template, different subject)
                        where the model confabulates
        target_layer:   the layer identified as the "recall event" layer
        baseline_norms: dict with "mean"/"std" keys, from compute_baseline_norms()
        correct_answer_text: the ALREADY-KNOWN correct answer (e.g. from
                        build_matched_pairs.py's "correct_model_answer").
                        Passing this explicitly avoids a redundant, and
                        potentially inconsistent, model.generate() call.
                        If None, falls back to generating fresh (logged
                        as a warning — this both costs extra forward
                        passes and can subtly mismatch the answer that
                        was used when the pair was originally labeled).
        confab_answer_text: same, for the confabulated example.
        control_mode:   "captured" for the real experiment, or "noise"
                        for the noise-vector control.
        max_answer_tokens: cap on tracked answer tokens.

    Returns:
        dict with before/after descriptors, shifts, and the model's
        output token (at the patched position specifically) before vs.
        after patching.
    """
    device = next(model.parameters()).device
    patcher = ActivationPatcher(model, structure)

    # ── Get confab answer token IDs ───────────────────────────────────────────
    if confab_answer_text is None:
        log.warning(
            "confab_answer_text not provided — extract_trajectory will call "
            "model.generate() internally. Pass the pre-known answer (from "
            "build_matched_pairs.py) to avoid this and guarantee consistency "
            "with how the pair was originally labeled."
        )
        traj_before = extract_trajectory(model, tokenizer, lens, structure, confab_prompt)
        confab_answer_token_ids = traj_before.answer_token_ids
        probs_before = None  # full probs unavailable via this path
    else:
        confab_answer_token_ids = _get_answer_token_ids(tokenizer, confab_answer_text, max_answer_tokens)
        if not confab_answer_token_ids:
            confab_answer_token_ids = [tokenizer.eos_token_id]
        # Single forward pass — returns full softmax at predict_positions[0].
        # We reuse this distribution for before-patch metrics, so NO second
        # identical forward pass is needed later.
        traj_before, _, probs_before = _compute_trajectory_with_optional_patch(
            model, tokenizer, lens, structure, confab_prompt, confab_answer_token_ids
        )

    descriptors_before = extract_all_descriptors(traj_before, baseline_norms)

    # ── Get correct answer token IDs ─────────────────────────────────────────
    correct_answer_token_ids = _get_answer_token_ids(tokenizer, correct_answer_text, max_answer_tokens)
    if not correct_answer_token_ids:
        # Fall back: tokenize a single-space placeholder so index lookups below
        # never fail with an empty list.
        correct_answer_token_ids = [tokenizer.eos_token_id]

    # ── Before-patch probabilities (from the pass already done above) ─────────
    correct_token_id_for_metric = correct_answer_token_ids[0]
    confab_token_id_for_metric  = confab_answer_token_ids[0]
    if probs_before is not None:
        # Use values from the traj_before pass — NO extra forward pass.
        prob_correct_token_before = probs_before[correct_token_id_for_metric].item()
        prob_confab_token_before  = probs_before[confab_token_id_for_metric].item()
    else:
        prob_correct_token_before = 0.0
        prob_confab_token_before  = 0.0

    # ── Capture correct example's MLP output AT ITS OWN LAST POSITION ────────
    patcher.capture_at_position(target_layer, position=-1)
    inputs_correct = tokenizer(correct_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        model(**inputs_correct)
    patcher.stop_capture()

    replacement = patcher.captured_output
    if control_mode == "noise":
        replacement = _make_noise_like(patcher.captured_output, seed=noise_seed)

    # ── Determine the correct destination position in the confab sequence ─────
    _, confab_predict_positions = build_teacher_forced_input(
        tokenizer, confab_prompt, confab_answer_token_ids, device
    )
    patch_position = confab_predict_positions[0]

    # ── After-patch forward pass ──────────────────────────────────────────────
    patcher.patch_at_position(target_layer, position=patch_position, replacement=replacement)
    traj_after, new_argmax_id, probs_after = _compute_trajectory_with_optional_patch(
        model, tokenizer, lens, structure, confab_prompt, confab_answer_token_ids
    )
    patcher.stop_patch()

    descriptors_after = extract_all_descriptors(traj_after, baseline_norms)

    # ── After-patch probabilities ──────────────────────────────────────────────
    prob_correct_token_after = probs_after[correct_token_id_for_metric].item()
    prob_confab_token_after  = probs_after[confab_token_id_for_metric].item()

    new_token_text           = tokenizer.decode([new_argmax_id])
    original_first_token_text = tokenizer.decode([confab_token_id_for_metric])

    # ── Causal metrics ────────────────────────────────────────────────────────
    #
    # PRIMARY (content-specific): did P(correct_answer_first_token) INCREASE
    # at the patched position after injecting the correct subject's MLP output?
    correct_token_gain = prob_correct_token_after - prob_correct_token_before

    # SECONDARY: did the confab token probability drop?
    confab_token_drop  = prob_confab_token_before - prob_confab_token_after

    # TERTIARY: did the argmax flip away from the confab token?
    argmax_flipped = new_token_text.strip().lower() != original_first_token_text.strip().lower()

    # answer_changed = any meaningful causal disruption or flip away from the confabulated state:
    # argmax flipped OR significant confab drop (>= 10pp) OR gain in correct token (>= 5pp)
    answer_changed = argmax_flipped or (confab_token_drop >= 0.10) or (correct_token_gain >= 0.05)

    return {
        "correct_prompt":   correct_prompt,
        "confab_prompt":    confab_prompt,
        "target_layer":     target_layer,
        "patch_position":   patch_position,
        "control_mode":     control_mode,
        "descriptors_before": descriptors_before,
        "descriptors_after":  descriptors_after,
        "crystallization_shift": (
            descriptors_before["crystallization"] - descriptors_after["crystallization"]
        ),
        "mlp_alignment_shift": (
            descriptors_after["mlp_alignment"] - descriptors_before["mlp_alignment"]
        ),
        # Token-level results
        "original_first_token":       original_first_token_text,
        "new_first_token":            new_token_text,
        "argmax_flipped":             argmax_flipped,
        # PRIMARY causal metric — content-specific probability gain
        "prob_correct_token_before":  prob_correct_token_before,
        "prob_correct_token_after":   prob_correct_token_after,
        "correct_token_gain":         correct_token_gain,
        # Secondary — confab-token disruption
        "prob_confab_token_before":   prob_confab_token_before,
        "prob_confab_token_after":    prob_confab_token_after,
        "confab_token_drop":          confab_token_drop,
        # Composite primary metric
        "answer_changed": answer_changed,
    }


def run_specificity_check(
    model, tokenizer, lens, structure,
    correct_prompt: str, confab_prompt: str,
    target_layer: int, baseline_norms: Dict[str, List[float]],
    correct_answer_text: Optional[str] = None,
    confab_answer_text: Optional[str] = None,
    n_random_layers: int = 2,
) -> List[Dict]:
    """
    Specificity control 1 — RANDOM LAYER: repeat the patching experiment
    at random, UNRELATED layers (not the flagged target_layer) using the
    real captured MLP output.
    """
    n_layers = structure["n_layers"]
    candidate_layers = [l for l in range(n_layers) if l != target_layer]
    random_layers = random.sample(candidate_layers, min(n_random_layers, len(candidate_layers)))

    results = []
    for layer in random_layers:
        result = run_patching_experiment(
            model, tokenizer, lens, structure,
            correct_prompt, confab_prompt, layer, baseline_norms,
            correct_answer_text=correct_answer_text, confab_answer_text=confab_answer_text,
            control_mode="captured",
        )
        result["is_control"] = True
        result["control_type"] = "random_layer"
        results.append(result)

    return results


def run_noise_control_check(
    model, tokenizer, lens, structure,
    correct_prompt: str, confab_prompt: str,
    target_layer: int, baseline_norms: Dict[str, List[float]],
    correct_answer_text: Optional[str] = None,
    confab_answer_text: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict:
    """
    Specificity control 2 — NOISE VECTOR: repeat the patch at the SAME
    target_layer and SAME position, but substitute norm-matched random
    noise instead of the correct example's actual MLP output.
    """
    result = run_patching_experiment(
        model, tokenizer, lens, structure,
        correct_prompt, confab_prompt, target_layer, baseline_norms,
        correct_answer_text=correct_answer_text, confab_answer_text=confab_answer_text,
        control_mode="noise",
        noise_seed=seed,
    )
    result["is_control"] = True
    result["control_type"] = "noise_vector"
    return result
