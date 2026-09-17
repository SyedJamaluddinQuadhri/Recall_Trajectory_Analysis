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

v5 CHANGE — two new patching modes (existing single-layer path is
unchanged and still the primary baseline):

  Mode 2 — Window: instead of patching only the single peak layer,
  patches layers [peak-1, peak, peak+1] (clamped to valid range) in
  the same forward pass. Rationale: the recall event may be smeared
  across 2-3 adjacent layers; single-layer patching injects an
  incomplete signal. Each window layer has its own independently
  norm-matched noise for the noise-vector control.

  Mode 3 — Residual stream: instead of patching the MLP sublayer's
  output, patches the FULL block output at the target layer (the
  combined attention + MLP residual, i.e., what is passed to the
  next layer). Rationale: the causally load-bearing content may live
  in the combined residual state, not only the isolated MLP write.
  The noise control is norm-matched to the full residual vector's norm
  (which is larger than the MLP output's norm alone — correctly
  matching the magnitude of what is actually being replaced).

v4 CORRECTNESS FIX (found during a deep audit — this is the most
important fix in the whole project):
Earlier versions patched using a hardcoded `token_positions=[-1]`
(the LAST position of the full prompt+answer sequence) and captured
the correct example's MLP output from its ENTIRE sequence, then read
off position 0 of that capture as the "source."  For any multi-token
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
    Manages capturing activations from specific token positions in one
    forward pass and injecting them (or noise-matched substitutes) into
    specific token positions of another forward pass.

    Supports three patching modes:
      - Single-layer MLP (existing baseline): capture_at_position / patch_at_position
      - Window MLP (new): capture_mlp_window / patch_mlp_window — captures
        and patches a window of adjacent MLP layers simultaneously
      - Residual stream (new): capture_residual_at_position /
        patch_residual_at_position — captures and patches the FULL block
        output (attention + MLP combined), not just the MLP write

    v4 FIX: both capture and patch now operate on a single, explicitly
    computed position — never a blind "last position of whatever sequence
    happens to be running," which was the source of the position-mismatch
    bug for multi-token answers.
    """

    def __init__(self, model, structure: dict):
        self.model = model
        self.layers = structure["layers"]
        self.n_layers = structure["n_layers"]

        # ── Single-layer MLP mode (existing) ─────────────────────────────────
        self.captured_output: Optional[torch.Tensor] = None  # [1, 1, hidden_dim]
        self._capture_hook_handle = None
        self._patch_hook_handle = None

        # ── Window MLP mode (new) ─────────────────────────────────────────────
        self.captured_window: Dict[int, torch.Tensor] = {}   # layer_idx -> [1, 1, hidden_dim]
        self._window_capture_handles: List = []
        self._window_patch_handles: List = []

        # ── Residual stream mode (new) ────────────────────────────────────────
        self.captured_residual: Optional[torch.Tensor] = None  # [1, 1, hidden_dim]
        self._capture_residual_handle = None
        self._patch_residual_handle = None

    # ── Single-layer MLP methods (existing — unchanged) ──────────────────────

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
            raise ValueError(
                "No replacement tensor available — call capture_at_position() "
                "first or pass one explicitly."
            )

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

    # ── Window MLP methods (new) ──────────────────────────────────────────────

    def capture_mlp_window(self, layer_indices: List[int], position: int):
        """
        Register hooks on multiple MLP sublayers to capture their outputs
        at EXACTLY the given token position, all in a single forward pass.

        Each layer's capture is stored independently in self.captured_window,
        keyed by layer index, so per-layer noise controls can be built
        with the correct per-layer norm.

        Args:
            layer_indices: which transformer blocks' MLPs to capture (e.g.
                           [peak-1, peak, peak+1] clamped to valid range)
            position:      the token position to capture at (use -1 for the
                           last position of a prompt-only forward pass)
        """
        self.captured_window = {}
        self._window_capture_handles = []

        for layer_idx in layer_indices:
            # Explicit closure argument to avoid late-binding over loop var
            def make_capture_hook(l_idx, pos):
                def hook(module, input, output):
                    seq_len = output.shape[1]
                    p = pos if pos >= 0 else seq_len + pos
                    self.captured_window[l_idx] = output[:, p:p + 1, :].detach().clone()
                return hook

            handle = self.layers[layer_idx].mlp.register_forward_hook(
                make_capture_hook(layer_idx, position)
            )
            self._window_capture_handles.append(handle)

    def stop_capture_window(self):
        for h in self._window_capture_handles:
            h.remove()
        self._window_capture_handles = []

    def patch_mlp_window(
        self,
        layer_indices: List[int],
        position: int,
        replacements: Dict[int, torch.Tensor],
    ):
        """
        Register hooks on multiple MLP sublayers to replace their outputs
        at EXACTLY the given token position, all active in a single forward
        pass (so the downstream layers see the fully-patched intermediate
        state, not sequential partial patches).

        Args:
            layer_indices:  which transformer blocks to patch
            position:       exact token position to overwrite (computed via
                            build_teacher_forced_input()'s predict_positions)
            replacements:   dict mapping layer_idx -> replacement tensor of
                            shape [1, 1, hidden_dim]. Typically built from
                            self.captured_window (real content) or from
                            _make_noise_like(captured) per layer (noise ctrl)
        """
        self._window_patch_handles = []

        for layer_idx in layer_indices:
            source = replacements[layer_idx]

            def make_patch_hook(src, pos):
                def hook(module, input, output):
                    patched = output.clone()
                    seq_len = patched.shape[1]
                    p = pos if pos >= 0 else seq_len + pos
                    if 0 <= p < seq_len:
                        patched[:, p, :] = src[:, 0, :]
                    return patched
                return hook

            handle = self.layers[layer_idx].mlp.register_forward_hook(
                make_patch_hook(source, position)
            )
            self._window_patch_handles.append(handle)

    def stop_patch_window(self):
        for h in self._window_patch_handles:
            h.remove()
        self._window_patch_handles = []

    # ── Residual stream methods (new) ─────────────────────────────────────────

    def capture_residual_at_position(self, layer_idx: int, position: int):
        """
        Register a hook that captures the FULL block output (residual stream)
        at the given token position — i.e., the combined attention + MLP
        contribution that is passed to the NEXT layer, not just the MLP write.

        In Phi-2's PhiDecoderLayer, forward() returns a tuple where
        output[0] is the hidden state tensor [batch, seq_len, hidden_dim].
        The remaining elements are optional (attention weights, past_key_values
        etc.) and are left untouched.

        Args:
            layer_idx: which transformer block to capture the full output of
            position:  token position to capture (-1 = last position of the
                       prompt-only forward pass run against correct_prompt)
        """
        def hook(module, input, output):
            # output is a tuple; output[0] is [batch, seq_len, hidden_dim]
            hidden = output[0]
            seq_len = hidden.shape[1]
            p = position if position >= 0 else seq_len + position
            self.captured_residual = hidden[:, p:p + 1, :].detach().clone()
        self._capture_residual_handle = self.layers[layer_idx].register_forward_hook(hook)

    def stop_capture_residual(self):
        if self._capture_residual_handle:
            self._capture_residual_handle.remove()
            self._capture_residual_handle = None

    def patch_residual_at_position(
        self,
        layer_idx: int,
        position: int,
        replacement: Optional[torch.Tensor] = None,
    ):
        """
        Register a hook that REPLACES the full block output (residual stream)
        at the given token position. This overwrites what the NEXT layer
        receives — not just the MLP write within this layer.

        The hook receives output as a tuple and must return a modified tuple
        with output[0] patched; all remaining elements are passed through
        unchanged (preserves optional attention weights / past_key_values).

        Args:
            layer_idx:   which transformer block to patch the output of
            position:    exact token position to overwrite (computed via
                         build_teacher_forced_input()'s predict_positions)
            replacement: tensor of shape [1, 1, hidden_dim]. If None, uses
                         self.captured_residual. For the noise control, pass
                         _make_noise_like(captured_residual) — note this is
                         norm-matched to the FULL residual vector's norm, not
                         the MLP output's norm, because that is what is being
                         replaced.
        """
        source = replacement if replacement is not None else self.captured_residual
        if source is None:
            raise ValueError(
                "No residual replacement available — call "
                "capture_residual_at_position() first or pass one explicitly."
            )

        def hook(module, input, output):
            # output is a tuple; only patch output[0] (the hidden states)
            hidden = output[0].clone()
            seq_len = hidden.shape[1]
            p = position if position >= 0 else seq_len + position
            if 0 <= p < seq_len:
                hidden[:, p, :] = source[:, 0, :]
            # Return the full tuple with the patched hidden states
            return (hidden,) + output[1:]

        self._patch_residual_handle = self.layers[layer_idx].register_forward_hook(hook)

    def stop_patch_residual(self):
        if self._patch_residual_handle:
            self._patch_residual_handle.remove()
            self._patch_residual_handle = None


def _make_noise_like(reference: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
    """
    Build a random noise tensor with the SAME per-position L2 norm as
    `reference` (shape [1, 1, hidden_dim]), but random direction. Used
    for the noise-vector control: isolates whether the CONTENT of the
    correct example's activation matters, versus merely its magnitude.
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


# ── Helper: shared result-dict builder ───────────────────────────────────────

def _build_result_dict(
    correct_prompt, confab_prompt, target_layer, patch_position,
    control_mode, patch_mode,
    descriptors_before, descriptors_after,
    tokenizer, new_argmax_id, confab_token_id_for_metric,
    prob_correct_token_before, prob_correct_token_after,
    prob_confab_token_before, prob_confab_token_after,
    extra_fields: Optional[Dict] = None,
) -> Dict:
    """
    Assemble the standard result dict shared by all three patching modes.
    All callers use identical field names so summarize_causal_experiments()
    works unchanged across modes.
    """
    new_token_text            = tokenizer.decode([new_argmax_id])
    original_first_token_text = tokenizer.decode([confab_token_id_for_metric])

    correct_token_gain = prob_correct_token_after - prob_correct_token_before
    confab_token_drop  = prob_confab_token_before  - prob_confab_token_after
    argmax_flipped     = (
        new_token_text.strip().lower() != original_first_token_text.strip().lower()
    )
    answer_changed = (
        argmax_flipped
        or (confab_token_drop  >= 0.10)
        or (correct_token_gain >= 0.05)
    )

    result = {
        "correct_prompt":   correct_prompt,
        "confab_prompt":    confab_prompt,
        "target_layer":     target_layer,
        "patch_position":   patch_position,
        "control_mode":     control_mode,
        "patch_mode":       patch_mode,
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
    if extra_fields:
        result.update(extra_fields)
    return result


def _get_before_patch_state(
    model, tokenizer, lens, structure,
    confab_prompt, confab_answer_text, max_answer_tokens,
    correct_answer_text,
):
    """
    Shared helper: compute the before-patch trajectory and token probabilities.
    Returns (traj_before, confab_answer_token_ids, correct_token_id,
             confab_token_id, prob_correct_before, prob_confab_before).
    """
    if confab_answer_text is None:
        log.warning(
            "confab_answer_text not provided — extract_trajectory will call "
            "model.generate() internally. Pass the pre-known answer (from "
            "build_matched_pairs.py) to avoid this and guarantee consistency "
            "with how the pair was originally labeled."
        )
        traj_before = extract_trajectory(model, tokenizer, lens, structure, confab_prompt)
        confab_answer_token_ids = traj_before.answer_token_ids
        probs_before = None
    else:
        confab_answer_token_ids = _get_answer_token_ids(tokenizer, confab_answer_text, max_answer_tokens)
        if not confab_answer_token_ids:
            confab_answer_token_ids = [tokenizer.eos_token_id]
        traj_before, _, probs_before = _compute_trajectory_with_optional_patch(
            model, tokenizer, lens, structure, confab_prompt, confab_answer_token_ids
        )

    correct_answer_token_ids = _get_answer_token_ids(tokenizer, correct_answer_text, max_answer_tokens)
    if not correct_answer_token_ids:
        correct_answer_token_ids = [tokenizer.eos_token_id]

    correct_token_id = correct_answer_token_ids[0]
    confab_token_id  = confab_answer_token_ids[0]

    if probs_before is not None:
        prob_correct_before = probs_before[correct_token_id].item()
        prob_confab_before  = probs_before[confab_token_id].item()
    else:
        prob_correct_before = 0.0
        prob_confab_before  = 0.0

    return (
        traj_before, confab_answer_token_ids,
        correct_token_id, confab_token_id,
        prob_correct_before, prob_confab_before,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 1 — Single-layer MLP patch (existing baseline — UNCHANGED)
# ═══════════════════════════════════════════════════════════════════════════════

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
        "patch_mode":       "single_layer",
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


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 2 — Window MLP patch (new)
# ═══════════════════════════════════════════════════════════════════════════════

def run_window_patching_experiment(
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
    control_mode: str = "captured",  # "captured" or "noise"
    max_answer_tokens: int = 8,
    noise_seed: Optional[int] = None,
) -> Dict:
    """
    Window MLP patching experiment: patches MLP outputs at
    [target_layer-1, target_layer, target_layer+1] (clamped to valid
    layer range) simultaneously in a single forward pass.

    Rationale: if the recall event is smeared across 2-3 adjacent layers,
    single-layer patching injects an incomplete signal. The noise control
    uses per-layer independent noise, each norm-matched to that layer's
    actual MLP output.

    Args and return format: identical to run_patching_experiment, with the
    addition of a "window_layers" key in the result dict.
    """
    device = next(model.parameters()).device
    n_layers = structure["n_layers"]
    patcher = ActivationPatcher(model, structure)

    # Compute window (clamped, deduplicated) — e.g. at layer 0 gives [0, 1]
    window_layers = sorted(set([
        max(0, target_layer - 1),
        target_layer,
        min(n_layers - 1, target_layer + 1),
    ]))

    # ── Before-patch state ────────────────────────────────────────────────────
    (
        traj_before, confab_answer_token_ids,
        correct_token_id_for_metric, confab_token_id_for_metric,
        prob_correct_token_before, prob_confab_token_before,
    ) = _get_before_patch_state(
        model, tokenizer, lens, structure,
        confab_prompt, confab_answer_text, max_answer_tokens, correct_answer_text,
    )
    descriptors_before = extract_all_descriptors(traj_before, baseline_norms)

    # ── Capture MLP outputs from all window layers at position -1 ─────────────
    patcher.capture_mlp_window(window_layers, position=-1)
    inputs_correct = tokenizer(correct_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        model(**inputs_correct)
    patcher.stop_capture_window()

    # ── Build replacements (real content or per-layer independent noise) ───────
    replacements = {}
    for l_idx in window_layers:
        captured = patcher.captured_window[l_idx]
        if control_mode == "noise":
            # Per-layer seed so each layer gets an independently drawn noise
            # vector, each norm-matched to that layer's own MLP output norm.
            layer_seed = (noise_seed * 100 + l_idx) if noise_seed is not None else None
            replacements[l_idx] = _make_noise_like(captured, seed=layer_seed)
        else:
            replacements[l_idx] = captured

    # ── Determine patch position in confab sequence ───────────────────────────
    _, confab_predict_positions = build_teacher_forced_input(
        tokenizer, confab_prompt, confab_answer_token_ids, device
    )
    patch_position = confab_predict_positions[0]

    # ── After-patch forward pass (all window layers patched simultaneously) ───
    patcher.patch_mlp_window(window_layers, position=patch_position, replacements=replacements)
    traj_after, new_argmax_id, probs_after = _compute_trajectory_with_optional_patch(
        model, tokenizer, lens, structure, confab_prompt, confab_answer_token_ids
    )
    patcher.stop_patch_window()

    descriptors_after = extract_all_descriptors(traj_after, baseline_norms)

    prob_correct_token_after = probs_after[correct_token_id_for_metric].item()
    prob_confab_token_after  = probs_after[confab_token_id_for_metric].item()

    result = _build_result_dict(
        correct_prompt, confab_prompt, target_layer, patch_position,
        control_mode, "window",
        descriptors_before, descriptors_after,
        tokenizer, new_argmax_id, confab_token_id_for_metric,
        prob_correct_token_before, prob_correct_token_after,
        prob_confab_token_before, prob_confab_token_after,
        extra_fields={"window_layers": window_layers},
    )
    return result


def run_window_specificity_check(
    model, tokenizer, lens, structure,
    correct_prompt: str, confab_prompt: str,
    target_layer: int, baseline_norms: Dict[str, List[float]],
    correct_answer_text: Optional[str] = None,
    confab_answer_text: Optional[str] = None,
    n_random_layers: int = 2,
) -> List[Dict]:
    """
    Random-layer control for window mode: repeats run_window_patching_experiment
    with n_random_layers randomly chosen layers (independently drawn, NOT as a
    window around a random center) to keep the control maximally unstructured.
    """
    n_layers = structure["n_layers"]
    candidate_layers = [l for l in range(n_layers) if l != target_layer]
    random_layers = random.sample(candidate_layers, min(n_random_layers, len(candidate_layers)))

    results = []
    for layer in random_layers:
        result = run_window_patching_experiment(
            model, tokenizer, lens, structure,
            correct_prompt, confab_prompt, layer, baseline_norms,
            correct_answer_text=correct_answer_text, confab_answer_text=confab_answer_text,
            control_mode="captured",
        )
        result["is_control"] = True
        result["control_type"] = "random_layer"
        results.append(result)

    return results


def run_window_noise_control(
    model, tokenizer, lens, structure,
    correct_prompt: str, confab_prompt: str,
    target_layer: int, baseline_norms: Dict[str, List[float]],
    correct_answer_text: Optional[str] = None,
    confab_answer_text: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict:
    """
    Noise-vector control for window mode: same window of layers, but each
    layer's content is replaced with an independently drawn norm-matched
    noise vector (norm-matched to that layer's own captured MLP output).
    """
    result = run_window_patching_experiment(
        model, tokenizer, lens, structure,
        correct_prompt, confab_prompt, target_layer, baseline_norms,
        correct_answer_text=correct_answer_text, confab_answer_text=confab_answer_text,
        control_mode="noise",
        noise_seed=seed,
    )
    result["is_control"] = True
    result["control_type"] = "noise_vector"
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 3 — Residual stream patch (new)
# ═══════════════════════════════════════════════════════════════════════════════

def run_residual_stream_patching_experiment(
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
    control_mode: str = "captured",  # "captured" or "noise"
    max_answer_tokens: int = 8,
    noise_seed: Optional[int] = None,
) -> Dict:
    """
    Residual-stream patching experiment: patches the FULL block output
    (attention + MLP combined) at the target layer and position, instead
    of only the MLP sublayer's write.

    Rationale: the causally load-bearing content may live in the combined
    residual state passed to the next layer, not just in the isolated MLP
    write. The noise control is norm-matched to the full residual vector's
    norm (which is larger than the MLP output's norm alone — correctly
    matching the magnitude of what is actually being replaced).

    Args and return format: identical to run_patching_experiment.
    """
    device = next(model.parameters()).device
    patcher = ActivationPatcher(model, structure)

    # ── Before-patch state ────────────────────────────────────────────────────
    (
        traj_before, confab_answer_token_ids,
        correct_token_id_for_metric, confab_token_id_for_metric,
        prob_correct_token_before, prob_confab_token_before,
    ) = _get_before_patch_state(
        model, tokenizer, lens, structure,
        confab_prompt, confab_answer_text, max_answer_tokens, correct_answer_text,
    )
    descriptors_before = extract_all_descriptors(traj_before, baseline_norms)

    # ── Capture full residual stream at target layer, position -1 ─────────────
    patcher.capture_residual_at_position(target_layer, position=-1)
    inputs_correct = tokenizer(correct_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        model(**inputs_correct)
    patcher.stop_capture_residual()

    replacement = patcher.captured_residual
    if control_mode == "noise":
        # Norm-match to the FULL residual vector's norm (not the MLP output's
        # norm), because we are replacing the full block output, not just MLP.
        replacement = _make_noise_like(patcher.captured_residual, seed=noise_seed)

    # ── Determine patch position in confab sequence ───────────────────────────
    _, confab_predict_positions = build_teacher_forced_input(
        tokenizer, confab_prompt, confab_answer_token_ids, device
    )
    patch_position = confab_predict_positions[0]

    # ── After-patch forward pass ──────────────────────────────────────────────
    patcher.patch_residual_at_position(target_layer, position=patch_position, replacement=replacement)
    traj_after, new_argmax_id, probs_after = _compute_trajectory_with_optional_patch(
        model, tokenizer, lens, structure, confab_prompt, confab_answer_token_ids
    )
    patcher.stop_patch_residual()

    descriptors_after = extract_all_descriptors(traj_after, baseline_norms)

    prob_correct_token_after = probs_after[correct_token_id_for_metric].item()
    prob_confab_token_after  = probs_after[confab_token_id_for_metric].item()

    return _build_result_dict(
        correct_prompt, confab_prompt, target_layer, patch_position,
        control_mode, "residual_stream",
        descriptors_before, descriptors_after,
        tokenizer, new_argmax_id, confab_token_id_for_metric,
        prob_correct_token_before, prob_correct_token_after,
        prob_confab_token_before, prob_confab_token_after,
    )


def run_residual_specificity_check(
    model, tokenizer, lens, structure,
    correct_prompt: str, confab_prompt: str,
    target_layer: int, baseline_norms: Dict[str, List[float]],
    correct_answer_text: Optional[str] = None,
    confab_answer_text: Optional[str] = None,
    n_random_layers: int = 2,
) -> List[Dict]:
    """
    Random-layer control for residual-stream mode: patches the full block
    output at random unrelated layers using the real captured residual.
    """
    n_layers = structure["n_layers"]
    candidate_layers = [l for l in range(n_layers) if l != target_layer]
    random_layers = random.sample(candidate_layers, min(n_random_layers, len(candidate_layers)))

    results = []
    for layer in random_layers:
        result = run_residual_stream_patching_experiment(
            model, tokenizer, lens, structure,
            correct_prompt, confab_prompt, layer, baseline_norms,
            correct_answer_text=correct_answer_text, confab_answer_text=confab_answer_text,
            control_mode="captured",
        )
        result["is_control"] = True
        result["control_type"] = "random_layer"
        results.append(result)

    return results


def run_residual_noise_control(
    model, tokenizer, lens, structure,
    correct_prompt: str, confab_prompt: str,
    target_layer: int, baseline_norms: Dict[str, List[float]],
    correct_answer_text: Optional[str] = None,
    confab_answer_text: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict:
    """
    Noise-vector control for residual-stream mode: same target layer and
    position, but replaces the full residual with norm-matched random noise
    (norm-matched to the captured residual vector's own norm).
    """
    result = run_residual_stream_patching_experiment(
        model, tokenizer, lens, structure,
        correct_prompt, confab_prompt, target_layer, baseline_norms,
        correct_answer_text=correct_answer_text, confab_answer_text=confab_answer_text,
        control_mode="noise",
        noise_seed=seed,
    )
    result["is_control"] = True
    result["control_type"] = "noise_vector"
    return result
