import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from src.utils import get_logger, load_json, get_project_root
from src.instrumentation.model_loader import load_phi2, get_model_structure
from src.instrumentation.tuned_lens import load_tuned_lens
from src.instrumentation.trajectory_extractor import extract_trajectory, tokenize_answer, build_teacher_forced_input
from src.features.shape_descriptors import find_peak_confidence_layer
from src.dataset.build_matched_pairs import build_matched_pairs, build_default_subject_list
from src.causal.activation_patching import _make_noise_like

def main():
    root = get_project_root()
    pairs_path = str(root / "data" / "processed" / "matched_pairs.json")

    model, tokenizer = load_phi2()
    device = next(model.parameters()).device
    structure = get_model_structure(model)
    lens = load_tuned_lens(structure["hidden_dim"], structure["n_layers"])

    matched_pairs = load_json(pairs_path)
    n = len(matched_pairs)
    print(f"Running Window & Residual Patching Experiments on {n} matched pairs...")

    # We will test 4 conditions on each pair:
    # 1. Single-layer MLP patch (baseline reference)
    # 2. Window MLP patch (layers tl-1, tl, tl+1)
    # 3. Next-layer MLP patch (layer tl+1, where the confidence jump actually materializes)
    # 4. Residual stream patch (full block output at target_layer)
    #
    # In each condition, we compare TARGET (real content) vs NOISE (norm-matched random noise)

    results = {
        "single_mlp_target": [], "single_mlp_noise": [],
        "next_layer_mlp_target": [], "next_layer_mlp_noise": [],
        "window_mlp_target": [], "window_mlp_noise": [],
        "residual_target": [], "residual_noise": [],
    }

    for i, pair in enumerate(matched_pairs):
        c_prompt = pair["correct_prompt"]
        f_prompt = pair["confab_prompt"]
        c_ans = pair.get("correct_model_answer")
        f_ans = pair.get("confab_model_answer")

        c_tok_ids = tokenize_answer(tokenizer, c_ans)[:8] or [tokenizer.eos_token_id]
        f_tok_ids = tokenize_answer(tokenizer, f_ans)[:8] or [tokenizer.eos_token_id]
        c_tok = c_tok_ids[0]
        f_tok = f_tok_ids[0]

        c_in = tokenizer(c_prompt, return_tensors="pt").to(device)
        f_in = tokenizer(f_prompt, return_tensors="pt").to(device)

        # Baseline confab probabilities before patch
        with torch.no_grad():
            f_base_logits = model(**f_in).logits[0, -1]
            f_base_probs = F.softmax(f_base_logits, dim=-1)
            p_f_before = f_base_probs[f_tok].item()
            p_c_before = f_base_probs[c_tok].item()
            orig_argmax = f_base_logits.argmax().item()
            orig_text = tokenizer.decode([orig_argmax]).strip().lower()

        # Target layer from correct trajectory
        correct_traj = extract_trajectory(model, tokenizer, lens, structure, c_prompt, answer_text=c_ans)
        tl = find_peak_confidence_layer(correct_traj.prob_trajectory)

        def evaluate_patch(hooks):
            with torch.no_grad():
                out = model(**f_in)
                logits = out.logits[0, -1]
                probs = F.softmax(logits, dim=-1)
            for h in hooks:
                h.remove()
            p_f_after = probs[f_tok].item()
            p_c_after = probs[c_tok].item()
            new_argmax = logits.argmax().item()
            new_text = tokenizer.decode([new_argmax]).strip().lower()

            confab_drop = p_f_before - p_f_after
            correct_gain = p_c_after - p_c_before
            flipped = (new_text != orig_text)
            chg = flipped or (confab_drop >= 0.10) or (correct_gain >= 0.05)
            return {"changed": chg, "flipped": flipped, "confab_drop": confab_drop, "correct_gain": correct_gain}

        # ── 1. Single Layer MLP Patch (L_tl) ───────────────────────────────────
        cap_mlp = {}
        def make_cap_hook(l_idx):
            def hook(m, inp, outp):
                cap_mlp[l_idx] = outp[:, -1:, :].detach().clone()
            return hook

        h_cap = structure["layers"][tl].mlp.register_forward_hook(make_cap_hook(tl))
        with torch.no_grad():
            model(**c_in)
        h_cap.remove()

        def make_mlp_patch_hook(replacement):
            def hook(m, inp, outp):
                patched = outp.clone()
                patched[:, -1, :] = replacement[:, 0, :]
                return patched
            return hook

        # Target
        h_patch = [structure["layers"][tl].mlp.register_forward_hook(make_mlp_patch_hook(cap_mlp[tl]))]
        results["single_mlp_target"].append(evaluate_patch(h_patch))

        # Noise
        noise_single = _make_noise_like(cap_mlp[tl])
        h_patch_n = [structure["layers"][tl].mlp.register_forward_hook(make_mlp_patch_hook(noise_single))]
        results["single_mlp_noise"].append(evaluate_patch(h_patch_n))

        # ── 2. Next-Layer MLP Patch (L_tl+1) ────────────────────────────────────
        tl_next = min(tl + 1, structure["n_layers"] - 1)
        h_cap_next = structure["layers"][tl_next].mlp.register_forward_hook(make_cap_hook(tl_next))
        with torch.no_grad():
            model(**c_in)
        h_cap_next.remove()

        h_patch_next = [structure["layers"][tl_next].mlp.register_forward_hook(make_mlp_patch_hook(cap_mlp[tl_next]))]
        results["next_layer_mlp_target"].append(evaluate_patch(h_patch_next))

        noise_next = _make_noise_like(cap_mlp[tl_next])
        h_patch_next_n = [structure["layers"][tl_next].mlp.register_forward_hook(make_mlp_patch_hook(noise_next))]
        results["next_layer_mlp_noise"].append(evaluate_patch(h_patch_next_n))

        # ── 3. Window MLP Patch (tl-1, tl, tl+1) ──────────────────────────────
        win_layers = [l for l in [tl - 1, tl, tl + 1] if 0 <= l < structure["n_layers"]]
        h_cap_win = [structure["layers"][l].mlp.register_forward_hook(make_cap_hook(l)) for l in win_layers]
        with torch.no_grad():
            model(**c_in)
        for h in h_cap_win:
            h.remove()

        h_patch_win = [structure["layers"][l].mlp.register_forward_hook(make_mlp_patch_hook(cap_mlp[l])) for l in win_layers]
        results["window_mlp_target"].append(evaluate_patch(h_patch_win))

        h_patch_win_n = [structure["layers"][l].mlp.register_forward_hook(make_mlp_patch_hook(_make_noise_like(cap_mlp[l]))) for l in win_layers]
        results["window_mlp_noise"].append(evaluate_patch(h_patch_win_n))

        # ── 4. Full Residual Stream Patch (whole block output at tl) ───────────
        # In transformers, block forward returns tuple (hidden_states, ...) or hidden_states
        cap_resid = {}
        def make_resid_cap_hook(l_idx):
            def hook(m, inp, outp):
                hs = outp[0] if isinstance(outp, tuple) else outp
                cap_resid[l_idx] = hs[:, -1:, :].detach().clone()
            return hook

        h_cap_res = structure["layers"][tl].register_forward_hook(make_resid_cap_hook(tl))
        with torch.no_grad():
            model(**c_in)
        h_cap_res.remove()

        def make_resid_patch_hook(replacement):
            def hook(m, inp, outp):
                is_tuple = isinstance(outp, tuple)
                hs = outp[0] if is_tuple else outp
                patched_hs = hs.clone()
                patched_hs[:, -1, :] = replacement[:, 0, :]
                if is_tuple:
                    return (patched_hs,) + outp[1:]
                return patched_hs
            return hook

        h_patch_res = [structure["layers"][tl].register_forward_hook(make_resid_patch_hook(cap_resid[tl]))]
        results["residual_target"].append(evaluate_patch(h_patch_res))

        noise_resid = _make_noise_like(cap_resid[tl])
        h_patch_res_n = [structure["layers"][tl].register_forward_hook(make_resid_patch_hook(noise_resid))]
        results["residual_noise"].append(evaluate_patch(h_patch_res_n))

        if (i + 1) % 5 == 0 or (i + 1) == n:
            print(f"Evaluated {i+1}/{n} pairs...")

    # Print Comparison Table
    print("\n" + "=" * 85)
    print("STEP 2 & 3: ARCHITECTURAL INTERVENTION COMPARISON TABLE")
    print("=" * 85)
    print(f"{'Condition':<28} {'Ans Changed':>13} {'Flip Rate':>12} {'Confab Drop':>14} {'Correct Gain':>14}")
    print("-" * 85)

    def print_row(name, res):
        k = len(res)
        chg = sum(1 for r in res if r["changed"]) / k * 100
        flip = sum(1 for r in res if r["flipped"]) / k * 100
        drop = np.mean([r["confab_drop"] for r in res])
        gain = np.mean([r["correct_gain"] for r in res])
        print(f"{name:<28} {chg:>12.1f}% {flip:>11.1f}% {drop:>14.4f} {gain:>14.4f}")

    print_row("Single MLP (Target L_tl)", results["single_mlp_target"])
    print_row("Single MLP (Noise)", results["single_mlp_noise"])
    print("-" * 85)
    print_row("Next-Layer MLP (L_tl+1)", results["next_layer_mlp_target"])
    print_row("Next-Layer MLP (Noise)", results["next_layer_mlp_noise"])
    print("-" * 85)
    print_row("Window MLP (3-layer Target)", results["window_mlp_target"])
    print_row("Window MLP (3-layer Noise)", results["window_mlp_noise"])
    print("-" * 85)
    print_row("Residual Stream (Target)", results["residual_target"])
    print_row("Residual Stream (Noise)", results["residual_noise"])
    print("=" * 85)

if __name__ == "__main__":
    main()
