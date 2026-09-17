import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
import torch
import torch.nn.functional as F

from src.utils import get_logger, load_json, get_project_root
from src.instrumentation.model_loader import load_phi2, get_model_structure
from src.instrumentation.tuned_lens import load_tuned_lens
from src.instrumentation.trajectory_extractor import extract_trajectory, tokenize_answer
from src.features.shape_descriptors import find_peak_confidence_layer
from src.causal.activation_patching import _make_noise_like

def bootstrap_ci(values, n_boot=2000, ci=95):
    if len(values) == 0:
        return (0.0, 0.0)
    boot_means = []
    arr = np.array(values)
    for _ in range(n_boot):
        sample = np.random.choice(arr, size=len(arr), replace=True)
        boot_means.append(np.mean(sample))
    lower = np.percentile(boot_means, (100 - ci) / 2)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return (float(lower), float(upper))

def main():
    root = get_project_root()
    pairs_path = str(root / "data" / "processed" / "matched_pairs.json")

    model, tokenizer = load_phi2()
    device = next(model.parameters()).device
    structure = get_model_structure(model)
    lens = load_tuned_lens(structure["hidden_dim"], structure["n_layers"])

    matched_pairs = load_json(pairs_path)
    n = len(matched_pairs)
    print(f"Loaded {n} matched pairs for comprehensive causal evaluation.")

    conditions = [
        "target_tl",
        "noise_tl",
        "random_layer",
        "target_tl_plus_1",
        "noise_tl_plus_1",
        "window_3layer_target",
        "window_3layer_noise",
        "residual_target",
        "residual_noise",
    ]

    results = {c: [] for c in conditions}
    spot_checks = []

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

        # Baseline confab logits and probs
        with torch.no_grad():
            f_base_logits = model(**f_in).logits[0, -1]
            f_base_probs = F.softmax(f_base_logits, dim=-1)
            p_f_before = f_base_probs[f_tok].item()
            p_c_before = f_base_probs[c_tok].item()
            orig_argmax = f_base_logits.argmax().item()
            orig_text = tokenizer.decode([orig_argmax]).strip().lower()

        correct_traj = extract_trajectory(model, tokenizer, lens, structure, c_prompt, answer_text=c_ans)
        tl = find_peak_confidence_layer(correct_traj.prob_trajectory)
        tl_jump = min(tl + 1, structure["n_layers"] - 1)

        # Pick random unrelated layer (>4 layers away from tl and tl_jump)
        candidate_rls = [l for l in range(structure["n_layers"]) if abs(l - tl) > 4 and abs(l - tl_jump) > 4]
        rl = random.choice(candidate_rls) if candidate_rls else 5

        if i < 8:
            spot_checks.append({
                "pair_idx": i,
                "relation": pair["relation"],
                "correct_subject": pair["correct_subject"],
                "confab_subject": pair["confab_subject"],
                "peak_layer_tl": tl,
                "jump_layer_tl_plus_1": tl_jump,
                "depth_pct": (tl_jump / structure["n_layers"]) * 100,
            })

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

        def make_mlp_patch_hook(replacement):
            def hook(m, inp, outp):
                patched = outp.clone()
                patched[:, -1, :] = replacement[:, 0, :]
                return patched
            return hook

        cap_mlp = {}
        def make_cap_hook(l_idx):
            def hook(m, inp, outp):
                cap_mlp[l_idx] = outp[:, -1:, :].detach().clone()
            return hook

        # Capture MLPs at relevant layers: tl-1, tl, tl_jump, rl
        needed_layers = list(set([l for l in [tl - 1, tl, tl_jump, rl] if 0 <= l < structure["n_layers"]]))
        h_caps = [structure["layers"][l].mlp.register_forward_hook(make_cap_hook(l)) for l in needed_layers]
        
        # Also capture residual stream at tl_jump
        cap_resid = {}
        def make_resid_cap(m, inp, outp):
            hs = outp[0] if isinstance(outp, tuple) else outp
            cap_resid[tl_jump] = hs[:, -1:, :].detach().clone()
        h_res_cap = structure["layers"][tl_jump].register_forward_hook(make_resid_cap)

        with torch.no_grad():
            model(**c_in)
        for h in h_caps:
            h.remove()
        h_res_cap.remove()

        # 1. Target L_tl
        h1 = [structure["layers"][tl].mlp.register_forward_hook(make_mlp_patch_hook(cap_mlp[tl]))]
        results["target_tl"].append(evaluate_patch(h1))

        # 2. Noise L_tl
        h2 = [structure["layers"][tl].mlp.register_forward_hook(make_mlp_patch_hook(_make_noise_like(cap_mlp[tl])))]
        results["noise_tl"].append(evaluate_patch(h2))

        # 3. Random Layer L_rl
        h3 = [structure["layers"][rl].mlp.register_forward_hook(make_mlp_patch_hook(cap_mlp[rl]))]
        results["random_layer"].append(evaluate_patch(h3))

        # 4. Target L_tl+1 (actual jump layer)
        h4 = [structure["layers"][tl_jump].mlp.register_forward_hook(make_mlp_patch_hook(cap_mlp[tl_jump]))]
        results["target_tl_plus_1"].append(evaluate_patch(h4))

        # 5. Noise L_tl+1
        h5 = [structure["layers"][tl_jump].mlp.register_forward_hook(make_mlp_patch_hook(_make_noise_like(cap_mlp[tl_jump])))]
        results["noise_tl_plus_1"].append(evaluate_patch(h5))

        # 6. Window 3-layer Target [tl-1, tl, tl_jump]
        win_layers = [l for l in [tl - 1, tl, tl_jump] if 0 <= l < structure["n_layers"]]
        h6 = [structure["layers"][l].mlp.register_forward_hook(make_mlp_patch_hook(cap_mlp[l])) for l in win_layers]
        results["window_3layer_target"].append(evaluate_patch(h6))

        # 7. Window 3-layer Noise
        h7 = [structure["layers"][l].mlp.register_forward_hook(make_mlp_patch_hook(_make_noise_like(cap_mlp[l]))) for l in win_layers]
        results["window_3layer_noise"].append(evaluate_patch(h7))

        # 8. Residual Stream Target at tl_jump
        def make_resid_patch_hook(replacement):
            def hook(m, inp, outp):
                is_tuple = isinstance(outp, tuple)
                hs = outp[0] if is_tuple else outp
                patched = hs.clone()
                patched[:, -1, :] = replacement[:, 0, :]
                return (patched,) + outp[1:] if is_tuple else patched
            return hook
        h8 = [structure["layers"][tl_jump].register_forward_hook(make_resid_patch_hook(cap_resid[tl_jump]))]
        results["residual_target"].append(evaluate_patch(h8))

        # 9. Residual Stream Noise at tl_jump
        h9 = [structure["layers"][tl_jump].register_forward_hook(make_resid_patch_hook(_make_noise_like(cap_resid[tl_jump])))]
        results["residual_noise"].append(evaluate_patch(h9))

    print("\n" + "=" * 90)
    print("STEP 4 SPOT-CHECK OF TARGET LAYERS")
    print("=" * 90)
    print(f"{'Idx':<4} {'Relation':<14} {'Correct Subject':<22} {'Confab Subject':<20} {'Peak L_tl':<10} {'Jump L_tl+1':<12} {'Depth %':<8}")
    print("-" * 90)
    for sc in spot_checks:
        print(f"{sc['pair_idx']:<4} {sc['relation']:<14} {sc['correct_subject']:<22} {sc['confab_subject']:<20} {sc['peak_layer_tl']:<10} {sc['jump_layer_tl_plus_1']:<12} {sc['depth_pct']:>6.1f}%")

    print("\n" + "=" * 105)
    print("FULL CAUSAL ACTIVATION PATCHING EXPERIMENT RESULTS (N = 24)")
    print("=" * 105)
    header = f"{'Condition':<28} {'Ans Changed (%)':>16} {'95% CI':>16} {'Flip (%)':>10} {'Confab Drop':>14} {'Correct Gain':>14}"
    print(header)
    print("-" * 105)

    condition_labels = [
        ("Single MLP (Target L_tl)", "target_tl"),
        ("Single MLP (Noise L_tl)", "noise_tl"),
        ("Random Layer Control (L_rl)", "random_layer"),
        ("---------------------------", None),
        ("Jump Layer (Target L_tl+1)", "target_tl_plus_1"),
        ("Jump Layer (Noise L_tl+1)", "noise_tl_plus_1"),
        ("---------------------------", None),
        ("Window 3-Layer (Target)", "window_3layer_target"),
        ("Window 3-Layer (Noise)", "window_3layer_noise"),
        ("---------------------------", None),
        ("Residual Stream (Target)", "residual_target"),
        ("Residual Stream (Noise)", "residual_noise"),
    ]

    for label, key in condition_labels:
        if key is None:
            print("-" * 105)
            continue
        res = results[key]
        chg_list = [1.0 if r["changed"] else 0.0 for r in res]
        chg_rate = np.mean(chg_list) * 100
        ci_low, ci_high = bootstrap_ci(chg_list)
        ci_str = f"[{ci_low*100:.1f}%, {ci_high*100:.1f}%]"
        flip_rate = sum(1 for r in res if r["flipped"]) / len(res) * 100
        avg_drop = np.mean([r["confab_drop"] for r in res])
        avg_gain = np.mean([r["correct_gain"] for r in res])
        print(f"{label:<28} {chg_rate:>15.1f}% {ci_str:>16} {flip_rate:>9.1f}% {avg_drop:>14.4f} {avg_gain:>14.4f}")
    print("=" * 105)

if __name__ == "__main__":
    main()
