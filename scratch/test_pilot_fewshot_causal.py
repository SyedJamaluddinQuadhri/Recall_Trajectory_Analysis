import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
from src.instrumentation.model_loader import load_phi2, get_model_structure
from src.instrumentation.tuned_lens import load_tuned_lens
from src.instrumentation.trajectory_extractor import extract_trajectory
from src.features.shape_descriptors import find_peak_confidence_layer
from src.causal.activation_patching import _make_noise_like

def main():
    model, tokenizer = load_phi2()
    device = next(model.parameters()).device
    structure = get_model_structure(model)
    lens = load_tuned_lens(structure["hidden_dim"], structure["n_layers"])

    # Test 5 matched pairs with 2-shot direct answer format
    exemplars = """Q: What city was Stephen Hawking born in?
A: Oxford

Q: What city was Niels Bohr born in?
A: Copenhagen

"""
    pairs = [
        # (correct_subject, correct_gold, confab_subject, confab_gold)
        ("Marie Curie", "Warsaw", "Srinivasa Ramanujan", "Erode"),
        ("Albert Einstein", "Ulm", "Charles Darwin", "Shrewsbury"),
        ("Galileo Galilei", "Pisa", "Max Planck", "Kiel"),
        ("Nikola Tesla", "Smiljan", "Emmy Noether", "Erlangen"),
        ("Alan Turing", "London", "Dorothy Hodgkin", "Cairo"),
    ]

    target_results = []
    layer_results = []
    noise_results = []

    for c_subj, c_gold, f_subj, f_gold in pairs:
        c_prompt = exemplars + f"Q: What city was {c_subj} born in?\nA:"
        f_prompt = exemplars + f"Q: What city was {f_subj} born in?\nA:"

        c_in = tokenizer(c_prompt, return_tensors="pt").to(device)
        f_in = tokenizer(f_prompt, return_tensors="pt").to(device)

        # Baseline predictions
        with torch.no_grad():
            f_logits_before = model(**f_in).logits[0, -1]
            f_probs_before = F.softmax(f_logits_before, dim=-1)
            orig_argmax = f_logits_before.argmax().item()
            orig_tok_text = tokenizer.decode([orig_argmax]).strip().lower()

        # Find target layer from correct example
        c_traj = extract_trajectory(model, tokenizer, lens, structure, c_prompt, answer_text=c_gold)
        target_layer = find_peak_confidence_layer(c_traj.prob_trajectory)

        # 1. Target Layer Patch
        captured = None
        def cap_h(m, inp, outp):
            nonlocal captured
            captured = outp[:, -1:, :].detach().clone()
        h_c = structure["layers"][target_layer].mlp.register_forward_hook(cap_h)
        with torch.no_grad():
            model(**c_in)
        h_c.remove()

        def patch_h(m, inp, outp):
            patched = outp.clone()
            patched[:, -1, :] = captured[:, 0, :]
            return patched
        h_p = structure["layers"][target_layer].mlp.register_forward_hook(patch_h)
        with torch.no_grad():
            f_logits_target = model(**f_in).logits[0, -1]
        h_p.remove()

        f_probs_target = F.softmax(f_logits_target, dim=-1)
        new_argmax_target = f_logits_target.argmax().item()
        target_flipped = (tokenizer.decode([new_argmax_target]).strip().lower() != orig_tok_text)
        target_drop = (f_probs_before[orig_argmax] - f_probs_target[orig_argmax]).item()

        # 2. Random Layer Control
        rand_layer = 10 if target_layer != 10 else 5
        captured_rand = None
        def cap_r(m, inp, outp):
            nonlocal captured_rand
            captured_rand = outp[:, -1:, :].detach().clone()
        h_cr = structure["layers"][rand_layer].mlp.register_forward_hook(cap_r)
        with torch.no_grad():
            model(**c_in)
        h_cr.remove()

        def patch_r(m, inp, outp):
            patched = outp.clone()
            patched[:, -1, :] = captured_rand[:, 0, :]
            return patched
        h_pr = structure["layers"][rand_layer].mlp.register_forward_hook(patch_r)
        with torch.no_grad():
            f_logits_rand = model(**f_in).logits[0, -1]
        h_pr.remove()

        f_probs_rand = F.softmax(f_logits_rand, dim=-1)
        new_argmax_rand = f_logits_rand.argmax().item()
        rand_flipped = (tokenizer.decode([new_argmax_rand]).strip().lower() != orig_tok_text)
        rand_drop = (f_probs_before[orig_argmax] - f_probs_rand[orig_argmax]).item()

        # 3. Noise Vector Control
        noise = _make_noise_like(captured)
        def patch_noise(m, inp, outp):
            patched = outp.clone()
            patched[:, -1, :] = noise[:, 0, :]
            return patched
        h_pn = structure["layers"][target_layer].mlp.register_forward_hook(patch_noise)
        with torch.no_grad():
            f_logits_noise = model(**f_in).logits[0, -1]
        h_pn.remove()

        f_probs_noise = F.softmax(f_logits_noise, dim=-1)
        new_argmax_noise = f_logits_noise.argmax().item()
        noise_flipped = (tokenizer.decode([new_argmax_noise]).strip().lower() != orig_tok_text)
        noise_drop = (f_probs_before[orig_argmax] - f_probs_noise[orig_argmax]).item()

        print(f"Pair ({c_subj} -> {f_subj}):")
        print(f"  Orig: {tokenizer.decode([orig_argmax])!r} (p={f_probs_before[orig_argmax]:.3f})")
        print(f"  Target (L{target_layer}): {tokenizer.decode([new_argmax_target])!r} (p={f_probs_target[orig_argmax]:.3f}, drop={target_drop:+.3f}, flip={target_flipped})")
        print(f"  Random (L{rand_layer}):   {tokenizer.decode([new_argmax_rand])!r} (p={f_probs_rand[orig_argmax]:.3f}, drop={rand_drop:+.3f}, flip={rand_flipped})")
        print(f"  Noise  (L{target_layer}): {tokenizer.decode([new_argmax_noise])!r} (p={f_probs_noise[orig_argmax]:.3f}, drop={noise_drop:+.3f}, flip={noise_flipped})")

        target_results.append({"flipped": target_flipped, "drop": target_drop})
        layer_results.append({"flipped": rand_flipped, "drop": rand_drop})
        noise_results.append({"flipped": noise_flipped, "drop": noise_drop})

    print("\n=== Summary (5 pairs) ===")
    print(f"Target: flip={sum(r['flipped'] for r in target_results)}/5, avg_drop={np.mean([r['drop'] for r in target_results]):.4f}")
    print(f"Random: flip={sum(r['flipped'] for r in layer_results)}/5, avg_drop={np.mean([r['drop'] for r in layer_results]):.4f}")
    print(f"Noise:  flip={sum(r['flipped'] for r in noise_results)}/5, avg_drop={np.mean([r['drop'] for r in noise_results]):.4f}")

if __name__ == "__main__":
    main()
