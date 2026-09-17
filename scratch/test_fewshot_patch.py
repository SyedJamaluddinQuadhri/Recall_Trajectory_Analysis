import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from src.instrumentation.model_loader import load_phi2, get_model_structure
from src.instrumentation.tuned_lens import load_tuned_lens
from src.instrumentation.trajectory_extractor import extract_trajectory
from src.features.shape_descriptors import find_peak_confidence_layer

def main():
    model, tokenizer = load_phi2()
    device = next(model.parameters()).device
    structure = get_model_structure(model)
    lens = load_tuned_lens(structure["hidden_dim"], structure["n_layers"])

    # 2-shot prompt format where model answers directly with the entity
    p_correct = """Q: In what city was Stephen Hawking born?
A: Oxford

Q: In what city was Niels Bohr born?
A: Copenhagen

Q: In what city was Marie Curie born?
A:"""

    p_confab = """Q: In what city was Stephen Hawking born?
A: Oxford

Q: In what city was Niels Bohr born?
A: Copenhagen

Q: In what city was Srinivasa Ramanujan born?
A:"""

    c_in = tokenizer(p_correct, return_tensors="pt").to(device)
    f_in = tokenizer(p_confab, return_tensors="pt").to(device)

    with torch.no_grad():
        c_logits = model(**c_in).logits[0, -1]
        f_logits = model(**f_in).logits[0, -1]

    c_top = torch.topk(F.softmax(c_logits, dim=-1), 5)
    f_top = torch.topk(F.softmax(f_logits, dim=-1), 5)

    print("Correct (Marie Curie) top 5:")
    for v, t in zip(c_top.values, c_top.indices):
        print(f"  {tokenizer.decode([t])!r}: {v.item():.4f}")

    print("\nConfab (Srinivasa Ramanujan) top 5:")
    for v, t in zip(f_top.values, f_top.indices):
        print(f"  {tokenizer.decode([t])!r}: {v.item():.4f}")

    # Extract trajectory for correct example to find peak layer
    correct_traj = extract_trajectory(model, tokenizer, lens, structure, p_correct, answer_text="Warsaw")
    target_layer = find_peak_confidence_layer(correct_traj.prob_trajectory)
    print(f"\nTarget layer: {target_layer}")

    # Capture MLP at target_layer
    captured = None
    def cap_hook(m, inp, outp):
        nonlocal captured
        captured = outp[:, -1:, :].detach().clone()
    
    h = structure["layers"][target_layer].mlp.register_forward_hook(cap_hook)
    with torch.no_grad():
        model(**c_in)
    h.remove()

    # Patch into confab at target_layer
    def patch_hook(m, inp, outp):
        patched = outp.clone()
        patched[:, -1, :] = captured[:, 0, :]
        return patched

    h_patch = structure["layers"][target_layer].mlp.register_forward_hook(patch_hook)
    with torch.no_grad():
        f_patched_logits = model(**f_in).logits[0, -1]
    h_patch.remove()

    patched_top = torch.topk(F.softmax(f_patched_logits, dim=-1), 5)
    print("\nPatched Confab top 5 (Target layer):")
    for v, t in zip(patched_top.values, patched_top.indices):
        print(f"  {tokenizer.decode([t])!r}: {v.item():.4f}")

    # Random layer control
    rand_layer = 10 if target_layer != 10 else 5
    captured_rand = None
    def cap_rand(m, inp, outp):
        nonlocal captured_rand
        captured_rand = outp[:, -1:, :].detach().clone()
    hr = structure["layers"][rand_layer].mlp.register_forward_hook(cap_rand)
    with torch.no_grad():
        model(**c_in)
    hr.remove()

    def patch_rand(m, inp, outp):
        patched = outp.clone()
        patched[:, -1, :] = captured_rand[:, 0, :]
        return patched
    hr_p = structure["layers"][rand_layer].mlp.register_forward_hook(patch_rand)
    with torch.no_grad():
        f_rand_logits = model(**f_in).logits[0, -1]
    hr_p.remove()

    rand_top = torch.topk(F.softmax(f_rand_logits, dim=-1), 5)
    print(f"\nPatched Confab top 5 (Random layer {rand_layer}):")
    for v, t in zip(rand_top.values, rand_top.indices):
        print(f"  {tokenizer.decode([t])!r}: {v.item():.4f}")

if __name__ == "__main__":
    main()
