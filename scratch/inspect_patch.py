import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from src.utils import load_json
from src.instrumentation.model_loader import load_phi2, get_model_structure
from src.instrumentation.tuned_lens import load_tuned_lens
from src.instrumentation.trajectory_extractor import (
    extract_trajectory, tokenize_answer, build_teacher_forced_input
)
from src.features.shape_descriptors import find_peak_confidence_layer

def test_pair(idx=0):
    pairs = load_json("data/processed/matched_pairs.json")
    pair = pairs[idx]
    print(f"=== Testing Pair {idx} ===")
    print("Relation:", pair["relation"])
    print("Correct Prompt:", pair["correct_prompt"])
    print("Correct Answer:", pair["correct_model_answer"])
    print("Confab Prompt:", pair["confab_prompt"])
    print("Confab Answer:", pair["confab_model_answer"])

    model, tokenizer = load_phi2()
    device = next(model.parameters()).device
    structure = get_model_structure(model)
    lens = load_tuned_lens(structure["hidden_dim"], structure["n_layers"])

    correct_traj = extract_trajectory(
        model, tokenizer, lens, structure, pair["correct_prompt"],
        answer_text=pair["correct_model_answer"],
    )
    target_layer = find_peak_confidence_layer(correct_traj.prob_trajectory)
    print(f"Target layer: {target_layer}")
    print(f"Correct prob traj around target layer {target_layer}:")
    for l in range(max(0, target_layer-2), min(32, target_layer+3)):
        print(f"  Layer {l}: prob = {correct_traj.prob_trajectory[l]:.4f}")

    # Inspect tokenization
    c_ids = tokenizer(pair["correct_prompt"], return_tensors="pt").input_ids[0].tolist()
    f_ids = tokenizer(pair["confab_prompt"], return_tensors="pt").input_ids[0].tolist()
    print("Correct prompt tokens:", [tokenizer.decode([t]) for t in c_ids])
    print("Confab prompt tokens:", [tokenizer.decode([t]) for t in f_ids])

    # Let's see what model generates for both prompts directly
    with torch.no_grad():
        c_in = tokenizer(pair["correct_prompt"], return_tensors="pt").to(device)
        f_in = tokenizer(pair["confab_prompt"], return_tensors="pt").to(device)
        c_out = model.generate(**c_in, max_new_tokens=8, do_sample=False)
        f_out = model.generate(**f_in, max_new_tokens=8, do_sample=False)
        print("Model gen from correct prompt:", tokenizer.decode(c_out[0][c_in.input_ids.shape[1]:]))
        print("Model gen from confab prompt:", tokenizer.decode(f_out[0][f_in.input_ids.shape[1]:]))

        # Check logits at prompt[-1]
        out_f = model(**f_in)
        logits_f = out_f.logits[0, -1]
        probs_f = F.softmax(logits_f, dim=-1)
        top5_f = torch.topk(probs_f, 5)
        print("Top 5 from unpatched confab prompt:")
        for idx, (val, tok) in enumerate(zip(top5_f.values, top5_f.indices)):
            print(f"  {idx+1}: {tokenizer.decode([tok])!r} (prob={val.item():.4f}, id={tok.item()})")

    # Now capture MLP at target_layer for correct_prompt
    captured_mlp = None
    def capture_hook(m, inp, outp):
        nonlocal captured_mlp
        captured_mlp = outp[:, -1:, :].detach().clone()
    
    h_cap = structure["layers"][target_layer].mlp.register_forward_hook(capture_hook)
    with torch.no_grad():
        model(**c_in)
    h_cap.remove()

    print(f"Captured MLP norm: {captured_mlp.norm().item():.4f}")

    # Patch into confab prompt
    def patch_hook(m, inp, outp):
        patched = outp.clone()
        patched[:, -1, :] = captured_mlp[:, 0, :]
        return patched

    h_patch = structure["layers"][target_layer].mlp.register_forward_hook(patch_hook)
    with torch.no_grad():
        out_f_patched = model(**f_in)
        logits_patched = out_f_patched.logits[0, -1]
        probs_patched = F.softmax(logits_patched, dim=-1)
        top5_p = torch.topk(probs_patched, 5)
        print("Top 5 from PATCHED confab prompt:")
        for idx, (val, tok) in enumerate(zip(top5_p.values, top5_p.indices)):
            print(f"  {idx+1}: {tokenizer.decode([tok])!r} (prob={val.item():.4f}, id={tok.item()})")
        
        # Test generate with patch hook active!
        gen_patched = model.generate(**f_in, max_new_tokens=8, do_sample=False)
        print("Model gen with patch active:", tokenizer.decode(gen_patched[0][f_in.input_ids.shape[1]:]))
    h_patch.remove()

if __name__ == "__main__":
    test_pair(0)
