import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import torch
import torch.nn.functional as F

from src.instrumentation.model_loader import load_phi2, get_model_structure
from src.instrumentation.tuned_lens import load_tuned_lens
from src.instrumentation.trajectory_extractor import extract_trajectory, tokenize_answer
from src.features.shape_descriptors import find_peak_confidence_layer
from src.causal.activation_patching import _make_noise_like

with open("data/processed/labeled_qa.json", "r", encoding="utf-8") as f:
    data = json.load(f)

model, tokenizer = load_phi2()
device = next(model.parameters()).device
structure = get_model_structure(model)
lens = load_tuned_lens(structure["hidden_dim"], structure["n_layers"])

popqa = [d for d in data if d.get("source") == "popqa" and d.get("subject")]
tmpl = "In what city was {subject} born?"
items = [d for d in popqa if d["question"] == tmpl.replace("{subject}", d["subject"])]
corr = [d for d in items if d["label"] == "correct"]
conf = [d for d in items if d["label"] == "confabulated"]

print(f"Testing 5 pairs directly from labeled_qa (zero-shot prompt):")
for i in range(5):
    c_item = corr[i]
    f_item = conf[i]

    c_prompt = f"Q: {c_item['question']}\nA:"
    f_prompt = f"Q: {f_item['question']}\nA:"
    c_ans = c_item["model_answer"]
    f_ans = f_item["model_answer"]

    c_tok = tokenize_answer(tokenizer, c_ans)[0]
    f_tok = tokenize_answer(tokenizer, f_ans)[0]

    c_in = tokenizer(c_prompt, return_tensors="pt").to(device)
    f_in = tokenizer(f_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        f_logits = model(**f_in).logits[0, -1]
        probs = F.softmax(f_logits, dim=-1)
        p_f_base = probs[f_tok].item()
        p_c_base = probs[c_tok].item()

    c_traj = extract_trajectory(model, tokenizer, lens, structure, c_prompt, answer_text=c_ans)
    tl = find_peak_confidence_layer(c_traj.prob_trajectory)

    # Capture MLP at tl
    cap = {}
    def hook(m, inp, outp):
        cap[tl] = outp[:, -1:, :].detach().clone()
    h = structure["layers"][tl].mlp.register_forward_hook(hook)
    with torch.no_grad():
        model(**c_in)
    h.remove()

    def eval_patch(replacement):
        def p_hook(m, inp, outp):
            patched = outp.clone()
            patched[:, -1, :] = replacement[:, 0, :]
            return patched
        hp = structure["layers"][tl].mlp.register_forward_hook(p_hook)
        with torch.no_grad():
            out = model(**f_in)
            p = F.softmax(out.logits[0, -1], dim=-1)
        hp.remove()
        return p[f_tok].item(), p[c_tok].item()

    p_f_tgt, p_c_tgt = eval_patch(cap[tl])
    p_f_noi, p_c_noi = eval_patch(_make_noise_like(cap[tl]))

    print(f"Pair {i}: {c_item['subject']} ({tokenizer.decode([c_tok])}) -> {f_item['subject']} ({tokenizer.decode([f_tok])}) | Layer: {tl}")
    print(f"  Target: confab_drop = {p_f_base - p_f_tgt:+.4f}, correct_gain = {p_c_tgt - p_c_base:+.4f}")
    print(f"  Noise:  confab_drop = {p_f_base - p_f_noi:+.4f}, correct_gain = {p_c_noi - p_c_base:+.4f}")
