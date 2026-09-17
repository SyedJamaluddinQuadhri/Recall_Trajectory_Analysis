import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import torch
from src.instrumentation.model_loader import load_phi2
from src.dataset.build_matched_pairs import generate_direct_answer
from src.dataset.build_factual_qa import exact_match

with open("data/processed/labeled_qa.json", "r", encoding="utf-8") as f:
    data = json.load(f)

model, tokenizer = load_phi2()

# Let's test birthplace template
exemplars_birth = (
    "Q: What city was Stephen Hawking born in?\nA: Oxford\n\n"
    "Q: What city was Niels Bohr born in?\nA: Copenhagen\n\n"
)
fmt_birth = "Q: What city was {subject} born in?\nA:"

birth_items = [d for d in data if d.get("subject") and "In what city was" in d["question"]][:10]

print("Testing direct answer generation on 10 subjects from labeled_qa.json:")
for item in birth_items:
    subj = item["subject"]
    prompt = exemplars_birth + fmt_birth.format(subject=subj)
    ans = generate_direct_answer(model, tokenizer, prompt)
    match = exact_match(ans, item["gold_answers"])
    print(f"Subject: {subj:<25} | Gold: {str(item['gold_answers']):<25} | Direct Ans: {ans:<15} | Match: {match}")
