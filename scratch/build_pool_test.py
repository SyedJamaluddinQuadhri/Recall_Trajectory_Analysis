import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from collections import defaultdict

with open("data/processed/labeled_qa.json", "r", encoding="utf-8") as f:
    data = json.load(f)

popqa = [d for d in data if d.get("source") == "popqa" and d.get("subject")]

templates = {
    "birthplace": "In what city was {subject} born?",
    "occupation": "What is {subject}'s occupation?",
    "genre": "What genre is {subject}?",
    "father": "Who is the father of {subject}?"
}

total_pairs = 0
for rel, tmpl in templates.items():
    items = [d for d in popqa if d["question"] == tmpl.replace("{subject}", d["subject"])]
    corr = [d for d in items if d["label"] == "correct"]
    conf = [d for d in items if d["label"] == "confabulated"]
    n_pairs = min(len(corr), len(conf))
    total_pairs += n_pairs
    print(f"{rel:<12}: total={len(items):<4} correct={len(corr):<4} confab={len(conf):<4} potential_pairs={n_pairs}")

print(f"Total available matched pairs: {total_pairs}")
