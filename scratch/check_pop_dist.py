import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json

with open("data/processed/labeled_qa.json", "r", encoding="utf-8") as f:
    data = json.load(f)

popqa = [d for d in data if d.get("source") == "popqa" and d.get("subject")]

templates = {
    "birthplace": "In what city was {subject} born?",
    "occupation": "What is {subject}'s occupation?",
    "genre": "What genre is {subject}?",
    "father": "Who is the father of {subject}?"
}

for rel, tmpl in templates.items():
    items = [d for d in popqa if d["question"] == tmpl.replace("{subject}", d["subject"])]
    corr = [d for d in items if d["label"] == "correct"]
    conf = [d for d in items if d["label"] == "confabulated"]
    corr_pops = [d.get("popularity", 0) for d in corr]
    conf_pops = [d.get("popularity", 0) for d in conf]
    import numpy as np
    print(f"=== {rel} ===")
    print(f"  Correct: {len(corr)} | Pop: min={min(corr_pops)}, median={int(np.median(corr_pops))}, max={max(corr_pops)}")
    print(f"  Confab:  {len(conf)} | Pop: min={min(conf_pops)}, median={int(np.median(conf_pops))}, max={max(conf_pops)}")
