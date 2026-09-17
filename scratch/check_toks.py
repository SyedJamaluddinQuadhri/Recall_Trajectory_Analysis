import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from transformers import AutoTokenizer
from src.instrumentation.trajectory_extractor import tokenize_answer

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

with open("data/processed/labeled_qa.json", "r", encoding="utf-8") as f:
    data = json.load(f)

templates = {
    "birthplace": "In what city was {subject} born?",
    "occupation": "What is {subject}'s occupation?",
    "genre": "What genre is {subject}?",
    "father": "Who is the father of {subject}?"
}

for rel, tmpl in templates.items():
    print(f"=== {rel} ===")
    items = [d for d in data if d.get("subject") and d["question"] == tmpl.replace("{subject}", d["subject"])]
    corr = [d for d in items if d["label"] == "correct"][:3]
    conf = [d for d in items if d["label"] == "confabulated"][:3]
    for c in corr:
        toks = tokenize_answer(tokenizer, c["model_answer"])
        first_tok_str = tokenizer.decode([toks[0]]) if toks else ""
        print(f"  [CORR] {c['subject']:<20} ans: {repr(c['model_answer']):<35} first_tok: {repr(first_tok_str)}")
    for c in conf:
        toks = tokenize_answer(tokenizer, c["model_answer"])
        first_tok_str = tokenizer.decode([toks[0]]) if toks else ""
        print(f"  [CONF] {c['subject']:<20} ans: {repr(c['model_answer']):<35} first_tok: {repr(first_tok_str)}")
