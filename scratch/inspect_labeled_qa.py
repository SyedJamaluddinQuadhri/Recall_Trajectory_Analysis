import json

with open("data/processed/labeled_qa.json", "r", encoding="utf-8") as f:
    data = json.load(f)

templates = [
    "In what city was {subject} born?",
    "What is {subject}'s occupation?",
    "What genre is {subject}?",
    "Who is the father of {subject}?"
]

for template in templates:
    print(f"=== {template} ===")
    matches = [d for d in data if d.get("subject") and d["question"] == template.replace("{subject}", d["subject"])]
    correct = [d for d in matches if d["label"] == "correct"]
    confab = [d for d in matches if d["label"] == "confabulated"]
    print(f"Total: {len(matches)}, Correct: {len(correct)}, Confab: {len(confab)}")
    for d in correct[:3]:
        print("  [CORR]", d["subject"], "-> model_answer:", repr(d["model_answer"]), "gold:", d["gold_answers"], "pop:", d["popularity"])
    for d in confab[:3]:
        print("  [CONF]", d["subject"], "-> model_answer:", repr(d["model_answer"]), "gold:", d["gold_answers"], "pop:", d["popularity"])
