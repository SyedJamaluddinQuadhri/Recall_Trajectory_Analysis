import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from src.instrumentation.model_loader import load_phi2
from src.dataset.build_matched_pairs import generate_direct_answer, RELATION_TEMPLATES
from src.dataset.build_factual_qa import exact_match

with open("data/processed/labeled_qa.json", "r", encoding="utf-8") as f:
    data = json.load(f)

model, tokenizer = load_phi2()

popqa = [d for d in data if d.get("source") == "popqa" and d.get("subject")]

# Let's test the 4 relation templates
templates = [
    {
        "relation": "birthplace",
        "exemplars": "Q: What city was Stephen Hawking born in?\nA: Oxford\n\nQ: What city was Niels Bohr born in?\nA: Copenhagen\n\n",
        "format": "Q: What city was {subject} born in?\nA:",
        "popqa_filter": lambda d: "In what city was" in d["question"],
    },
    {
        "relation": "occupation",
        "exemplars": "Q: What is Leonardo da Vinci's occupation?\nA: Artist\n\nQ: What is Albert Einstein's occupation?\nA: Physicist\n\n",
        "format": "Q: What is {subject}'s occupation?\nA:",
        "popqa_filter": lambda d: "What is" in d["question"] and "occupation" in d["question"],
    },
    {
        "relation": "genre",
        "exemplars": "Q: What genre is Queen?\nA: Rock\n\nQ: What genre is Miles Davis?\nA: Jazz\n\n",
        "format": "Q: What genre is {subject}?\nA:",
        "popqa_filter": lambda d: "What genre is" in d["question"],
    },
    {
        "relation": "father",
        "exemplars": "Q: Who is the father of Alexander the Great?\nA: Philip II of Macedon\n\nQ: Who is the father of Elizabeth I?\nA: Henry VIII\n\n",
        "format": "Q: Who is the father of {subject}?\nA:",
        "popqa_filter": lambda d: "Who is the father of" in d["question"],
    }
]

for t in templates:
    rel = t["relation"]
    items = [d for d in popqa if t["popqa_filter"](d)][:30]
    n_corr, n_conf = 0, 0
    for it in items:
        p = t["exemplars"] + t["format"].format(subject=it["subject"])
        ans = generate_direct_answer(model, tokenizer, p)
        is_corr = exact_match(ans, it["gold_answers"])
        if is_corr:
            n_corr += 1
        else:
            n_conf += 1
    print(f"Relation: {rel:<12} (tested 30): correct={n_corr:<2} confab={n_conf:<2}")
