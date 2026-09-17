import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import random

from src.utils import get_logger, get_project_root, save_json
from src.instrumentation.model_loader import load_phi2
from src.dataset.build_matched_pairs import generate_direct_answer, build_default_subject_list
from src.dataset.build_factual_qa import exact_match, is_refusal

root = get_project_root()
labeled_path = str(root / "data" / "processed" / "labeled_qa.json")
output_path = str(root / "data" / "processed" / "matched_pairs.json")

with open(labeled_path, "r", encoding="utf-8") as f:
    labeled = json.load(f)

popqa = [d for d in labeled if d.get("source") == "popqa" and d.get("subject")]

templates = [
    {
        "relation": "birthplace",
        "exemplars": "Q: What city was Stephen Hawking born in?\nA: Oxford\n\nQ: What city was Niels Bohr born in?\nA: Copenhagen\n\n",
        "format": "Q: What city was {subject} born in?\nA:",
        "target_pairs": 95,
    },
    {
        "relation": "occupation",
        "exemplars": "Q: What was Leonardo da Vinci's primary occupation?\nA: Artist\n\nQ: What was Albert Einstein's primary occupation?\nA: Physicist\n\n",
        "format": "Q: What was {subject}'s primary occupation?\nA:",
        "target_pairs": 120,
    },
    {
        "relation": "genre",
        "exemplars": "Q: What genre is Queen?\nA: Rock\n\nQ: What genre is Miles Davis?\nA: Jazz\n\n",
        "format": "Q: What genre is {subject}?\nA:",
        "target_pairs": 270,
    },
    {
        "relation": "father",
        "exemplars": "Q: Who is the father of Alexander the Great?\nA: Philip II of Macedon\n\nQ: Who is the father of Charlemagne?\nA: Pepin the Short\n\n",
        "format": "Q: Who is the father of {subject}?\nA:",
        "target_pairs": 30,
    },
    {
        "relation": "nationality",
        "exemplars": "Q: What nationality was Winston Churchill?\nA: British\n\nQ: What nationality was Napoleon Bonaparte?\nA: French\n\n",
        "format": "Q: What nationality was {subject}?\nA:",
        "target_pairs": 15,
    },
    {
        "relation": "death_year",
        "exemplars": "Q: In what year did George Washington die?\nA: 1799\n\nQ: In what year did Abraham Lincoln die?\nA: 1865\n\n",
        "format": "Q: In what year did {subject} die?\nA:",
        "target_pairs": 15,
    },
    {
        "relation": "founded_by",
        "exemplars": "Q: Who founded Microsoft?\nA: Bill Gates\n\nQ: Who founded Apple?\nA: Steve Jobs\n\n",
        "format": "Q: Who founded {subject}?\nA:",
        "target_pairs": 15,
    },
]

by_relation = {}

for d in popqa:
    q = d["question"]
    s = d["subject"]
    if "In what city was" in q:
        by_relation.setdefault("birthplace", []).append({
            "subject": s, "relation": "birthplace", "gold": d["gold_answers"][0],
            "gold_answers": d["gold_answers"], "popularity": d.get("popularity", 0)
        })
    elif "occupation" in q.lower():
        by_relation.setdefault("occupation", []).append({
            "subject": s, "relation": "occupation", "gold": d["gold_answers"][0],
            "gold_answers": d["gold_answers"], "popularity": d.get("popularity", 0)
        })
    elif "genre" in q.lower():
        by_relation.setdefault("genre", []).append({
            "subject": s, "relation": "genre", "gold": d["gold_answers"][0],
            "gold_answers": d["gold_answers"], "popularity": d.get("popularity", 0)
        })
    elif "father" in q.lower():
        by_relation.setdefault("father", []).append({
            "subject": s, "relation": "father", "gold": d["gold_answers"][0],
            "gold_answers": d["gold_answers"], "popularity": d.get("popularity", 0)
        })

# Also add curated seeds for all relations
seeds = build_default_subject_list()
for s in seeds:
    by_relation.setdefault(s["relation"], []).append({
        **s,
        "gold_answers": [s["gold"]],
        "popularity": 1000000,
    })

print("Subjects available per relation:")
for r, items in by_relation.items():
    print(f"  {r:<14}: {len(items)}")

# Load existing pairs to avoid re-generating what is already built
if os.path.exists(output_path):
    with open(output_path, "r", encoding="utf-8") as f:
        matched_pairs = json.load(f)
    print(f"Loaded {len(matched_pairs)} existing pairs from {output_path}")
else:
    matched_pairs = []

model, tokenizer = load_phi2()

for rel_entry in templates:
    rel = rel_entry["relation"]
    existing_count = sum(1 for p in matched_pairs if p["relation"] == rel)
    target_n = rel_entry["target_pairs"]
    if existing_count >= target_n:
        print(f"Skipping '{rel}': already have {existing_count} >= {target_n} pairs.")
        continue

    target_needed = target_n - existing_count
    existing_subjects = {p["correct_subject"] for p in matched_pairs} | {p["confab_subject"] for p in matched_pairs}
    all_items = by_relation.get(rel, [])
    items = [it for it in all_items if it["subject"] not in existing_subjects]
    if not items:
        continue

    print(f"\nProcessing '{rel}' (have {existing_count}, targeting {target_needed} more)...", flush=True)

    items_by_pop = sorted(items, key=lambda x: x.get("popularity", 0), reverse=True)
    high_pop = items_by_pop[:max(1, len(items_by_pop)//2)]
    low_pop = items_by_pop[max(1, len(items_by_pop)//2):]

    correct_items = []
    confab_items = []

    # Get correct items from high pop
    for idx, it in enumerate(high_pop):
        p = rel_entry["exemplars"] + rel_entry["format"].format(subject=it["subject"])
        ans = generate_direct_answer(model, tokenizer, p)
        if is_refusal(ans):
            continue
        corr = exact_match(ans, it.get("gold_answers", [it["gold"]]))
        rec = {**it, "prompt": p, "model_answer": ans}
        if corr:
            correct_items.append(rec)
        else:
            confab_items.append(rec)
        if len(correct_items) % 10 == 0 and len(correct_items) > 0:
            print(f"    [high-pop {idx+1}/{len(high_pop)}] Correct: {len(correct_items)}/{target_needed}...", flush=True)
        if len(correct_items) >= target_needed:
            break

    # Get confab items from low pop
    for idx, it in enumerate(low_pop):
        p = rel_entry["exemplars"] + rel_entry["format"].format(subject=it["subject"])
        ans = generate_direct_answer(model, tokenizer, p)
        if is_refusal(ans):
            continue
        corr = exact_match(ans, it.get("gold_answers", [it["gold"]]))
        rec = {**it, "prompt": p, "model_answer": ans}
        if not corr:
            confab_items.append(rec)
        else:
            correct_items.append(rec)
        if len(confab_items) >= len(correct_items):
            break

    n_pairs = min(len(correct_items), len(confab_items), target_needed)
    print(f"  -> Built {n_pairs} new pairs for '{rel}' (Correct: {len(correct_items)}, Confab: {len(confab_items)})", flush=True)

    for i in range(n_pairs):
        matched_pairs.append({
            "relation": rel,
            "correct_subject": correct_items[i]["subject"],
            "correct_prompt": correct_items[i]["prompt"],
            "correct_gold": correct_items[i]["gold"],
            "correct_model_answer": correct_items[i]["model_answer"],
            "correct_popularity": correct_items[i].get("popularity"),
            "confab_subject": confab_items[i]["subject"],
            "confab_prompt": confab_items[i]["prompt"],
            "confab_gold": confab_items[i]["gold"],
            "confab_model_answer": confab_items[i]["model_answer"],
            "confab_popularity": confab_items[i].get("popularity"),
        })

    # Save checkpoint after each relation
    save_json(matched_pairs, output_path)
    print(f"  [Checkpoint saved: {len(matched_pairs)} total pairs now on disk]", flush=True)

print(f"\n" + "=" * 60)
print(f"TOTAL MATCHED PAIRS IN EXPANDED POOL: {len(matched_pairs)}")
print("=" * 60, flush=True)
print(f"Saved {len(matched_pairs)} matched pairs to {output_path}", flush=True)
