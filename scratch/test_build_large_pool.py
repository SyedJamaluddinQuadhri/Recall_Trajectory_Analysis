import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import random

from src.utils import get_logger, get_project_root, save_json
from src.instrumentation.model_loader import load_phi2
from src.dataset.build_matched_pairs import generate_direct_answer, RELATION_TEMPLATES, build_default_subject_list
from src.dataset.build_factual_qa import exact_match, is_refusal

root = get_project_root()
labeled_path = str(root / "data" / "processed" / "labeled_qa.json")

with open(labeled_path, "r", encoding="utf-8") as f:
    labeled = json.load(f)

# Extract subjects from PopQA in labeled_qa
popqa = [d for d in labeled if d.get("source") == "popqa" and d.get("subject")]

extracted_subjects = []

# Birthplace
for d in popqa:
    if "In what city was" in d["question"]:
        extracted_subjects.append({
            "subject": d["subject"],
            "relation": "birthplace",
            "gold": d["gold_answers"][0] if d["gold_answers"] else "",
            "gold_answers": d["gold_answers"],
            "popularity": d.get("popularity", 0),
        })
    elif "What is" in d["question"] and "occupation" in d["question"]:
        extracted_subjects.append({
            "subject": d["subject"],
            "relation": "occupation",
            "gold": d["gold_answers"][0] if d["gold_answers"] else "",
            "gold_answers": d["gold_answers"],
            "popularity": d.get("popularity", 0),
        })

print(f"Extracted {len(extracted_subjects)} subjects from labeled_qa (birthplace + occupation).")

# Also add the curated seeds
seeds = build_default_subject_list()
for s in seeds:
    extracted_subjects.append({
        **s,
        "gold_answers": [s["gold"]],
        "popularity": 1000000,  # famous entities
    })

print(f"Total subject pool: {len(extracted_subjects)} subjects.")

# Let's shuffle with a fixed seed so it is deterministic
random.Random(42).shuffle(extracted_subjects)

# Now let's test building pairs with model
model, tokenizer = load_phi2()

by_relation = {}
for it in extracted_subjects:
    by_relation.setdefault(it["relation"], []).append(it)

matched_pairs = []
target_pairs = 200

for rel_entry in RELATION_TEMPLATES:
    relation = rel_entry["relation"]
    items = by_relation.get(relation, [])
    if not items:
        continue

    print(f"Processing relation '{relation}' ({len(items)} items)...")
    correct_items = []
    confab_items = []

    # Sort items by popularity descending to find correct items from top, confab items from bottom/middle
    items_by_pop = sorted(items, key=lambda x: x.get("popularity", 0), reverse=True)
    high_pop = items_by_pop[:len(items_by_pop)//2]
    low_pop = items_by_pop[len(items_by_pop)//2:]

    # Test high pop for correct
    for item in high_pop:
        prompt = rel_entry["exemplars"] + rel_entry["format"].format(subject=item["subject"])
        ans = generate_direct_answer(model, tokenizer, prompt)
        if is_refusal(ans):
            continue
        is_corr = exact_match(ans, item.get("gold_answers", [item["gold"]]))
        rec = {**item, "prompt": prompt, "model_answer": ans}
        if is_corr:
            correct_items.append(rec)
        else:
            confab_items.append(rec)
        if len(correct_items) >= 50:
            break

    # Test low pop for confabulated
    for item in low_pop:
        prompt = rel_entry["exemplars"] + rel_entry["format"].format(subject=item["subject"])
        ans = generate_direct_answer(model, tokenizer, prompt)
        if is_refusal(ans):
            continue
        is_corr = exact_match(ans, item.get("gold_answers", [item["gold"]]))
        rec = {**item, "prompt": prompt, "model_answer": ans}
        if not is_corr:
            confab_items.append(rec)
        else:
            correct_items.append(rec)
        if len(confab_items) >= len(correct_items) + 10:
            break

    n_pairs_rel = min(len(correct_items), len(confab_items))
    print(f"  -> Found {len(correct_items)} correct, {len(confab_items)} confab -> {n_pairs_rel} pairs")

    for i in range(n_pairs_rel):
        matched_pairs.append({
            "relation": relation,
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

print(f"\nTotal Matched Pairs Built: {len(matched_pairs)}")
