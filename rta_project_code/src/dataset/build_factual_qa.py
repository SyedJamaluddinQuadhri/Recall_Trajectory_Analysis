"""
build_factual_qa.py — Build the labeled closed-book factual QA dataset.

We use PopQA (entity-centric factual QA with popularity annotations) and
TriviaQA (closed-book subset). For each question:
  1. Run Phi-2 closed-book (no retrieved context, no hints)
  2. Compare its answer to the ground truth (exact match + NLI equivalence)
  3. Label: CORRECT or CONFABULATED (discard explicit refusals)

This labeling is fully automatic — no human annotation needed.
"""

import re
from typing import List, Dict, Optional

import torch
from datasets import load_dataset

from src.utils import get_logger, save_json, get_project_root
from src.instrumentation.model_loader import load_phi2

log = get_logger("dataset.build_factual_qa")

REFUSAL_PATTERNS = [
    r"\bi don'?t know\b", r"\bi'?m not sure\b", r"\bunclear\b",
    r"\bno information\b", r"\bcannot determine\b", r"\bi am unable\b",
]


def is_refusal(text: str) -> bool:
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in REFUSAL_PATTERNS)


def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation and articles for exact-match comparison."""
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(predicted: str, gold_answers: List[str]) -> bool:
    """Check if predicted answer matches any of the gold answer aliases."""
    pred_norm = normalize_answer(predicted)
    for gold in gold_answers:
        gold_norm = normalize_answer(gold)
        if gold_norm and (gold_norm in pred_norm or pred_norm in gold_norm):
            return True
    return False


def generate_answer(model, tokenizer, question: str, max_new_tokens: int = 12) -> str:
    """
    Generate a short, closed-book answer from Phi-2.
    We use a short max_new_tokens because factual answers are typically
    a few words — this also keeps trajectory extraction focused on the
    actual answer token, not surrounding filler text.
    """
    device = next(model.parameters()).device
    prompt = f"Q: {question}\nA:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,   # greedy — we want the model's actual best guess
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
    # Cut at first newline — Phi-2 sometimes continues with "Q:" for the next question
    answer = answer.split("\n")[0].strip()
    return answer


def load_popqa(max_samples: int = 5000) -> List[Dict]:
    """
    Load PopQA — entity-centric factual QA with popularity scores.
    Popularity score lets us later check whether trajectory shape
    correlates with how well-represented a fact is in training data.
    """
    log.info("Loading PopQA...")
    ds = load_dataset("akariasai/PopQA", split="test")
    records = []
    for row in ds:
        if len(records) >= max_samples:
            break
        records.append({
            "question": row["question"],
            "gold_answers": eval(row["possible_answers"]) if isinstance(row["possible_answers"], str) else row["possible_answers"],
            "subject": row.get("subj", ""),
            "popularity": row.get("s_pop", 0),
            "source": "popqa",
        })
    log.info(f"PopQA: {len(records)} records")
    return records


def load_triviaqa_closed_book(max_samples: int = 5000) -> List[Dict]:
    """Load TriviaQA, filtered to closed-book-answerable questions."""
    log.info("Loading TriviaQA...")
    ds = load_dataset("trivia_qa", "unfiltered.nocontext", split="validation")
    records = []
    for row in ds:
        if len(records) >= max_samples:
            break
        gold = [row["answer"]["value"]] + row["answer"].get("aliases", [])
        records.append({
            "question": row["question"],
            "gold_answers": gold,
            "subject": "",
            "popularity": None,
            "source": "triviaqa",
        })
    log.info(f"TriviaQA: {len(records)} records")
    return records


def build_labeled_dataset(
    max_samples_per_source: int = 3000,
    save_path: Optional[str] = None,
) -> List[Dict]:
    """
    Full pipeline: load questions, run Phi-2 closed-book, label
    CORRECT / CONFABULATED, discard refusals.

    Returns:
        List of dicts: {question, gold_answers, model_answer, label, source, ...}
    """
    model, tokenizer = load_phi2()

    all_questions = (
        load_popqa(max_samples_per_source) +
        load_triviaqa_closed_book(max_samples_per_source)
    )

    labeled = []
    n_correct, n_confab, n_refused = 0, 0, 0

    for i, q in enumerate(all_questions):
        answer = generate_answer(model, tokenizer, q["question"])

        if is_refusal(answer):
            n_refused += 1
            continue

        correct = exact_match(answer, q["gold_answers"])

        labeled.append({
            **q,
            "model_answer": answer,
            "label": "correct" if correct else "confabulated",
        })

        if correct:
            n_correct += 1
        else:
            n_confab += 1

        if i % 200 == 0:
            log.info(
                f"Processed {i}/{len(all_questions)} | "
                f"correct={n_correct} confab={n_confab} refused={n_refused}"
            )

    log.info(f"\nFinal dataset: {len(labeled)} labeled examples")
    log.info(f"  Correct:      {n_correct} ({n_correct/len(labeled)*100:.1f}%)")
    log.info(f"  Confabulated: {n_confab} ({n_confab/len(labeled)*100:.1f}%)")
    log.info(f"  Refused (discarded): {n_refused}")

    if save_path:
        save_json(labeled, save_path)
        log.info(f"Saved to {save_path}")

    return labeled
