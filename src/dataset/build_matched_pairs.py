"""
build_matched_pairs.py — Build CounterFact-style matched (correct, confabulated)
pairs for the causal activation-patching experiment in Stage 4.

WHY THIS IS SEPARATE FROM THE MAIN DATASET:
For the causal patching experiment, we need STRUCTURALLY SIMILAR pairs:
the same relation type (e.g. "X was born in Y"), applied to two different
subjects — one where Phi-2 answers correctly, one where it confabulates.
This controls for confounds like question length, relation type, and
phrasing, isolating the trajectory-shape difference to the actual
correct-vs-confabulated distinction.

Approach: use a fixed set of relation templates (following the CounterFact
methodology from Meng et al. 2022 ROME) applied to many different subject
entities, then split subjects into "model knows this" vs "model doesn't"
based on Phi-2's own closed-book accuracy.
"""

from typing import List, Dict, Optional
import torch

from src.utils import get_logger, save_json
from src.dataset.build_factual_qa import generate_answer, exact_match, is_refusal

log = get_logger("dataset.matched_pairs")


# Relation templates with 2-shot direct exemplars so Phi-2 answers directly
# with the factual token rather than echoing the subject in full sentences.
RELATION_TEMPLATES = [
    {
        "relation": "birthplace",
        "exemplars": "Q: What city was Stephen Hawking born in?\nA: Oxford\n\nQ: What city was Niels Bohr born in?\nA: Copenhagen\n\n",
        "format": "Q: What city was {subject} born in?\nA:",
    },
    {
        "relation": "occupation",
        "exemplars": "Q: What was Leonardo da Vinci's primary occupation?\nA: Artist\n\nQ: What was Albert Einstein's primary occupation?\nA: Physicist\n\n",
        "format": "Q: What was {subject}'s primary occupation?\nA:",
    },
    {
        "relation": "nationality",
        "exemplars": "Q: What nationality was Winston Churchill?\nA: British\n\nQ: What nationality was Napoleon Bonaparte?\nA: French\n\n",
        "format": "Q: What nationality was {subject}?\nA:",
    },
    {
        "relation": "death_year",
        "exemplars": "Q: In what year did George Washington die?\nA: 1799\n\nQ: In what year did Abraham Lincoln die?\nA: 1865\n\n",
        "format": "Q: In what year did {subject} die?\nA:",
    },
    {
        "relation": "founded_by",
        "exemplars": "Q: Who founded Microsoft?\nA: Bill Gates\n\nQ: Who founded Apple?\nA: Steve Jobs\n\n",
        "format": "Q: Who founded {subject}?\nA:",
    },
    {
        "relation": "genre",
        "exemplars": "Q: What genre is Queen?\nA: Rock\n\nQ: What genre is Miles Davis?\nA: Jazz\n\n",
        "format": "Q: What genre is {subject}?\nA:",
    },
    {
        "relation": "father",
        "exemplars": "Q: Who is the father of Alexander the Great?\nA: Philip II of Macedon\n\nQ: Who is the father of Charlemagne?\nA: Pepin the Short\n\n",
        "format": "Q: Who is the father of {subject}?\nA:",
    },
]


def generate_direct_answer(model, tokenizer, prompt: str, max_new_tokens: int = 8) -> str:
    """Generate direct factual continuation following a prompt ending in 'A:'."""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return answer.split("\n")[0].strip()


def build_matched_pairs(
    subjects_with_facts: List[Dict],
    max_pairs: int = 200,
    save_path: Optional[str] = None,
    model=None,
    tokenizer=None,
) -> List[Dict]:
    """
    Build matched (correct, confabulated) pairs using the same relation
    template applied to different subjects.

    Args:
        subjects_with_facts: list of dicts like:
            {"subject": "Marie Curie", "relation": "birthplace", "gold": "Warsaw"}
        max_pairs: maximum number of matched pairs to build
        save_path: where to save the resulting pairs
        model: pre-loaded Phi-2 model (avoids double-loading VRAM in Stage 4).
               If None, loads internally (legacy behaviour).
        tokenizer: matching tokenizer. Required when model is provided.

    Returns:
        List of dicts: {
            relation, correct_subject, correct_prompt, correct_answer,
            confab_subject, confab_prompt, confab_model_answer, confab_gold_answer
        }
    """
    _owns_model = False
    if model is None:
        from src.instrumentation.model_loader import load_phi2
        model, tokenizer = load_phi2()
        _owns_model = True

    # Group by relation so we only pair within the same template
    by_relation = {}
    for item in subjects_with_facts:
        by_relation.setdefault(item["relation"], []).append(item)

    matched_pairs = []

    for relation, items in by_relation.items():
        rel_entry = next(
            (t for t in RELATION_TEMPLATES if t["relation"] == relation),
            None
        )
        if rel_entry is None:
            log.warning(f"No template for relation '{relation}', skipping")
            continue

        correct_items = []
        confab_items = []

        for item in items:
            prompt = rel_entry["exemplars"] + rel_entry["format"].format(subject=item["subject"])
            model_answer = generate_direct_answer(model, tokenizer, prompt)

            if is_refusal(model_answer):
                continue

            is_correct = exact_match(model_answer, [item["gold"]])
            record = {**item, "prompt": prompt, "model_answer": model_answer}

            if is_correct:
                correct_items.append(record)
            else:
                confab_items.append(record)

        # Pair each confabulated item with a correct item of the SAME relation
        n_pairs_this_relation = min(len(correct_items), len(confab_items))
        for i in range(n_pairs_this_relation):
            matched_pairs.append({
                "relation": relation,
                "correct_subject": correct_items[i]["subject"],
                "correct_prompt": correct_items[i]["prompt"],
                "correct_gold": correct_items[i]["gold"],
                "correct_model_answer": correct_items[i]["model_answer"],
                "confab_subject": confab_items[i]["subject"],
                "confab_prompt": confab_items[i]["prompt"],
                "confab_gold": confab_items[i]["gold"],
                "confab_model_answer": confab_items[i]["model_answer"],
            })

        if len(matched_pairs) >= max_pairs:
            break

    matched_pairs = matched_pairs[:max_pairs]
    log.info(f"Built {len(matched_pairs)} matched (correct, confabulated) pairs")

    if save_path:
        save_json(matched_pairs, save_path)
        log.info(f"Saved to {save_path}")

    return matched_pairs


def build_default_subject_list() -> List[Dict]:
    """
    Curated seed list of (subject, relation, gold) triples across common
    and rare entities. Expanded to ~70 subjects across 5 relations to give
    a realistic chance of 20+ matched pairs from the pilot run.

    Phi-2's closed-book accuracy on well-known people is high (correct
    bucket) while accuracy on less-famous figures is low (confabulated
    bucket), so mixing famous + obscure subjects gives natural pairing.
    """
    return [
        # ── birthplace (famous → correct, obscure → confabulated) ──────────
        {"subject": "Marie Curie",              "relation": "birthplace", "gold": "Warsaw"},
        {"subject": "Albert Einstein",          "relation": "birthplace", "gold": "Ulm"},
        {"subject": "Alan Turing",              "relation": "birthplace", "gold": "London"},
        {"subject": "Isaac Newton",             "relation": "birthplace", "gold": "Woolsthorpe"},
        {"subject": "Charles Darwin",           "relation": "birthplace", "gold": "Shrewsbury"},
        {"subject": "Galileo Galilei",          "relation": "birthplace", "gold": "Pisa"},
        {"subject": "Nikola Tesla",             "relation": "birthplace", "gold": "Smiljan"},
        {"subject": "Stephen Hawking",          "relation": "birthplace", "gold": "Oxford"},
        {"subject": "Richard Feynman",          "relation": "birthplace", "gold": "New York"},
        {"subject": "Max Planck",               "relation": "birthplace", "gold": "Kiel"},
        {"subject": "Niels Bohr",               "relation": "birthplace", "gold": "Copenhagen"},
        {"subject": "Erwin Schrödinger",        "relation": "birthplace", "gold": "Vienna"},
        {"subject": "Srinivasa Ramanujan",      "relation": "birthplace", "gold": "Erode"},
        {"subject": "Emmy Noether",             "relation": "birthplace", "gold": "Erlangen"},
        {"subject": "Rosalind Franklin",        "relation": "birthplace", "gold": "London"},
        {"subject": "Hedy Lamarr",              "relation": "birthplace", "gold": "Vienna"},
        {"subject": "Chien-Shiung Wu",          "relation": "birthplace", "gold": "Liuhe"},
        {"subject": "Cecilia Payne-Gaposchkin", "relation": "birthplace", "gold": "Wendover"},
        {"subject": "Lise Meitner",             "relation": "birthplace", "gold": "Vienna"},
        {"subject": "Dorothy Hodgkin",          "relation": "birthplace", "gold": "Cairo"},
        {"subject": "Vera Rubin",               "relation": "birthplace", "gold": "Philadelphia"},
        {"subject": "Subrahmanyan Chandrasekhar","relation": "birthplace", "gold": "Lahore"},
        {"subject": "Paul Dirac",               "relation": "birthplace", "gold": "Bristol"},
        {"subject": "Werner Heisenberg",        "relation": "birthplace", "gold": "Würzburg"},
        {"subject": "Enrico Fermi",             "relation": "birthplace", "gold": "Rome"},

        # ── occupation ─────────────────────────────────────────────────────
        {"subject": "Ada Lovelace",             "relation": "occupation", "gold": "mathematician"},
        {"subject": "Nikola Tesla",             "relation": "occupation", "gold": "engineer"},
        {"subject": "Hedy Lamarr",              "relation": "occupation", "gold": "actress"},
        {"subject": "Katherine Johnson",        "relation": "occupation", "gold": "mathematician"},
        {"subject": "Marie Curie",              "relation": "occupation", "gold": "physicist"},
        {"subject": "Albert Einstein",          "relation": "occupation", "gold": "physicist"},
        {"subject": "Leonardo da Vinci",        "relation": "occupation", "gold": "artist"},
        {"subject": "Sigmund Freud",            "relation": "occupation", "gold": "neurologist"},
        {"subject": "Florence Nightingale",     "relation": "occupation", "gold": "nurse"},
        {"subject": "Charles Dickens",          "relation": "occupation", "gold": "novelist"},
        {"subject": "Galileo Galilei",          "relation": "occupation", "gold": "astronomer"},
        {"subject": "Gregor Mendel",            "relation": "occupation", "gold": "friar"},
        {"subject": "Emmy Noether",             "relation": "occupation", "gold": "mathematician"},
        {"subject": "Subrahmanyan Chandrasekhar","relation": "occupation", "gold": "astrophysicist"},
        {"subject": "Sofia Kovalevskaya",       "relation": "occupation", "gold": "mathematician"},

        # ── nationality ─────────────────────────────────────────────────────
        {"subject": "Isaac Newton",             "relation": "nationality", "gold": "English"},
        {"subject": "Marie Curie",              "relation": "nationality", "gold": "Polish"},
        {"subject": "Albert Einstein",          "relation": "nationality", "gold": "German"},
        {"subject": "Nikola Tesla",             "relation": "nationality", "gold": "Serbian"},
        {"subject": "Galileo Galilei",          "relation": "nationality", "gold": "Italian"},
        {"subject": "Wangari Maathai",          "relation": "nationality", "gold": "Kenyan"},
        {"subject": "Chien-Shiung Wu",          "relation": "nationality", "gold": "Chinese"},
        {"subject": "Lise Meitner",             "relation": "nationality", "gold": "Austrian"},
        {"subject": "Enrico Fermi",             "relation": "nationality", "gold": "Italian"},
        {"subject": "Paul Dirac",               "relation": "nationality", "gold": "British"},
        {"subject": "Werner Heisenberg",        "relation": "nationality", "gold": "German"},
        {"subject": "Niels Bohr",               "relation": "nationality", "gold": "Danish"},
        {"subject": "Max Planck",               "relation": "nationality", "gold": "German"},
        {"subject": "Cecilia Payne-Gaposchkin", "relation": "nationality", "gold": "British"},
        {"subject": "Sofia Kovalevskaya",       "relation": "nationality", "gold": "Russian"},

        # ── death_year ──────────────────────────────────────────────────────
        {"subject": "Albert Einstein",          "relation": "death_year", "gold": "1955"},
        {"subject": "Marie Curie",              "relation": "death_year", "gold": "1934"},
        {"subject": "Srinivasa Ramanujan",      "relation": "death_year", "gold": "1920"},
        {"subject": "Emmy Noether",             "relation": "death_year", "gold": "1935"},
        {"subject": "Alan Turing",              "relation": "death_year", "gold": "1954"},
        {"subject": "Isaac Newton",             "relation": "death_year", "gold": "1727"},
        {"subject": "Charles Darwin",           "relation": "death_year", "gold": "1882"},
        {"subject": "Nikola Tesla",             "relation": "death_year", "gold": "1943"},
        {"subject": "Max Planck",               "relation": "death_year", "gold": "1947"},
        {"subject": "Niels Bohr",               "relation": "death_year", "gold": "1962"},
        {"subject": "Enrico Fermi",             "relation": "death_year", "gold": "1954"},
        {"subject": "Paul Dirac",               "relation": "death_year", "gold": "1984"},
        {"subject": "Lise Meitner",             "relation": "death_year", "gold": "1968"},
        {"subject": "Cecilia Payne-Gaposchkin", "relation": "death_year", "gold": "1979"},
        {"subject": "Dorothy Hodgkin",          "relation": "death_year", "gold": "1994"},

        # ── founded_by ──────────────────────────────────────────────────────
        {"subject": "Microsoft",   "relation": "founded_by", "gold": "Bill Gates"},
        {"subject": "Apple",       "relation": "founded_by", "gold": "Steve Jobs"},
        {"subject": "Amazon",      "relation": "founded_by", "gold": "Jeff Bezos"},
        {"subject": "Google",      "relation": "founded_by", "gold": "Larry Page"},
        {"subject": "Facebook",    "relation": "founded_by", "gold": "Mark Zuckerberg"},
        {"subject": "Tesla",       "relation": "founded_by", "gold": "Elon Musk"},
        {"subject": "Intel",       "relation": "founded_by", "gold": "Gordon Moore"},
        {"subject": "IBM",         "relation": "founded_by", "gold": "Charles Flint"},
        {"subject": "Infosys",     "relation": "founded_by", "gold": "Narayana Murthy"},
        {"subject": "Tata Group",  "relation": "founded_by", "gold": "Jamsetji Tata"},
        {"subject": "Sony",        "relation": "founded_by", "gold": "Masaru Ibuka"},
        {"subject": "Samsung",     "relation": "founded_by", "gold": "Lee Byung-chul"},
        {"subject": "Toyota",      "relation": "founded_by", "gold": "Kiichiro Toyoda"},
        {"subject": "Volkswagen",  "relation": "founded_by", "gold": "Ferdinand Porsche"},
        {"subject": "Airbus",      "relation": "founded_by", "gold": "Henri Ziegler"},
    ]
