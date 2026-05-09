from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "based",
    "be",
    "can",
    "does",
    "for",
    "from",
    "if",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "student",
    "the",
    "then",
    "to",
    "what",
}


@dataclass(slots=True)
class Rule:
    antecedents: set[str]
    consequent: str
    premise_index: int
    raw: str


def normalize_phrase(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    words = [_normalize_word(word) for word in text.split() if word not in STOPWORDS]
    return " ".join(words)


def _normalize_word(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("ed"):
        return word[:-2]
    if len(word) > 4 and word.endswith("es"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _split_conjunction(text: str) -> list[str]:
    return [part.strip(" .") for part in re.split(r"\band\b|,", text) if part.strip(" .")]


def parse_rule(premise: str, idx: int) -> Rule | None:
    match = re.search(r"\bif\b(?P<lhs>.+?)(?:,|\bthen\b)(?P<rhs>.+)$", premise, re.I)
    if not match:
        return None
    lhs_parts = _split_conjunction(match.group("lhs"))
    antecedents = {normalize_phrase(part) for part in lhs_parts if normalize_phrase(part)}
    consequent = normalize_phrase(match.group("rhs"))
    if not antecedents or not consequent:
        return None
    return Rule(antecedents=antecedents, consequent=consequent, premise_index=idx, raw=premise)


def _fact_from_premise(premise: str) -> str | None:
    if re.search(r"\bif\b", premise, re.I):
        return None
    normalized = normalize_phrase(premise)
    return normalized or None


def derive_facts(premises: list[str]) -> tuple[set[str], list[int]]:
    facts: set[str] = set()
    used_indices: list[int] = []
    rules: list[Rule] = []
    for idx, premise in enumerate(premises):
        rule = parse_rule(premise, idx)
        if rule is not None:
            rules.append(rule)
            continue
        fact = _fact_from_premise(premise)
        if fact:
            facts.add(fact)
            used_indices.append(idx)

    changed = True
    while changed:
        changed = False
        for rule in rules:
            if rule.consequent in facts:
                continue
            if all(any(ant in fact or fact in ant for fact in facts) for ant in rule.antecedents):
                facts.add(rule.consequent)
                used_indices.append(rule.premise_index)
                changed = True
    return facts, sorted(set(used_indices))


def extract_options(question: str) -> dict[str, str]:
    options: dict[str, str] = {}
    pattern = re.compile(r"(?:^|\n)\s*([A-D])[\).]\s*(.+?)(?=\n\s*[A-D][\).]|\Z)", re.S)
    for match in pattern.finditer(question):
        options[match.group(1)] = " ".join(match.group(2).split())
    return options


def choose_option(question: str, facts: set[str]) -> str | None:
    options = extract_options(question)
    if not options:
        return None
    best_label: str | None = None
    best_score = -1
    fact_text = " ".join(facts)
    for label, option in options.items():
        option_terms = set(normalize_phrase(option).split())
        score = sum(1 for term in option_terms if term in fact_text)
        if score > best_score:
            best_label = label
            best_score = score
    return best_label


def answer_yes_no(question: str, facts: set[str]) -> str | None:
    q_norm = normalize_phrase(question)
    fact_text = " ".join(facts)
    if "cannot" in fact_text or "not pass" in fact_text:
        if any(term in q_norm for term in ("receive", "get", "pass", "eligible")):
            return "No"
    q_terms = [term for term in q_norm.split() if len(term) > 3]
    overlap = sum(1 for term in q_terms if term in fact_text)
    if overlap >= max(2, len(q_terms) // 3):
        return "Yes"
    if question.strip().lower().startswith(("does ", "can ", "is ", "are ")):
        return "Uncertain"
    return None


def select_relevant_premises(question: str, premises: list[str], facts: set[str], used: list[int]) -> list[str]:
    q_terms = set(normalize_phrase(question).split())
    selected: list[int] = []
    for idx, premise in enumerate(premises):
        p_terms = set(normalize_phrase(premise).split())
        if idx in used or len(q_terms & p_terms) >= 2:
            selected.append(idx)
    if not selected and premises:
        selected.append(0)
    del facts
    return [premises[idx] for idx in sorted(set(selected))]


def solve_logic_question(question: str, premises_nl: list[str]) -> dict[str, Any]:
    facts, used = derive_facts(premises_nl)
    selected = select_relevant_premises(question, premises_nl, facts, used)
    option = choose_option(question, facts)
    if option is not None:
        answer = option
        confidence = 0.68
    else:
        answer = answer_yes_no(question, facts) or "Uncertain"
        confidence = 0.72 if answer != "Uncertain" else 0.35
    derived_preview = "; ".join(sorted(facts)[:5])
    explanation = (
        "The rule engine derives facts from the supplied premises and compares them "
        f"with the question. Derived facts include: {derived_preview}."
    )
    return {
        "answer": answer,
        "explanation": explanation,
        "premises": selected,
        "fol": "",
        "confidence": confidence,
    }
