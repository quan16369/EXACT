from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Any, Iterable

from .schema import ExactExample


def _read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"{path} must contain a JSON list or JSONL records")
        return data
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _get_first(record: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in record:
            return record[key]
    return default


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in _as_list(value)]


def _pick_by_index(values: list[Any], idx: int, default: str = "") -> str:
    if not values:
        return default
    if idx < len(values):
        return str(values[idx])
    return str(values[-1])


def _looks_like_logic(record: dict[str, Any]) -> bool:
    keys = set(record)
    return bool(
        {"premises-NL", "premises_NL", "premises_nl", "premises-FOL", "questions"}
        & keys
    )


def flatten_logic_record(record: dict[str, Any], record_index: int) -> list[ExactExample]:
    premises_nl = _string_list(
        _get_first(record, "premises-NL", "premises_NL", "premises_nl", "premises")
    )
    premises_fol = _string_list(
        _get_first(record, "premises-FOL", "premises_FOL", "premises_fol")
    )
    questions = _string_list(_get_first(record, "questions", "question"))
    answers = _as_list(_get_first(record, "answers", "answer"))
    explanations = _as_list(_get_first(record, "explanation", "explanations"))
    base_id = str(_get_first(record, "id", "record_id", default=f"logic_{record_index:05d}"))

    examples: list[ExactExample] = []
    for question_index, question in enumerate(questions):
        example_id = f"{base_id}_q{question_index}" if len(questions) > 1 else base_id
        examples.append(
            ExactExample(
                example_id=example_id,
                task="logic",
                question=question,
                answer=_pick_by_index(answers, question_index, default=""),
                explanation=_pick_by_index(explanations, question_index, default=""),
                premises_nl=premises_nl,
                premises_fol=premises_fol,
                metadata={"source_record_id": base_id, "question_index": question_index},
            )
        )
    return examples


def normalize_physics_record(record: dict[str, Any], record_index: int) -> ExactExample:
    cot_raw = _get_first(record, "cot", "CoT", "solution", "steps")
    if isinstance(cot_raw, str):
        cot = [line.strip() for line in cot_raw.splitlines() if line.strip()]
    else:
        cot = _string_list(cot_raw)
    return ExactExample(
        example_id=str(_get_first(record, "id", "problem_id", default=f"physics_{record_index:05d}")),
        task="physics",
        question=str(_get_first(record, "question", "prompt", default="")),
        answer=str(_get_first(record, "answer", "final_answer", default="")),
        explanation=str(_get_first(record, "explanation", default="")),
        cot=cot,
        unit=(str(record["unit"]) if "unit" in record and record["unit"] is not None else None),
        metadata={"source_record_id": _get_first(record, "id", "problem_id", default=record_index)},
    )


def normalize_records(paths: Iterable[Path]) -> list[ExactExample]:
    examples: list[ExactExample] = []
    for path in paths:
        records = _read_json_or_jsonl(path)
        for idx, record in enumerate(records):
            if _looks_like_logic(record):
                examples.extend(flatten_logic_record(record, idx))
            else:
                examples.append(normalize_physics_record(record, idx))
    return examples


def write_normalized_jsonl(examples: Iterable[ExactExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for example in examples:
            row = {
                "id": example.example_id,
                "task": example.task,
                "question": example.question,
                "answer": example.answer,
                "explanation": example.explanation,
                "premises_nl": example.premises_nl,
                "premises_fol": example.premises_fol,
                "cot": example.cot,
                "unit": example.unit,
                "metadata": example.metadata,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
