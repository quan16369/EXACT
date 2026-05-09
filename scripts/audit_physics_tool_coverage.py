#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from exact.data import normalize_records
from exact.tool_call_data import CapturedPythonTool, _answers_match
from exact.physics import solve_physics_question


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("Physics_Problems_Text_Only.csv"))
    parser.add_argument("--examples-per-bucket", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = [example for example in normalize_records([args.input]) if example.task == "physics"]
    counts: Counter[str] = Counter()
    buckets: dict[str, list[dict[str, str]]] = {}

    for example in examples:
        answer = str(example.answer).strip()
        tool = CapturedPythonTool(timeout=3.0)
        solved = solve_physics_question(example.question, tool=tool)
        solver_answer = str(solved.get("answer", "")).strip()

        if not answer:
            if solver_answer and solver_answer != "Uncertain" and tool.calls:
                bucket = "missing_gold_solver_solved"
            else:
                bucket = "missing_gold_unsolved"
        elif solver_answer == "Uncertain" or not tool.calls:
            bucket = "has_gold_solver_unsolved"
        elif _answers_match(solver_answer, answer):
            bucket = "solver_matches_gold"
        else:
            bucket = "solver_disagrees_gold"

        counts[bucket] += 1
        if len(buckets.setdefault(bucket, [])) < args.examples_per_bucket:
            buckets[bucket].append(
                {
                    "id": example.example_id,
                    "question": example.question,
                    "gold_answer": answer,
                    "unit": str(example.unit or ""),
                    "solver_answer": solver_answer,
                    "tool_code": tool.calls[0]["code"] if tool.calls else "",
                    "tool_output": tool.calls[0]["output"] if tool.calls else "",
                }
            )

    print(json.dumps({"counts": dict(counts), "examples": buckets}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

