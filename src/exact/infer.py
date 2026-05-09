from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .logic import solve_logic_question
from .physics import solve_physics_question


def _premises_from_payload(payload: dict[str, Any]) -> list[str]:
    value = (
        payload.get("premises-NL")
        or payload.get("premises_NL")
        or payload.get("premises_nl")
        or payload.get("premises")
        or []
    )
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def predict(payload: dict[str, Any]) -> dict[str, Any]:
    question = str(payload.get("question") or payload.get("query") or "")
    premises = _premises_from_payload(payload)
    if premises:
        return solve_logic_question(question, premises)
    return solve_physics_question(question)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None, help="JSON request path. Defaults to stdin.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input is None:
        payload = json.load(sys.stdin)
    else:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
    print(json.dumps(predict(payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

