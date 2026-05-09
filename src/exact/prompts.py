from __future__ import annotations

import json

from .schema import ExactExample

LOGIC_SYSTEM_PROMPT = """You answer educational regulation questions.
Return valid JSON with at least answer and explanation.
Use only the supplied natural-language premises. Cite the relevant premises in
the premises field when possible. If the answer is not entailed, say Uncertain."""

PHYSICS_SYSTEM_PROMPT = """You solve electricity and electrostatics problems.
Return valid JSON with answer and explanation. Include unit, cot, and premises
when useful. Convert units before calculating and keep the final answer concise."""


def build_user_prompt(example: ExactExample) -> str:
    if example.task == "logic":
        premise_lines = "\n".join(
            f"{idx + 1}. {premise}" for idx, premise in enumerate(example.premises_nl)
        )
        return f"Premises:\n{premise_lines}\n\nQuestion:\n{example.question}"
    return f"Question:\n{example.question}"


def build_completion(example: ExactExample) -> str:
    return json.dumps(example.target_payload(), ensure_ascii=False, separators=(",", ":"))


def system_prompt_for_task(task: str) -> str:
    return LOGIC_SYSTEM_PROMPT if task == "logic" else PHYSICS_SYSTEM_PROMPT

