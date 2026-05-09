from __future__ import annotations

import json
import re
from typing import Any

from .physics import solve_physics_question
from .schema import ExactExample
from .tools import PythonTool


class CapturedPythonTool(PythonTool):
    def __init__(self, timeout: float = 3.0) -> None:
        super().__init__(timeout=timeout)
        self.calls: list[dict[str, str]] = []

    def execute(self, code: str):
        result = super().execute(code)
        self.calls.append(
            {
                "tool": "python",
                "code": result.code,
                "output": result.stdout.strip() if result.ok else result.output,
            }
        )
        return result


TOOL_SYSTEM_PROMPT = """You solve physics questions by using the Python tool.
First return exactly one JSON tool call:
{"tool":"python","code":"..."}
After the tool output is provided, return the final JSON answer with answer,
unit, explanation, cot, premises, and confidence."""


def build_tool_call_prompt(example: ExactExample) -> str:
    return (
        "<|im_start|>system\n"
        f"{TOOL_SYSTEM_PROMPT}<|im_end|>\n"
        "<|im_start|>user\n"
        f"Question:\n{example.question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _json_compact(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


NUMBER_PATTERN = re.compile(
    r"[-+]?\d+(?:\.\d+)?(?:\s*(?:x|\*|×)\s*10\s*\^?\s*[-+]?\d+|[eE][-+]?\d+)?"
)


def _parse_number(raw: str) -> float:
    compact = raw.replace(" ", "").replace("×", "x")
    match = re.fullmatch(r"([-+]?\d+(?:\.\d+)?)(?:x|\*)10\^?([-+]?\d+)", compact)
    if match:
        return float(match.group(1)) * (10 ** int(match.group(2)))
    return float(compact)


def _extract_numbers(text: str) -> list[float]:
    return [_parse_number(match.group(0)) for match in NUMBER_PATTERN.finditer(text)]


def _answers_match(predicted: Any, expected: str) -> bool:
    pred_nums = _extract_numbers(str(predicted))
    exp_nums = _extract_numbers(str(expected))
    if pred_nums and exp_nums:
        if len(pred_nums) != len(exp_nums):
            return False
        for pred_num, exp_num in zip(pred_nums, exp_nums):
            scale = max(1.0, abs(exp_num))
            if abs(pred_num - exp_num) / scale > 1e-3:
                return False
        return True
    return str(predicted).strip() == str(expected).strip()


def _gold_answer_tool_call(example: ExactExample) -> dict[str, str] | None:
    if not str(example.answer).strip():
        return None
    code = (
        "# Fallback calculation target from the worked solution.\n"
        f"answer = {example.answer!r}\n"
        "print(answer)"
    )
    return {
        "tool": "python",
        "code": code,
        "output": str(example.answer).strip(),
    }


def _final_explanation(example: ExactExample, solved: dict[str, Any]) -> str:
    if example.explanation:
        return example.explanation
    if example.cot:
        return " ".join(example.cot)
    return str(solved.get("explanation", ""))


def make_tool_call_completion(
    example: ExactExample,
    *,
    fallback_to_gold_answer: bool = False,
) -> str | None:
    tool = CapturedPythonTool(timeout=3.0)
    solved = solve_physics_question(example.question, tool=tool)
    use_solver_call = bool(tool.calls) and solved.get("answer") != "Uncertain"
    if use_solver_call and str(example.answer).strip():
        use_solver_call = _answers_match(solved.get("answer"), example.answer)
    if use_solver_call:
        first_call = tool.calls[0]
    elif fallback_to_gold_answer:
        first_call = _gold_answer_tool_call(example)
        if first_call is None:
            return None
    else:
        return None
    tool_call = {"tool": "python", "code": first_call["code"]}
    final_payload = {
        "answer": str(example.answer).strip() or solved.get("answer", ""),
        "unit": str(example.unit).strip() if example.unit is not None else solved.get("unit", ""),
        "explanation": _final_explanation(example, solved),
        "cot": example.cot or solved.get("cot", []),
        "premises": solved.get("premises", []),
        "confidence": solved.get("confidence", 0.8),
    }
    return (
        _json_compact(tool_call)
        + "<|im_end|>\n"
        + "<|im_start|>tool name=python\n"
        + first_call["output"]
        + "<|im_end|>\n"
        + "<|im_start|>assistant\n"
        + _json_compact(final_payload)
        + "<|im_end|>"
    )
