from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

TaskType = Literal["logic", "physics"]


@dataclass(slots=True)
class ExactExample:
    example_id: str
    task: TaskType
    question: str
    answer: str
    explanation: str
    premises_nl: list[str] = field(default_factory=list)
    premises_fol: list[str] = field(default_factory=list)
    cot: list[str] = field(default_factory=list)
    unit: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def target_payload(self) -> dict[str, Any]:
        explanation = self.explanation
        if not explanation and self.task == "physics" and self.cot:
            explanation = " ".join(self.cot)
        payload: dict[str, Any] = {
            "answer": self.answer,
            "explanation": explanation,
        }
        if "tool_calls" in self.metadata:
            payload["tool_calls"] = self.metadata["tool_calls"]
        if self.task == "logic":
            if self.premises_fol:
                payload["fol"] = "\n".join(self.premises_fol)
            if self.premises_nl:
                payload["premises"] = self.premises_nl
        if self.task == "physics":
            if self.unit:
                payload["unit"] = self.unit
            if self.cot:
                payload["cot"] = self.cot
        return payload
