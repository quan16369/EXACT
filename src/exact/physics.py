from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from .tools import PythonTool, PythonToolError

PREFIX_MULTIPLIERS = {
    "p": 1e-12,
    "n": 1e-9,
    "u": 1e-6,
    "\u00b5": 1e-6,
    "\u03bc": 1e-6,
    "m": 1e-3,
    "c": 1e-2,
    "k": 1e3,
    "M": 1e6,
}

BASE_UNITS = {"V", "A", "F", "W", "J", "C", "N", "m", "ohm"}


@dataclass(slots=True)
class Measurement:
    value: float
    base_unit: str
    raw_unit: str
    symbol: str | None = None


@dataclass(slots=True)
class PhysicsAnswer:
    answer: str
    unit: str
    explanation: str
    cot: list[str]
    premises: list[str]
    confidence: float
    tool_calls: list[dict[str, str]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "unit": self.unit,
            "explanation": self.explanation,
            "cot": self.cot,
            "premises": self.premises,
            "confidence": self.confidence,
            "tool_calls": self.tool_calls,
        }


NUMBER_PATTERN = (
    r"[-+]?\d+(?:\.\d+)?(?:\s*(?:x|\*|\u00d7)\s*10\s*\^?\s*[-+]?\d+|e[-+]?\d+)?"
)
ASSIGNMENT_RE = re.compile(
    rf"\b(?P<symbol>[A-Za-z])\s*=\s*(?P<number>{NUMBER_PATTERN})\s*(?P<unit>[A-Za-z\u00b5\u03bc\u03a9]+)"
)
MEASUREMENT_RE = re.compile(
    rf"(?P<number>{NUMBER_PATTERN})\s*(?P<unit>[A-Za-z\u00b5\u03bc\u03a9]+)"
)


def parse_number(raw: str) -> float:
    compact = raw.replace(" ", "").replace("\u00d7", "x")
    match = re.fullmatch(r"([-+]?\d+(?:\.\d+)?)(?:x|\*)10\^?([-+]?\d+)", compact)
    if match:
        return float(match.group(1)) * (10 ** int(match.group(2)))
    return float(compact)


def normalize_unit(raw_unit: str) -> tuple[str, float] | None:
    unit = raw_unit.strip()
    lower = unit.lower()
    if lower in {"ohm", "ohms"} or unit == "\u03a9":
        return "ohm", 1.0
    if unit == "m":
        return "m", 1.0
    if unit in BASE_UNITS:
        return unit, 1.0
    for prefix, multiplier in sorted(PREFIX_MULTIPLIERS.items(), key=lambda item: -len(item[0])):
        if unit.startswith(prefix):
            rest = unit[len(prefix) :]
            if rest in BASE_UNITS:
                return rest, multiplier
            if rest == "\u03a9":
                return "ohm", multiplier
    return None


def extract_measurements(question: str) -> list[Measurement]:
    measurements: list[Measurement] = []
    occupied_spans: set[tuple[int, int]] = set()
    for match in ASSIGNMENT_RE.finditer(question):
        normalized = normalize_unit(match.group("unit"))
        if normalized is None:
            continue
        base_unit, multiplier = normalized
        measurements.append(
            Measurement(
                value=parse_number(match.group("number")) * multiplier,
                base_unit=base_unit,
                raw_unit=match.group("unit"),
                symbol=match.group("symbol"),
            )
        )
        occupied_spans.add(match.span())

    for match in MEASUREMENT_RE.finditer(question):
        if any(start <= match.start() < end for start, end in occupied_spans):
            continue
        normalized = normalize_unit(match.group("unit"))
        if normalized is None:
            continue
        base_unit, multiplier = normalized
        measurements.append(
            Measurement(
                value=parse_number(match.group("number")) * multiplier,
                base_unit=base_unit,
                raw_unit=match.group("unit"),
            )
        )
    return measurements


def _by_symbol(measurements: list[Measurement], symbol: str, base_unit: str) -> float | None:
    for measurement in measurements:
        if measurement.symbol == symbol and measurement.base_unit == base_unit:
            return measurement.value
    return None


def _first_unit(measurements: list[Measurement], base_unit: str) -> float | None:
    for measurement in measurements:
        if measurement.base_unit == base_unit:
            return measurement.value
    return None


def _quantity(
    measurements: list[Measurement],
    symbols: tuple[str, ...],
    base_unit: str,
) -> float | None:
    for symbol in symbols:
        value = _by_symbol(measurements, symbol, base_unit)
        if value is not None:
            return value
    return _first_unit(measurements, base_unit)


def format_number(value: float) -> str:
    if math.isfinite(value) and abs(value - round(value)) < 1e-10:
        return str(int(round(value)))
    return f"{value:.10g}"


def _float_from_output(output: str) -> float:
    matches = re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", output)
    if not matches:
        raise ValueError(f"No numeric value in Python output: {output!r}")
    return float(matches[-1])


def _run_python_calculation(
    code: str,
    fallback: float,
    *,
    tool: PythonTool | None = None,
) -> tuple[float, list[dict[str, str]]]:
    tool = tool or PythonTool(timeout=3.0)
    try:
        result = tool.execute(code)
    except PythonToolError as exc:
        return fallback, [{"tool": "python", "code": code, "output": f"[REJECTED] {exc}"}]
    if not result.ok:
        return fallback, [{"tool": "python", "code": result.code, "output": result.output}]
    try:
        value = _float_from_output(result.stdout)
    except ValueError:
        value = fallback
    return value, [{"tool": "python", "code": result.code, "output": result.stdout.strip()}]


def _answer(
    value: float,
    unit: str,
    formula: str,
    substitution: str,
    *,
    tool_calls: list[dict[str, str]] | None = None,
    confidence: float = 0.86,
) -> PhysicsAnswer:
    formatted = format_number(value)
    return PhysicsAnswer(
        answer=formatted,
        unit=unit,
        explanation=f"Using {formula}, {substitution}, so the answer is {formatted} {unit}.",
        cot=[
            "Identify the relevant quantities and convert them to SI units.",
            f"Apply {formula}.",
            f"Use the Python calculation tool to compute {substitution}.",
        ],
        premises=[formula],
        confidence=confidence,
        tool_calls=tool_calls or [],
    )


def solve_physics_question(question: str, *, tool: PythonTool | None = None) -> dict[str, Any]:
    q_lower = question.lower()
    measurements = extract_measurements(question)

    if "capacitor" in q_lower and "energy" in q_lower:
        capacitance = _quantity(measurements, ("C",), "F")
        voltage = _quantity(measurements, ("U", "V"), "V")
        if capacitance is not None and voltage is not None:
            fallback = 0.5 * capacitance * voltage * voltage
            code = f"C = {capacitance!r}\nU = {voltage!r}\nE = 0.5 * C * U**2\nE"
            value, tool_calls = _run_python_calculation(code, fallback, tool=tool)
            return _answer(
                value,
                "J",
                "E = 1/2 C U^2",
                f"E = 0.5 * {capacitance:g} * {voltage:g}^2",
                tool_calls=tool_calls,
            ).as_dict()

    if "charge" in q_lower and "capacitor" in q_lower:
        capacitance = _quantity(measurements, ("C",), "F")
        voltage = _quantity(measurements, ("U", "V"), "V")
        if capacitance is not None and voltage is not None:
            fallback = capacitance * voltage
            code = f"C = {capacitance!r}\nU = {voltage!r}\nQ = C * U\nQ"
            value, tool_calls = _run_python_calculation(code, fallback, tool=tool)
            return _answer(
                value,
                "C",
                "Q = C U",
                f"Q = {capacitance:g} * {voltage:g}",
                tool_calls=tool_calls,
            ).as_dict()

    voltage = _quantity(measurements, ("U", "V"), "V")
    current = _quantity(measurements, ("I",), "A")
    resistance = _quantity(measurements, ("R",), "ohm")
    power = _quantity(measurements, ("P",), "W")

    if "resistance" in q_lower and voltage is not None and current is not None:
        fallback = voltage / current
        code = f"U = {voltage!r}\nI = {current!r}\nR = U / I\nR"
        value, tool_calls = _run_python_calculation(code, fallback, tool=tool)
        return _answer(value, "ohm", "R = U / I", f"R = {voltage:g} / {current:g}", tool_calls=tool_calls).as_dict()

    if ("current" in q_lower or "amper" in q_lower) and voltage is not None and resistance is not None:
        fallback = voltage / resistance
        code = f"U = {voltage!r}\nR = {resistance!r}\nI = U / R\nI"
        value, tool_calls = _run_python_calculation(code, fallback, tool=tool)
        return _answer(value, "A", "I = U / R", f"I = {voltage:g} / {resistance:g}", tool_calls=tool_calls).as_dict()

    if "voltage" in q_lower and current is not None and resistance is not None:
        fallback = current * resistance
        code = f"I = {current!r}\nR = {resistance!r}\nU = I * R\nU"
        value, tool_calls = _run_python_calculation(code, fallback, tool=tool)
        return _answer(value, "V", "U = I R", f"U = {current:g} * {resistance:g}", tool_calls=tool_calls).as_dict()

    if "power" in q_lower:
        if voltage is not None and current is not None:
            fallback = voltage * current
            code = f"U = {voltage!r}\nI = {current!r}\nP = U * I\nP"
            value, tool_calls = _run_python_calculation(code, fallback, tool=tool)
            return _answer(value, "W", "P = U I", f"P = {voltage:g} * {current:g}", tool_calls=tool_calls).as_dict()
        if current is not None and resistance is not None:
            fallback = current * current * resistance
            code = f"I = {current!r}\nR = {resistance!r}\nP = I**2 * R\nP"
            value, tool_calls = _run_python_calculation(code, fallback, tool=tool)
            return _answer(value, "W", "P = I^2 R", f"P = {current:g}^2 * {resistance:g}", tool_calls=tool_calls).as_dict()
        if voltage is not None and resistance is not None:
            fallback = voltage * voltage / resistance
            code = f"U = {voltage!r}\nR = {resistance!r}\nP = U**2 / R\nP"
            value, tool_calls = _run_python_calculation(code, fallback, tool=tool)
            return _answer(value, "W", "P = U^2 / R", f"P = {voltage:g}^2 / {resistance:g}", tool_calls=tool_calls).as_dict()

    force = _quantity(measurements, ("F",), "N")
    charge = _quantity(measurements, ("q", "Q"), "C")
    if "electric field" in q_lower and force is not None and charge is not None:
        fallback = force / charge
        code = f"F = {force!r}\nq = {charge!r}\nE = F / q\nE"
        value, tool_calls = _run_python_calculation(code, fallback, tool=tool)
        return _answer(value, "N/C", "E = F / q", f"E = {force:g} / {charge:g}", tool_calls=tool_calls).as_dict()

    distance = _quantity(measurements, ("d",), "m")
    if "electric field" in q_lower and voltage is not None and distance is not None:
        fallback = voltage / distance
        code = f"U = {voltage!r}\nd = {distance!r}\nE = U / d\nE"
        value, tool_calls = _run_python_calculation(code, fallback, tool=tool)
        return _answer(value, "V/m", "E = U / d", f"E = {voltage:g} / {distance:g}", tool_calls=tool_calls).as_dict()

    if "series" in q_lower:
        resistors = [m.value for m in measurements if m.base_unit == "ohm"]
        if len(resistors) >= 2:
            fallback = sum(resistors)
            code = f"resistors = {resistors!r}\nR_eq = sum(resistors)\nR_eq"
            value, tool_calls = _run_python_calculation(code, fallback, tool=tool)
            return _answer(value, "ohm", "R_eq = sum R_i for series resistors", f"R_eq = sum({resistors})", tool_calls=tool_calls).as_dict()

    if "parallel" in q_lower:
        resistors = [m.value for m in measurements if m.base_unit == "ohm" and m.value != 0]
        if len(resistors) >= 2:
            fallback = 1.0 / sum(1.0 / r for r in resistors)
            code = f"resistors = {resistors!r}\nR_eq = 1 / sum(1 / r for r in resistors)\nR_eq"
            value, tool_calls = _run_python_calculation(code, fallback, tool=tool)
            return _answer(value, "ohm", "1/R_eq = sum 1/R_i for parallel resistors", f"R_eq = 1 / sum(1/R_i)", tool_calls=tool_calls).as_dict()

    return {
        "answer": "Uncertain",
        "explanation": "The baseline calculator could not identify a supported formula from the question.",
        "cot": ["Parse known quantities.", "No supported deterministic formula matched."],
        "premises": [],
        "confidence": 0.2,
        "tool_calls": [],
    }
