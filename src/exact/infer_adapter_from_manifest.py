from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Sample:
    problem_id: str
    category: str
    prompt_ids: list[int]
    gold_ids: list[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate answers from a LoRA adapter on a few pre-tokenized manifest rows."
    )
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/no_tool_manifest.csv"))
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--adapter-dir", type=Path, default=Path("outputs/qwen3_5_4b_no_tool_lora"))
    parser.add_argument("--eval-output-dir", type=Path, default=Path("outputs/eval_adapter"))
    parser.add_argument("--category", choices=["logic", "physics", "physics_tool_call"], default=None)
    parser.add_argument("--limit-examples", type=int, default=3)
    parser.add_argument(
        "--limit-per-category",
        type=int,
        default=None,
        help="Balanced mode: load up to this many rows per category. Overrides --limit-examples.",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--prompt-preview-chars", type=int, default=1200)
    parser.add_argument("--print-examples", type=int, default=3)
    return parser.parse_args()


def _first_loss_index(mask: list[int]) -> int:
    for idx, keep in enumerate(mask):
        if keep == 1:
            return idx
    return len(mask)


def load_samples(
    manifest_path: Path,
    *,
    category: str | None,
    limit_examples: int,
    limit_per_category: int | None,
    offset: int,
) -> list[Sample]:
    samples: list[Sample] = []
    matched_by_category: Counter[str] = Counter()
    kept_by_category: Counter[str] = Counter()
    with manifest_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if category is not None and row["category"] != category:
                continue
            row_category = row["category"]
            if matched_by_category[row_category] < offset:
                matched_by_category[row_category] += 1
                continue
            matched_by_category[row_category] += 1
            if limit_per_category is not None and kept_by_category[row_category] >= limit_per_category:
                continue
            input_ids = json.loads(row["input_ids_json"])
            mask = json.loads(row["mask_json"])
            if len(input_ids) != len(mask):
                raise ValueError(f"Length mismatch in row {row['problem_id']}")
            split_at = _first_loss_index(mask)
            if split_at == len(input_ids):
                continue
            samples.append(
                Sample(
                    problem_id=row["problem_id"],
                    category=row["category"],
                    prompt_ids=input_ids[:split_at],
                    gold_ids=input_ids[split_at:],
                )
            )
            kept_by_category[row_category] += 1
            if limit_per_category is None and len(samples) >= limit_examples:
                break
    if not samples:
        raise ValueError(
            "No samples loaded. Check --manifest, --category, --offset, and --limit-per-category."
        )
    return samples


def _shorten(text: str, limit: int) -> str:
    text = text.strip()
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n... [truncated]"


def _extract_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def extract_final_answer(text: str | None) -> str:
    if text is None:
        return "NOT_FOUND"
    payload = _extract_json_object(text)
    if payload is not None and "answer" in payload:
        return _stringify(payload.get("answer")).strip() or "NOT_FOUND"

    matches = re.findall(r"\\boxed\{([^}]*)(?:\}|$)", text)
    if matches:
        non_empty = [match.strip() for match in matches if match.strip()]
        return non_empty[-1] if non_empty else matches[-1].strip()

    patterns = [
        r"The final answer is:\s*([^\n]+)",
        r"Final answer is:\s*([^\n]+)",
        r"Final answer\s*[:：]\s*([^\n]+)",
        r"final answer\s*[:：]\s*([^\n]+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].strip()

    matches = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?", text)
    if matches:
        return matches[-1]

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else "NOT_FOUND"


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


_NUMBER_RE = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _extract_number(text: str) -> float | None:
    matches = _NUMBER_RE.findall(text)
    if not matches:
        return None
    try:
        return float(matches[-1].replace(",", ""))
    except ValueError:
        return None


def _normalize_answer_text(text: str) -> str:
    text = text.strip().strip("\"'`")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .。").lower()


def _normalize_unit(unit: str) -> str:
    unit = unit.strip().strip("\"'`")
    unit = unit.replace("μ", "u").replace("µ", "u")
    unit = unit.replace("Ω", "ohm").replace("Ω", "ohm")
    unit = unit.lower()
    unit = re.sub(r"\s+", "", unit)
    unit = unit.strip(".,;")
    if unit == "ohms":
        unit = "ohm"
    return unit


def _unit_from_answer(answer: str) -> str:
    answer = answer.strip()
    match = re.search(
        r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?\s*([A-Za-zμµΩΩ/^\-*·.]+)\s*$",
        answer,
    )
    return match.group(1).strip(".,;") if match else ""


def _extract_fields(text: str) -> tuple[str, str, bool]:
    payload = _extract_json_object(text)
    if payload is not None:
        answer = _stringify(payload.get("answer")).strip()
        unit = _stringify(payload.get("unit")).strip()
        if not unit:
            unit = _unit_from_answer(answer)
        return answer or "NOT_FOUND", unit, True
    answer = extract_final_answer(text)
    return answer, _unit_from_answer(answer), False


def verify_answer(gold: str, predicted: str) -> bool:
    gold_num = _extract_number(gold)
    predicted_num = _extract_number(predicted)
    if gold_num is not None and predicted_num is not None:
        return math.isclose(gold_num, predicted_num, rel_tol=1e-2, abs_tol=1e-5)
    return _normalize_answer_text(gold) == _normalize_answer_text(predicted)


def verify_unit(gold_unit: str, predicted_unit: str) -> bool:
    if not gold_unit.strip():
        return True
    return _normalize_unit(gold_unit) == _normalize_unit(predicted_unit)


def evaluate_prediction(sample: Sample, gold: str, prediction: str) -> dict[str, Any]:
    gold_answer, gold_unit, gold_json_ok = _extract_fields(gold)
    predicted_answer, predicted_unit, prediction_json_ok = _extract_fields(prediction)
    answer_correct = verify_answer(gold_answer, predicted_answer)
    unit_correct = verify_unit(gold_unit, predicted_unit)
    needs_unit = sample.category in {"physics", "physics_tool_call"} and bool(gold_unit.strip())
    correct = answer_correct and (unit_correct if needs_unit else True)
    return {
        "problem_id": sample.problem_id,
        "category": sample.category,
        "gold_answer": gold_answer,
        "predicted_answer": predicted_answer,
        "gold_unit": gold_unit,
        "predicted_unit": predicted_unit,
        "gold_json_ok": gold_json_ok,
        "prediction_json_ok": prediction_json_ok,
        "answer_correct": answer_correct,
        "unit_correct": unit_correct,
        "correct": correct,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _format_pct(value: float) -> str:
    return f"{value:.1f}%"


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    totals = Counter(row["category"] for row in rows)
    correct = Counter(row["category"] for row in rows if row["correct"])
    grand_total = sum(totals.values())
    result_rows: list[dict[str, Any]] = []
    for category in sorted(totals):
        total = totals[category]
        num_correct = correct[category]
        result_rows.append(
            {
                "category": category,
                "correct": num_correct,
                "total": total,
                "weightage": _format_pct(total / grand_total * 100),
                "percentage": _format_pct(num_correct / total * 100),
                "contribution": _format_pct(num_correct / grand_total * 100),
            }
        )
    total_correct = sum(correct.values())
    result_rows.append(
        {
            "category": "TOTAL",
            "correct": total_correct,
            "total": grand_total,
            "weightage": "100.0%",
            "percentage": _format_pct(total_correct / grand_total * 100),
            "contribution": _format_pct(total_correct / grand_total * 100),
        }
    )
    return result_rows


def print_results_table(rows: list[dict[str, Any]]) -> None:
    headers = ["category", "correct", "total", "weightage", "percentage", "contribution"]
    widths = {
        header: max(len(header), *(len(str(row[header])) for row in rows))
        for header in headers
    }
    print("\nRESULTS")
    print("  ".join(header.ljust(widths[header]) for header in headers))
    for row in rows:
        print("  ".join(str(row[header]).ljust(widths[header]) for header in headers))


def save_evaluation(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    validation_fields = [
        "problem_id",
        "category",
        "correct",
        "answer_correct",
        "unit_correct",
        "gold_json_ok",
        "prediction_json_ok",
        "gold_answer",
        "predicted_answer",
        "gold_unit",
        "predicted_unit",
        "prompt",
        "gold",
        "raw_output",
    ]
    result_fields = ["category", "correct", "total", "weightage", "percentage", "contribution"]
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "validation.csv", rows, validation_fields)
    mistakes = [row for row in rows if not row["correct"]]
    _write_csv(output_dir / "mistakes.csv", mistakes, validation_fields)
    mistakes_dir = output_dir / "mistakes"
    for category in sorted({row["category"] for row in mistakes}):
        category_rows = [row for row in mistakes if row["category"] == category]
        _write_csv(mistakes_dir / f"{category}.csv", category_rows, validation_fields)
    result_rows = summarize(rows)
    _write_csv(output_dir / "results.csv", result_rows, result_fields)
    print_results_table(result_rows)
    print(f"\nSaved validation to {output_dir / 'validation.csv'}")
    print(f"Saved summary to {output_dir / 'results.csv'}")
    print(f"Saved mistakes to {output_dir / 'mistakes.csv'}")


def _model_input_device(model):
    import torch

    try:
        device = model.get_input_embeddings().weight.device
    except Exception:
        device = next(model.parameters()).device
    if device.type == "meta":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def main() -> None:
    args = parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    samples = load_samples(
        args.manifest,
        category=args.category,
        limit_examples=args.limit_examples,
        limit_per_category=args.limit_per_category,
        offset=args.offset,
    )
    print(
        {
            "samples": len(samples),
            "manifest": str(args.manifest),
            "adapter_dir": str(args.adapter_dir),
            "categories": sorted({sample.category for sample in samples}),
        }
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.adapter_dir,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map="auto",
        torch_dtype=dtype,
        load_in_4bit=args.load_in_4bit,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.eval()

    device = _model_input_device(model)
    do_sample = args.temperature > 0
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": do_sample,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p

    rows: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples, start=1):
        input_ids = torch.tensor([sample.prompt_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_kwargs,
            )[0]
        generated_ids = output_ids[len(sample.prompt_ids) :]

        prompt = tokenizer.decode(sample.prompt_ids, skip_special_tokens=False)
        prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        gold = tokenizer.decode(sample.gold_ids, skip_special_tokens=True).strip()
        eval_row = evaluate_prediction(sample, gold, prediction)
        eval_row.update(
            {
                "prompt": prompt,
                "gold": gold,
                "raw_output": prediction,
            }
        )
        rows.append(eval_row)

        if idx <= args.print_examples:
            print("\n" + "=" * 80)
            print(f"[{idx}] {sample.problem_id} ({sample.category}) correct={eval_row['correct']}")
            print("\nPROMPT")
            print(_shorten(prompt, args.prompt_preview_chars))
            print("\nMODEL OUTPUT")
            print(prediction)
            print("\nGOLD")
            print(gold)

    save_evaluation(args.eval_output_dir, rows)


if __name__ == "__main__":
    main()
