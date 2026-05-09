#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

FIELDNAMES = [
    "problem_id",
    "source_problem_id",
    "category",
    "segment",
    "num_loss_tokens",
    "completion_token_count",
    "token_count",
    "input_ids_json",
    "mask_json",
]


def parse_repeat(raw: str) -> dict[str, int]:
    repeats: dict[str, int] = {}
    if not raw:
        return repeats
    for item in raw.split(","):
        if not item.strip():
            continue
        category, value = item.split("=", 1)
        repeats[category.strip()] = int(value)
    return repeats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--repeat-category",
        default="",
        help="Optional category repeat map, e.g. logic=2,physics_tool_call=2.",
    )
    parser.add_argument(
        "--include-category",
        default="",
        help="Optional comma-separated category allowlist.",
    )
    parser.add_argument(
        "--exclude-category",
        default="",
        help="Optional comma-separated category denylist.",
    )
    return parser.parse_args()


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            missing = [field for field in FIELDNAMES if field not in (reader.fieldnames or [])]
            if missing:
                raise ValueError(f"{path} missing fields: {missing}")
            for row in reader:
                rows.append({field: row[field] for field in FIELDNAMES})
    return rows


def parse_category_set(raw: str) -> set[str]:
    return {item.strip() for item in raw.split(",") if item.strip()}


def filter_rows(
    rows: list[dict[str, str]],
    *,
    include_categories: set[str],
    exclude_categories: set[str],
) -> list[dict[str, str]]:
    filtered: list[dict[str, str]] = []
    for row in rows:
        category = row["category"]
        if include_categories and category not in include_categories:
            continue
        if category in exclude_categories:
            continue
        filtered.append(row)
    return filtered


def dedupe_and_repeat(rows: list[dict[str, str]], repeats: dict[str, int]) -> list[dict[str, str]]:
    seen: set[str] = set()
    merged: list[dict[str, str]] = []
    for row in rows:
        base_id = row["problem_id"]
        repeat = max(1, repeats.get(row["category"], 1))
        for idx in range(repeat):
            copied = dict(row)
            copied["problem_id"] = base_id if repeat == 1 else f"{base_id}:repeat{idx}"
            if copied["problem_id"] in seen:
                suffix = 1
                candidate = f"{copied['problem_id']}:dup{suffix}"
                while candidate in seen:
                    suffix += 1
                    candidate = f"{copied['problem_id']}:dup{suffix}"
                copied["problem_id"] = candidate
            seen.add(copied["problem_id"])
            merged.append(copied)
    return merged


def main() -> None:
    args = parse_args()
    repeats = parse_repeat(args.repeat_category)
    rows = filter_rows(
        read_rows(args.input),
        include_categories=parse_category_set(args.include_category),
        exclude_categories=parse_category_set(args.exclude_category),
    )
    rows = dedupe_and_repeat(rows, repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter(row["category"] for row in rows)
    print(f"Wrote {len(rows)} rows to {args.output}")
    print(f"Categories: {dict(counts)}")


if __name__ == "__main__":
    main()
