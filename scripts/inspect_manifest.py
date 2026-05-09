#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    categories: Counter[str] = Counter()
    segments: Counter[str] = Counter()
    rows = 0
    total_tokens = 0
    total_loss_tokens = 0
    max_tokens = 0
    max_loss_tokens = 0
    bad_rows: list[str] = []

    with args.manifest.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows += 1
            input_ids = json.loads(row["input_ids_json"])
            mask = json.loads(row["mask_json"])
            if len(input_ids) != len(mask):
                bad_rows.append(f"{row['problem_id']}: len(input_ids) != len(mask)")
                continue
            loss_tokens = sum(1 for item in mask if item == 1)
            if loss_tokens != int(row["num_loss_tokens"]):
                bad_rows.append(f"{row['problem_id']}: num_loss_tokens mismatch")
            if len(input_ids) > args.max_seq_len:
                bad_rows.append(f"{row['problem_id']}: token_count > max_seq_len")
            categories[row["category"]] += 1
            segments[row.get("segment", "")] += 1
            total_tokens += len(input_ids)
            total_loss_tokens += loss_tokens
            max_tokens = max(max_tokens, len(input_ids))
            max_loss_tokens = max(max_loss_tokens, loss_tokens)

    summary = {
        "rows": rows,
        "categories": dict(categories),
        "segments": dict(segments),
        "total_tokens": total_tokens,
        "total_loss_tokens": total_loss_tokens,
        "max_tokens": max_tokens,
        "max_loss_tokens": max_loss_tokens,
        "bad_rows": bad_rows[:20],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if bad_rows:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

