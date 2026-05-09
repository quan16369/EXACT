#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from exact.data import normalize_records
from exact.tokenization import encode_text, load_tokenizer, write_manifest_csv, ManifestRecord
from exact.tool_call_data import build_tool_call_prompt, make_tool_call_completion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--tokenizer-file", type=Path, default=None)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--raw-output", type=Path, default=None)
    parser.add_argument(
        "--fallback-to-gold-answer",
        action="store_true",
        help="Emit a Python tool call that prints the gold answer when formula-code generation fails or disagrees.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = load_tokenizer(args.tokenizer, args.tokenizer_file)
    examples = [example for example in normalize_records(args.input) if example.task == "physics"]
    records: list[ManifestRecord] = []
    raw_rows: list[dict[str, str]] = []

    for example in examples:
        prompt_text = build_tool_call_prompt(example)
        completion_text = make_tool_call_completion(
            example,
            fallback_to_gold_answer=args.fallback_to_gold_answer,
        )
        if completion_text is None:
            continue
        prompt_ids = encode_text(tokenizer, prompt_text)
        completion_ids = encode_text(tokenizer, completion_text)
        input_ids = prompt_ids + completion_ids
        if len(input_ids) > args.max_seq_len:
            continue
        mask = [0] * len(prompt_ids) + [1] * len(completion_ids)
        records.append(
            ManifestRecord(
                problem_id=f"{example.example_id}:tool_call",
                source_problem_id=example.example_id,
                category="physics_tool_call",
                segment="exact_physics_tool_call.jsonl",
                num_loss_tokens=sum(mask),
                completion_token_count=len(completion_ids),
                token_count=len(input_ids),
                input_ids_json=json.dumps(input_ids),
                mask_json=json.dumps(mask),
            )
        )
        raw_rows.append(
            {
                "problem_id": example.example_id,
                "prompt": prompt_text,
                "completion": completion_text,
            }
        )

    count = write_manifest_csv(records, args.output)
    print(f"Wrote {count} tool-call manifest rows to {args.output}")

    if args.raw_output is not None:
        args.raw_output.parent.mkdir(parents=True, exist_ok=True)
        with args.raw_output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["problem_id", "prompt", "completion"])
            writer.writeheader()
            writer.writerows(raw_rows)


if __name__ == "__main__":
    main()
