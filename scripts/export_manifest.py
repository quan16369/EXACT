#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from exact.data import normalize_records, write_normalized_jsonl
from exact.tokenization import iter_manifest_records, load_tokenizer, write_manifest_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--normalized-output", type=Path, default=None)
    parser.add_argument("--task", choices=["all", "logic", "physics"], default="all")
    parser.add_argument("--tokenizer", default=None, help="HF tokenizer/model name or local path.")
    parser.add_argument("--tokenizer-file", type=Path, default=None, help="tokenizers JSON file.")
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--eos-text", default=None)
    parser.add_argument("--metadata-output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = normalize_records(args.input)
    if args.task != "all":
        examples = [example for example in examples if example.task == args.task]
    if args.normalized_output is not None:
        write_normalized_jsonl(examples, args.normalized_output)
    tokenizer = load_tokenizer(args.tokenizer, args.tokenizer_file)
    records = list(
        iter_manifest_records(
            examples,
            tokenizer,
            max_seq_len=args.max_seq_len,
            eos_text=args.eos_text,
        )
    )
    count = write_manifest_csv(records, args.output)
    categories = Counter(record.category for record in records)
    print(f"Wrote {count} manifest rows to {args.output}")
    print(f"Categories: {dict(categories)}")
    if args.metadata_output is not None:
        args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
        tokenizer_name = args.tokenizer or str(args.tokenizer_file)
        args.metadata_output.write_text(
            json.dumps(
                {
                    "tokenizer": tokenizer_name,
                    "max_seq_len": args.max_seq_len,
                    "loss_policy": "completion_only_mask_prompt",
                    "rows": count,
                    "categories": dict(categories),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
