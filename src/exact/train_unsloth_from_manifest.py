from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _load_config_defaults(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("--config must point to a JSON object")
    return data


def _default_path(defaults: dict[str, Any], key: str) -> Path | None:
    value = defaults.get(key)
    return Path(value) if value is not None else None


def parse_args() -> argparse.Namespace:
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("--config", type=Path, default=None)
    base_args, _ = base_parser.parse_known_args()
    defaults = _load_config_defaults(base_args.config)

    parser = argparse.ArgumentParser(parents=[base_parser])
    parser.add_argument("--manifest", type=Path, default=_default_path(defaults, "manifest"))
    parser.add_argument("--model-name-or-path", default=defaults.get("model_name_or_path"))
    parser.add_argument("--output-dir", type=Path, default=_default_path(defaults, "output_dir"))
    parser.add_argument(
        "--category",
        choices=["logic", "physics", "physics_tool_call"],
        default=defaults.get("category"),
    )
    parser.add_argument("--max-seq-len", type=int, default=int(defaults.get("max_seq_len", 8192)))
    parser.add_argument("--limit-examples", type=int, default=defaults.get("limit_examples"))
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=int(defaults.get("per_device_train_batch_size", 1)),
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=int(defaults.get("gradient_accumulation_steps", 16)),
    )
    parser.add_argument("--learning-rate", type=float, default=float(defaults.get("learning_rate", 2e-5)))
    parser.add_argument("--num-train-epochs", type=float, default=float(defaults.get("num_train_epochs", 3.0)))
    parser.add_argument("--warmup-ratio", type=float, default=float(defaults.get("warmup_ratio", 0.03)))
    parser.add_argument("--logging-steps", type=int, default=int(defaults.get("logging_steps", 10)))
    parser.add_argument("--seed", type=int, default=int(defaults.get("seed", 42)))
    parser.add_argument("--bf16", action="store_true", default=bool(defaults.get("bf16", True)))
    parser.add_argument("--load-in-4bit", action="store_true", default=bool(defaults.get("use_4bit", True)))
    parser.add_argument("--lora-r", type=int, default=int(defaults.get("lora_r", 16)))
    parser.add_argument("--lora-alpha", type=int, default=int(defaults.get("lora_alpha", 32)))
    parser.add_argument("--lora-dropout", type=float, default=float(defaults.get("lora_dropout", 0.05)))
    parser.add_argument(
        "--lora-target-modules",
        default=defaults.get(
            "lora_target_modules",
            "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        ),
    )
    parser.add_argument("--use-rslora", action="store_true", default=bool(defaults.get("use_rslora", False)))
    parser.add_argument(
        "--train-on-responses-only",
        action="store_true",
        default=bool(defaults.get("train_on_responses_only", False)),
        help="Do not use this with pre-tokenized masked manifests; kept only for explicit experiments.",
    )
    args = parser.parse_args()
    if args.manifest is None:
        parser.error("--manifest is required unless provided by --config")
    if args.model_name_or_path is None:
        parser.error("--model-name-or-path is required unless provided by --config")
    if args.output_dir is None:
        parser.error("--output-dir is required unless provided by --config")
    return args


def load_records(
    manifest_path: Path,
    *,
    category: str | None,
    max_seq_len: int,
    limit_examples: int | None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with manifest_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if category is not None and row["category"] != category:
                continue
            input_ids = json.loads(row["input_ids_json"])
            mask = json.loads(row["mask_json"])
            if len(input_ids) != len(mask):
                raise ValueError(f"Length mismatch in manifest row {row['problem_id']}")
            if len(input_ids) > max_seq_len:
                raise ValueError(
                    f"Manifest row {row['problem_id']} has {len(input_ids)} tokens > {max_seq_len}"
                )
            labels = [token if keep == 1 else -100 for token, keep in zip(input_ids, mask)]
            if all(label == -100 for label in labels):
                raise ValueError(f"Manifest row {row['problem_id']} has no unmasked loss tokens")
            records.append(
                {
                    "problem_id": row["problem_id"],
                    "category": row["category"],
                    "input_ids": input_ids,
                    "attention_mask": [1] * len(input_ids),
                    "labels": labels,
                }
            )
            if limit_examples is not None and len(records) >= limit_examples:
                break
    if not records:
        raise ValueError("No records loaded from manifest")
    records.sort(key=lambda record: record["problem_id"])
    return records


class MaskedDataCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        max_len = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        attention_mask = []
        labels = []
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(feature["attention_mask"] + [0] * pad_len)
            labels.append(feature["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _parse_target_modules(raw: str | list[str]) -> list[str]:
    if isinstance(raw, list):
        return raw
    if raw == "all-linear":
        return raw
    return [part.strip() for part in raw.split(",") if part.strip()]


def main() -> None:
    args = parse_args()

    from datasets import Dataset
    from transformers import Trainer, TrainingArguments
    from unsloth import FastLanguageModel

    records = load_records(
        args.manifest,
        category=args.category,
        max_seq_len=args.max_seq_len,
        limit_examples=args.limit_examples,
    )
    counts = Counter(record["category"] for record in records)
    print(
        {
            "records": len(records),
            "categories": dict(counts),
            "tokens": sum(len(record["input_ids"]) for record in records),
            "loss_tokens": sum(sum(1 for label in record["labels"] if label != -100) for record in records),
        }
    )
    dataset = Dataset.from_list(records)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name_or_path,
        max_seq_length=args.max_seq_len,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id or 0

    if args.lora_r > 0:
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,
            target_modules=_parse_target_modules(args.lora_target_modules),
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=args.seed,
            use_rslora=args.use_rslora,
            loftq_config=None,
        )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        bf16=args.bf16,
        seed=args.seed,
        remove_unused_columns=False,
        report_to="none",
        optim="adamw_8bit" if args.load_in_4bit else "adamw_torch",
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.95,
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        data_collator=MaskedDataCollator(pad_token_id=pad_token_id),
    )

    if args.train_on_responses_only:
        raise ValueError(
            "This script already uses pre-tokenized mask_json labels. "
            "Do not also call Unsloth train_on_responses_only."
        )

    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))


if __name__ == "__main__":
    main()

