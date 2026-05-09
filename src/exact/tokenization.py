from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .prompts import build_completion, build_user_prompt, system_prompt_for_task
from .schema import ExactExample


@dataclass(slots=True)
class ManifestRecord:
    problem_id: str
    source_problem_id: str
    category: str
    segment: str
    num_loss_tokens: int
    completion_token_count: int
    token_count: int
    input_ids_json: str
    mask_json: str


class WhitespaceTokenizer:
    """Small deterministic tokenizer for tests only."""

    eos_token = "<eos>"
    pad_token_id = 0

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {"<pad>": 0}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        tokens: list[int] = []
        for piece in text.replace("\n", " \n ").split():
            if piece not in self.vocab:
                self.vocab[piece] = len(self.vocab)
            tokens.append(self.vocab[piece])
        return tokens


def load_tokenizer(tokenizer_name_or_path: str | None = None, tokenizer_file: Path | None = None):
    if tokenizer_file is not None:
        from tokenizers import Tokenizer

        return Tokenizer.from_file(str(tokenizer_file))
    if tokenizer_name_or_path == "whitespace":
        return WhitespaceTokenizer()
    if not tokenizer_name_or_path:
        raise ValueError("Provide --tokenizer or --tokenizer-file")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def encode_text(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if hasattr(encoded, "ids"):
        return list(encoded.ids)
    return list(encoded)


def _has_chat_template(tokenizer: Any) -> bool:
    return bool(getattr(tokenizer, "chat_template", None)) and hasattr(
        tokenizer, "apply_chat_template"
    )


def render_prompt_text(tokenizer: Any, system_prompt: str, user_prompt: str) -> str:
    if _has_chat_template(tokenizer):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return (
        "<|im_start|>system\n"
        f"{system_prompt}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _completion_suffix(tokenizer: Any, eos_text: str | None) -> str:
    if eos_text is not None:
        return eos_text
    eos_token = getattr(tokenizer, "eos_token", None)
    if eos_token:
        return str(eos_token)
    return "<|im_end|>"


def build_manifest_record(
    example: ExactExample,
    tokenizer: Any,
    *,
    max_seq_len: int,
    eos_text: str | None = None,
) -> ManifestRecord:
    system_prompt = system_prompt_for_task(example.task)
    user_prompt = build_user_prompt(example)
    completion = build_completion(example)
    prompt_text = render_prompt_text(tokenizer, system_prompt, user_prompt)
    completion_text = completion + _completion_suffix(tokenizer, eos_text)
    prompt_ids = encode_text(tokenizer, prompt_text)
    completion_ids = encode_text(tokenizer, completion_text)
    input_ids = prompt_ids + completion_ids
    if len(input_ids) > max_seq_len:
        raise ValueError(
            f"{example.example_id} has {len(input_ids)} tokens, above max_seq_len={max_seq_len}"
        )
    mask = [0] * len(prompt_ids) + [1] * len(completion_ids)
    return ManifestRecord(
        problem_id=example.example_id,
        source_problem_id=example.example_id,
        category=example.task,
        segment=f"exact_{example.task}.jsonl",
        num_loss_tokens=sum(mask),
        completion_token_count=len(completion_ids),
        token_count=len(input_ids),
        input_ids_json=json.dumps(input_ids),
        mask_json=json.dumps(mask),
    )


def iter_manifest_records(
    examples: Iterable[ExactExample],
    tokenizer: Any,
    *,
    max_seq_len: int,
    eos_text: str | None = None,
) -> Iterable[ManifestRecord]:
    for example in examples:
        yield build_manifest_record(
            example,
            tokenizer,
            max_seq_len=max_seq_len,
            eos_text=eos_text,
        )


def write_manifest_csv(records: Iterable[ManifestRecord], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
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
    count = 0
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "problem_id": record.problem_id,
                    "source_problem_id": record.source_problem_id,
                    "category": record.category,
                    "segment": record.segment,
                    "num_loss_tokens": record.num_loss_tokens,
                    "completion_token_count": record.completion_token_count,
                    "token_count": record.token_count,
                    "input_ids_json": record.input_ids_json,
                    "mask_json": record.mask_json,
                }
            )
            count += 1
    return count
