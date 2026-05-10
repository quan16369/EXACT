#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python -m exact.infer_adapter_from_manifest \
  --manifest data/processed/tool_always_manifest.csv \
  --model-name-or-path Qwen/Qwen3.5-4B \
  --adapter-dir outputs/qwen3_5_4b_tool_always_lora \
  --eval-output-dir outputs/eval_tool_always "$@"
