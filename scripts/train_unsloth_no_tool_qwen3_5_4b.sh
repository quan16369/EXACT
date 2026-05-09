#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python -m exact.train_unsloth_from_manifest \
  --config configs/qwen3_5_4b_no_tool_lora.json "$@"

