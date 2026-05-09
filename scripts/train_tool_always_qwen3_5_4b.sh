#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python -m exact.train_from_manifest \
  --config configs/qwen3_5_4b_tool_always_lora.json "$@"

