#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python -m exact.train_unsloth_from_manifest \
  --config configs/colab_unsloth_tool_always.json "$@"

