#!/usr/bin/env bash
set -euo pipefail

TOKENIZER="${1:-models/qwen3_5_4b_tokenizer}"
LOGIC_INPUT="${2:-data/raw/Logic_Based_Educational_Queries.json}"
PHYSICS_INPUT="${3:-data/raw/Physics_Problems_Text_Only.csv}"
mkdir -p data/processed

PYTHONPATH=src python scripts/export_manifest.py \
  --input "${LOGIC_INPUT}" \
  --task logic \
  --tokenizer "${TOKENIZER}" \
  --output data/processed/logic_manifest.csv

PYTHONPATH=src python scripts/export_tool_call_manifest.py \
  --input "${PHYSICS_INPUT}" \
  --tokenizer "${TOKENIZER}" \
  --output data/processed/physics_tool_call_manifest.csv \
  --raw-output data/processed/physics_tool_call_raw.csv \
  --fallback-to-gold-answer

PYTHONPATH=src python scripts/merge_manifests.py \
  --input data/processed/logic_manifest.csv data/processed/physics_tool_call_manifest.csv \
  --output data/processed/tool_always_manifest.csv \
  --repeat-category logic=2,physics_tool_call=2

PYTHONPATH=src python scripts/inspect_manifest.py \
  --manifest data/processed/tool_always_manifest.csv
