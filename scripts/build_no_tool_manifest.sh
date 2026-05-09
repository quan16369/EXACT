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
  --output data/processed/no_tool_logic_manifest.csv

PYTHONPATH=src python scripts/export_manifest.py \
  --input "${PHYSICS_INPUT}" \
  --task physics \
  --tokenizer "${TOKENIZER}" \
  --output data/processed/no_tool_physics_manifest.csv

PYTHONPATH=src python scripts/merge_manifests.py \
  --input data/processed/no_tool_logic_manifest.csv data/processed/no_tool_physics_manifest.csv \
  --output data/processed/no_tool_manifest.csv \
  --repeat-category logic=2,physics=1

PYTHONPATH=src python scripts/inspect_manifest.py \
  --manifest data/processed/no_tool_manifest.csv

