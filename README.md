# EXACT 2026 Starter Repo

This repo is a pragmatic scaffold for EXACT 2026: explainable educational QA over
logic/regulation questions and physics questions.

The competition page currently says the logic dataset has 464 records and 913
questions, while the physics dataset has 5,520 problems. Treat a logic "record"
as a container with multiple questions. After flattening, the trainable units are
roughly 913 logic examples plus 5,520 physics examples.

## Strategy

Use a hybrid system, not a pure fine-tuned model.

- Logic: `premises-NL + question -> relevant premises -> structured hints/FOL -> rule engine -> JSON answer`.
- Physics: `question -> variables/formula -> Python calculator -> JSON answer`.
- SFT: teach the small open-source model to produce clean explanations, select
  premises, and follow the required JSON format.

For the official rule constraint, keep all LLM components open-source and at or
below 8B parameters.

## Repo Layout

- `src/exact/data.py`: normalize raw EXACT-style records into one example per question.
- `src/exact/tokenization.py`: pre-tokenize prompt/completion and write
  `input_ids_json` + `mask_json` manifests.
- `src/exact/train_from_manifest.py`: feed pre-tokenized examples to a causal LM.
- `src/exact/logic.py`: small baseline logic premise selector/rule engine.
- `src/exact/physics.py`: deterministic baseline calculator for common electricity formulas.
- `src/exact/infer.py`: route a request to logic or physics solver.
- `src/exact/api.py`: optional FastAPI endpoint.
- `data/examples/`: tiny local samples for smoke tests.

## Pre-tokenize Then Feed Into Model

The Nemotron-Kaggle pattern is:

1. Render prompt and completion.
2. Tokenize both before training.
3. Concatenate: `input_ids = prompt_ids + completion_ids`.
4. Build loss mask:
   - `mask = [0] * len(prompt_ids) + [1] * len(completion_ids)`
5. At training time:
   - `labels = [token if mask_i == 1 else -100 for token, mask_i in zip(input_ids, mask)]`
   - collator pads `input_ids`, `attention_mask`, and `labels`.
   - causal LM loss ignores `-100`.

That means after pre-tokenization you do not pass text into the model. You pass:

```python
batch = {
    "input_ids": LongTensor[B, T],
    "attention_mask": LongTensor[B, T],
    "labels": LongTensor[B, T],  # prompt positions are -100 when masked
}
outputs = model(**batch)
loss = outputs.loss
```

This repo intentionally uses only prompt masking. There is no full-loss or hybrid
instruction loss path, because the model should learn the answer/explanation
format, not memorize the prompt text.

Manifest schema is aligned with the reference repo:

```text
problem_id,source_problem_id,category,segment,num_loss_tokens,
completion_token_count,token_count,input_ids_json,mask_json
```

## Export A Manifest

Use a real model tokenizer for actual training. For a dependency-light smoke
test, use `--tokenizer whitespace`; for real training, replace it with the model
tokenizer path/name.

```bash
PYTHONPATH=src python scripts/export_manifest.py \
  --input data/examples/logic.jsonl data/examples/physics.jsonl \
  --tokenizer whitespace \
  --output data/processed/train_manifest.csv \
  --metadata-output data/processed/train_manifest.meta.json
```

Inspect the manifest before training:

```bash
PYTHONPATH=src python scripts/inspect_manifest.py \
  --manifest data/processed/train_manifest.csv
```

## Train

The Qwen config assumes the model id/path is `Qwen/Qwen3.5-4B-Instruct`. If your
local Hugging Face id differs, edit `configs/qwen3_5_4b_lora.json` or override
`--model-name-or-path`.

```bash
scripts/train_qwen3_5_4b.sh
```

Equivalent explicit command:

```bash
PYTHONPATH=src python -m exact.train_from_manifest \
  --config configs/qwen3_5_4b_lora.json
```

If `--lora-r 0`, the script performs normal full fine-tuning. The default config
uses QLoRA-style 4-bit loading, bf16 compute, Qwen projection MLP LoRA targets,
and the same masked-loss contract as the reference repo.

Unsloth trainer variant:

```bash
scripts/train_unsloth_tool_always_qwen3_5_4b.sh
```

or:

```bash
scripts/train_unsloth_no_tool_qwen3_5_4b.sh
```

This still trains from `input_ids_json`/`mask_json`; it does not call Unsloth's
`train_on_responses_only`, because prompt masking is already encoded in
`labels = token if mask == 1 else -100`.

## Run Baseline Inference

```bash
PYTHONPATH=src python -m exact.infer --input data/examples/logic_request.json
PYTHONPATH=src python -m exact.infer --input data/examples/physics_request.json
```

## Python Calculation Tool

The physics path now uses a Python execution tool similar in spirit to the AIMO
notebook:

```text
question
-> parse variables/formula
-> build a short Python snippet
-> PythonTool.execute(code)
-> read stdout
-> answer + explanation + cot + tool_calls
```

The tool implementation is in `src/exact/tools/python_tool.py`. It executes code
in an isolated subprocess with timeout and AST checks. The physics solver calls
it from `src/exact/physics.py` through `_run_python_calculation`.

Example output contains a trace:

```json
"tool_calls": [
  {
    "tool": "python",
    "code": "C = 0.1\nU = 30.0\nE = 0.5 * C * U ** 2\nprint(E)",
    "output": "45.0"
  }
]
```

For the final competition endpoint, keep `answer` and `explanation` mandatory.
If the evaluator rejects unknown fields, strip `tool_calls` at the API boundary
and keep the same evidence in `cot`.

To train Qwen to emit tool-call format, export a separate manifest:

```bash
PYTHONPATH=src python scripts/export_tool_call_manifest.py \
  --input data/examples/physics.jsonl \
  --tokenizer whitespace \
  --output data/processed/tool_call_manifest.csv \
  --raw-output data/processed/tool_call_raw.csv
```

Each completion teaches this sequence:

```text
assistant: {"tool":"python","code":"..."}
tool: 45.0
assistant: {"answer":"45","unit":"J","explanation":"..."}
```

The prompt is still masked. The assistant tool call, tool output, and final
assistant JSON are unmasked completion tokens. For real training, use the Qwen
tokenizer instead of `whitespace`.

You can train one adapter on logic and tool-call data together by merging
manifests:

```bash
PYTHONPATH=src python scripts/merge_manifests.py \
  --input data/processed/train_manifest.csv data/processed/tool_call_manifest.csv \
  --output data/processed/mixed_manifest.csv \
  --repeat-category logic=2,physics_tool_call=2

PYTHONPATH=src python scripts/inspect_manifest.py \
  --manifest data/processed/mixed_manifest.csv

PYTHONPATH=src python -m exact.train_from_manifest \
  --config configs/qwen3_5_4b_lora.json \
  --manifest data/processed/mixed_manifest.csv
```

The repeat map is optional. It is useful because raw physics rows are usually
more numerous than logic rows, while tool-call rows are high-value for teaching
the interaction pattern.

If you want every physics example to call Python first, do not include the plain
`physics` final-answer rows. Build this manifest instead:

```bash
scripts/build_tool_always_manifest.sh whitespace \
  data/raw/Logic_Based_Educational_Queries.json \
  data/raw/Physics_Problems_Text_Only.csv
```

For real training, pass the Qwen tokenizer/model path:

```bash
scripts/build_tool_always_manifest.sh models/qwen3_5_4b_tokenizer \
  data/raw/Logic_Based_Educational_Queries.json \
  data/raw/Physics_Problems_Text_Only.csv

scripts/train_tool_always_qwen3_5_4b.sh
```

That manifest contains only:

```text
logic
physics_tool_call
```

So the model learns: logic answers directly from premises, and physics always
emits a Python tool call before the final JSON.

For a no-tool baseline, build and train this manifest instead:

```bash
scripts/build_no_tool_manifest.sh models/qwen3_5_4b_tokenizer \
  data/raw/Logic_Based_Educational_Queries.json \
  data/raw/Physics_Problems_Text_Only.csv

scripts/train_no_tool_qwen3_5_4b.sh
```

That manifest contains only:

```text
logic
physics
```

It still uses prompt masking and pre-tokenized `input_ids_json`/`mask_json`, but
physics completions are direct final JSON answers rather than tool-call traces.

Optional API:

```bash
PYTHONPATH=src uvicorn exact.api:app --host 0.0.0.0 --port 8000
```

Request shape:

```json
{
  "question": "Can the student receive a B?",
  "premises-NL": ["If a student misses the lab exam, the lab score is zero."]
}
```

Response shape:

```json
{
  "answer": "No",
  "explanation": "...",
  "premises": ["..."],
  "confidence": 0.72
}
```

## Do We Need To Train Tool Use?

Not for the first strong baseline. For physics, make the orchestrator call Python
directly after extracting variables and formulas. Training the model to emit tool
calls is optional and usually needs more curated examples. For this competition,
the safer path is deterministic calculation plus model-generated explanation.

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests
```
