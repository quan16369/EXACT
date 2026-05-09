# Project Layout

Raw data and tokenizer assets are organized so pre-tokenized manifests can be
reproduced.

```text
data/raw/
  Logic_Based_Educational_Queries.json
  Physics_Problems_Text_Only.csv

models/qwen3_5_4b_tokenizer/
  config.json
  tokenizer.json
  tokenizer_config.json
  vocab.json
  merges.txt
  chat_template.jinja

data/processed/
  logic_manifest.csv
  physics_tool_call_manifest.csv
  tool_always_manifest.csv
  no_tool_manifest.csv
```

`data/processed/*.csv` are generated artifacts. They contain token ids generated
with the tokenizer path passed to the exporter. Rebuild them whenever tokenizer
files or prompt templates change.

The default build command uses the local tokenizer folder:

```bash
scripts/build_tool_always_manifest.sh
```

For a dependency-light smoke test, pass `whitespace` explicitly:

```bash
scripts/build_tool_always_manifest.sh whitespace
```

There is also a no-tool baseline that trains only final JSON completions:

```bash
scripts/build_no_tool_manifest.sh whitespace
```

The no-tool manifest contains `logic` and plain `physics` categories, with no
`physics_tool_call` rows.
