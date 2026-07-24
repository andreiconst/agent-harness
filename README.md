# agent-harness

A coding agent built from scratch, with the goal of getting a good score on
[SWE-bench](https://www.swebench.com/). This is a learning project: the point
is to understand every layer of the stack — the tool-use loop, the tools
themselves, and the eval harness — rather than to wrap an existing agent SDK.

## How it works

- [`src/agent_harness/agent.py`](src/agent_harness/agent.py) — the core loop:
  call Claude with a task and a set of tools, execute whatever tool calls come
  back, feed the results back in, repeat until the model stops calling tools.
- [`src/agent_harness/tools/bash.py`](src/agent_harness/tools/bash.py) — a
  persistent bash session (matches Anthropic's `bash_20250124` tool spec), so
  `cd` / exported vars / activated venvs survive across tool calls.
- [`src/agent_harness/tools/editor.py`](src/agent_harness/tools/editor.py) —
  a file-editing tool (`text_editor_20250124`): view, create, str_replace,
  insert, undo_edit.
- [`src/agent_harness/swebench/`](src/agent_harness/swebench) — loads SWE-bench
  instances from Hugging Face and checks out the repo at the issue's base
  commit.

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env   # then fill in ANTHROPIC_API_KEY
```

## Usage

Run the agent on one SWE-bench instance and see the diff it produces:

```bash
python scripts/run_instance.py astropy__astropy-12907 --output preds.jsonl
```

Run the tool unit tests (no API key needed):

```bash
pytest
```

Evaluate predictions with the official docker-based SWE-bench harness (needs
Docker + `uv pip install -e ".[eval]"`):

```bash
python scripts/evaluate.py --predictions_path preds.jsonl \
    --dataset_name princeton-nlp/SWE-bench_Lite --run_id my_run
```

## Status

This is boilerplate: the loop and tools work end-to-end on a single instance,
but there's no batching, no environment setup beyond a plain `git checkout`,
and no self-correction beyond "keep calling tools until the model stops." See
[docs/ROADMAP.md](docs/ROADMAP.md) for what's next.
