# agent-harness

A coding agent built from scratch, with the goal of getting a good score on
[SWE-bench](https://www.swebench.com/). This is a learning project: the point
is to understand every layer of the stack — the tool-use loop, the tools
themselves, and the eval harness — rather than to wrap an existing agent SDK.

## How it works

### The agent loop

[`Agent.run()`](src/agent_harness/agent.py) is the whole loop, end to end:

1. Seed `messages` with one user turn: the SWE-bench issue's problem
   statement.
2. Call `client.messages.create(...)`, passing the full `messages` history,
   a fixed `SYSTEM_PROMPT`, and the three tool definitions below. Append
   whatever Claude replies with (`response.content`, a list of text and/or
   `tool_use` blocks) to `messages` as an assistant turn.
3. Check `response.stop_reason`:
   - If it's **not** `"tool_use"` (i.e. Claude just replied with text — it
     thinks it's done), break out of the loop.
   - If it **is** `"tool_use"`, take every `tool_use` block in the response
     and run it through `_execute_tool()`, which dispatches to `bash.run()`
     or `editor.run()` based on the block's `name` and passes `block.input`
     straight through as keyword arguments — **except** a call to `submit`
     (see below), which is intercepted before execution. Each execution —
     success or exception — becomes one `tool_result` block (with
     `is_error: true` on failure so Claude sees the tool call failed, not
     that it succeeded with empty output).
   - Append all the `tool_result` blocks as a single new user turn.
   - If one of this turn's tool calls was `submit`, stop — no further API
     call. Otherwise go back to step 2.
4. Stop after `stop_reason != "tool_use"`, a `submit` call, or `max_turns`
   round-trips (default 40), whichever comes first — this is the only budget
   control right now, there's no token or wall-clock limit yet.

After the loop ends, `Agent.diff()` just shells out to `git diff` in the
repo's working directory to capture everything the agent changed, which is
what gets submitted as the SWE-bench prediction.

There's no context trimming and no prompt caching yet — see
[docs/ROADMAP.md](docs/ROADMAP.md). On a real run against astropy, an
untightened version of this loop burned 40 turns and $2.50 mostly on failed
environment setup and redundant re-verification after the fix was already
found; the `submit` tool and output truncation below are the fixes for that.

### The tools

The agent has three tools: two modeled directly on the tool specs Anthropic
ships for computer-use/coding agents (`bash_20250124` and
`text_editor_20250728` — we implement the *execution* side ourselves, Claude
just gets the tool name/type and decides when to call them), plus one custom
tool for signaling completion.

**`bash`** ([`tools/bash.py`](src/agent_harness/tools/bash.py)) — a single
persistent `/bin/bash` subprocess per `Agent` instance, not a fresh
`subprocess.run()` per call. This matters because `cd some/dir`, `export
FOO=bar`, or activating a virtualenv in one tool call needs to still be in
effect on the next call, exactly like a human's terminal session. Mechanics:
  - A background thread continuously reads the subprocess's stdout/stderr
    (merged) into a queue.
  - Each call to `run(command)` writes `command` to the subprocess's stdin,
    followed by `echo '<random-uuid-sentinel>' $?`.
  - It then drains the output queue until it sees a line containing that
    sentinel — that's how it knows the command finished (bash gives no other
    signal) — and parses the trailing `$?` off that line as the exit code.
  - If no sentinel shows up within `timeout` seconds (default 60), it gives
    up and reports a timeout rather than hanging forever.
  - `run(command, restart=True)` kills and respawns the subprocess, for when
    a command wedges the shell.
  - **Output is capped** at 1500 chars from the head + 1500 from the tail
    (build logs and compiler errors usually matter most at the very end, so
    a head-only cap loses exactly the useful part). Anything cut is written
    in full to a temp file, and the truncated output tells the model that
    file's path so it can `grep`/`sed`/`tail` it directly instead of us
    building a second "read more output" tool. This matters because
    whatever's in `tool_result` gets resent to the API on *every* later
    turn — an untruncated giant build log doesn't just cost once, it costs
    on every subsequent request for the rest of the run.

**`str_replace_based_edit_tool`** ([`tools/editor.py`](src/agent_harness/tools/editor.py))
— structured file edits instead of Claude writing raw shell/`sed` to touch
files. Four commands, dispatched on the `command` argument:
  - `view` — show a file with line numbers (optionally restricted to a
    `view_range`), or list a directory's contents if `path` is a dir.
  - `create` — write `file_text` to `path`, making parent directories as
    needed.
  - `str_replace` — replace `old_str` with `new_str`, but only if `old_str`
    appears in the file *exactly once*; raises otherwise (ambiguous edit) so
    Claude has to supply enough surrounding context to pin down the match.
  - `insert` — insert `new_str` as a new line after line `insert_line`.

**`submit`** — a plain custom tool (`name`/`description`/`input_schema`, no
special Anthropic type), unlike the other two. Takes one field, `summary`,
and the loop treats seeing this tool call as a hard stop — no further API
calls, whatever `git diff` looks like at that point is final. Without this,
the only way the loop knew to stop was the model happening not to call a
tool on some turn, which let it wander (re-verifying an already-confirmed
fix, writing unsolicited docs) right up to `max_turns`. The system prompt
tells the model to call `submit` as soon as the fix is verified once.

### Everything else

- [`src/agent_harness/swebench/`](src/agent_harness/swebench) — loads
  SWE-bench instances from Hugging Face (`loader.py`) and does a plain
  `git clone` + `git checkout <base_commit>` of the issue's repo
  (`setup_repo.py`) into a working directory for the agent to operate in.

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env   # then fill in ANTHROPIC_API_KEY
```

## Usage

Run the agent on one SWE-bench instance and see the diff it produces. By
default it prints each step live — assistant text, every tool call with its
arguments, and every tool result (truncated if long) — so you can watch what
the agent is actually doing turn by turn (pass `--quiet` to suppress this):

```bash
python scripts/run_instance.py astropy__astropy-12907 --output preds.jsonl
```

The full trajectory (every message the model actually saw — bash output is
still capped per above, with a path to the untruncated log) is also saved as
JSON next to the repo checkout, so you can review it after the fact, e.g. to
see exactly what a run did overnight, or to compare two runs of the same
instance:

```bash
python scripts/show_transcript.py /path/to/<instance_id>.trajectory.json
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
