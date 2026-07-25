# Roadmap

Rough order, roughly in order of expected impact on SWE-bench score.

## Environment fidelity
~~`setup_repo.py` only does `git clone`~~ — done: `docker_env.py` +
`run_instance.py --docker` gives the agent the real per-instance SWE-bench
environment (pulls the official prebuilt image, bind-mounts it to a host
dir). Verified end-to-end against astropy — including two real gotchas that
are now fixed: conda env activation (`BashTool.init_commands`) and the
editor tool's container-vs-host path mismatch (`EditorTool.container_workdir`).
The plain `git clone` path (`setup_repo.py`) stays as the default for
fast/no-Docker iteration.

## Agent loop
- Context management: individual bash outputs are now capped (head+tail,
  overflow to a file), but a long trajectory still accumulates many *old*
  tool results that stay in context even once they're stale (e.g. a file
  view superseded by a later edit). Prompt caching (below) makes the *cost*
  of this cheaper but doesn't shrink the actual context — still need context
  editing or summarization of old turns for that.
- ~~Prompt caching~~ — done: `cache_control: {"type": "ephemeral"}` on every
  request; `AgentResult`/`run_instance.py` report per-turn and total
  input/output/cache_read/cache_creation tokens plus a cache-hit-rate summary.
- ~~A "submit" tool~~ — done: `submit` ends the loop explicitly instead of
  relying on `stop_reason != "tool_use"`.
- ~~Redundant re-verification~~ — done (prompt-level): once `--docker` gave
  the agent a working test suite, it started re-running the same passing
  `pytest` file repeatedly and writing redundant standalone confirmation
  scripts. System prompt now says: verify with pytest exactly once, submit
  immediately if it passes, don't chase pre-existing unrelated failures.
  Worth revisiting if it still happens — a stronger lever would be rejecting
  a second identical tool call outright rather than just asking nicely.
- Retry/self-repair: if the produced patch doesn't apply or tests still fail,
  loop back with that feedback instead of stopping.

## Harness mechanics
- Batch runner: iterate over a full split (SWE-bench Lite is 300 instances),
  run with concurrency, write one predictions.jsonl.
- Timeouts and cost caps per instance so a stuck agent doesn't run forever.

## Eval
- Wire up `scripts/evaluate.py` output parsing to get a pass/fail table and
  score summary instead of reading the raw swebench harness logs.
