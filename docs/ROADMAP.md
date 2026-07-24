# Roadmap

Rough order, roughly in order of expected impact on SWE-bench score.

## Environment fidelity (biggest gap right now)
`setup_repo.py` only does `git clone` + `git checkout base_commit`. The agent
has no Python environment with the target repo's dependencies installed, so
it can't actually run the failing test. The official SWE-bench harness solves
this with per-instance docker images. Options:
- Use `swebench`'s own docker images to build/run the agent's container
  instead of just its evaluation step.
- Or hand-roll conda env setup per repo (more work, more understanding).

## Agent loop
- Context management: long trajectories will blow past the context window on
  harder instances. Need truncation/summarization of old tool results.
- Prompt caching (`cache_control` on the system prompt / tool defs) to cut
  cost on long trajectories.
- A "submit"/"done" tool instead of relying on stop_reason != tool_use, so the
  model has an explicit way to signal completion.
- Retry/self-repair: if the produced patch doesn't apply or tests still fail,
  loop back with that feedback instead of stopping.

## Harness mechanics
- Batch runner: iterate over a full split (SWE-bench Lite is 300 instances),
  run with concurrency, write one predictions.jsonl.
- Timeouts and cost caps per instance so a stuck agent doesn't run forever.
- Logging: save full transcripts per instance for later inspection/debugging,
  not just the final diff.

## Eval
- Wire up `scripts/evaluate.py` output parsing to get a pass/fail table and
  score summary instead of reading the raw swebench harness logs.
