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
- Context management: individual bash outputs are now capped (head+tail,
  overflow to a file), but a long trajectory still accumulates many *old*
  tool results that stay in context even once they're stale (e.g. a file
  view superseded by a later edit). Need context editing or summarization of
  old turns, not just per-call output capping.
- Prompt caching (`cache_control` on the system prompt / tool defs) to cut
  cost on long trajectories — the system prompt and tool defs are static
  every turn, so this is close to free.
- ~~A "submit" tool~~ — done: `submit` ends the loop explicitly instead of
  relying on `stop_reason != "tool_use"`.
- Retry/self-repair: if the produced patch doesn't apply or tests still fail,
  loop back with that feedback instead of stopping.

## Harness mechanics
- Batch runner: iterate over a full split (SWE-bench Lite is 300 instances),
  run with concurrency, write one predictions.jsonl.
- Timeouts and cost caps per instance so a stuck agent doesn't run forever.

## Eval
- Wire up `scripts/evaluate.py` output parsing to get a pass/fail table and
  score summary instead of reading the raw swebench harness logs.
