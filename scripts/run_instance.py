#!/usr/bin/env python3
"""Run the agent on a single SWE-bench instance and print/save the resulting diff.

Example:
    python scripts/run_instance.py astropy__astropy-12907 --output preds.jsonl
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from agent_harness.agent import Agent
from agent_harness.llm import DEFAULT_MODEL
from agent_harness.swebench.loader import get_instance
from agent_harness.swebench.setup_repo import prepare_repo
from agent_harness.transcript import save_transcript


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance_id", help="e.g. astropy__astropy-12907")
    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Lite")
    parser.add_argument("--split", default="test")
    parser.add_argument("--workdir", default=None, help="defaults to a fresh temp dir")
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"model id to run the agent on (default: {DEFAULT_MODEL})",
    )
    parser.add_argument("--output", default=None, help="path to append the prediction jsonl line to")
    parser.add_argument(
        "--quiet", action="store_true", help="don't print each step as the agent runs"
    )
    parser.add_argument(
        "--log",
        default=None,
        help=(
            "also write the full run to this file — untruncated tool calls and "
            "results, unlike the terminal. Works with --quiet, and is written as "
            "the run goes, so a crashed run still leaves everything up to the crash"
        ),
    )
    parser.add_argument(
        "--docker",
        action="store_true",
        help=(
            "run bash commands inside the instance's real SWE-bench Docker "
            "environment instead of a plain git checkout (needs Docker running; "
            "pulls the official prebuilt image on first use per repo)"
        ),
    )
    parser.add_argument(
        "--keep-container",
        action="store_true",
        help="with --docker, don't remove the container when done (for debugging)",
    )
    parser.add_argument(
        "--keep-workdir",
        action="store_true",
        help=(
            "don't delete the repo checkout when done (for debugging). Off by "
            "default: every run extracts a fresh copy of the repo into a new "
            "temp dir that otherwise never gets cleaned up, which adds up fast "
            "across repeated runs"
        ),
    )
    args = parser.parse_args()

    # Line-buffered so the log survives a crash mid-run.
    log_file = open(args.log, "w", buffering=1) if args.log else None

    def report(line: str = "") -> None:
        """Print, and mirror into the same log the agent is writing to."""
        print(line)
        if log_file:
            log_file.write(line + "\n")

    instance = get_instance(args.instance_id, split=args.split, dataset=args.dataset)

    workdir = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="agent-harness-"))

    docker_env = None
    if args.docker:
        from agent_harness.swebench.docker_env import provision_container

        docker_env = provision_container(instance, workdir)
        repo_path = docker_env.repo_path
    else:
        repo_path = prepare_repo(instance, workdir)
        # `prepare_repo` only checks out source — it installs nothing. Without
        # the instance's real environment the agent cannot run the test suite
        # at all, and burns its whole budget on pip and conftest archaeology
        # instead of on the bug. That failure is invisible in the transcript
        # until you read 20 turns of it, so say it up front.
        report(
            "WARNING: running without --docker. The repo is a bare checkout with no "
            "dependencies installed, so the test suite will almost certainly not run "
            "and the agent will spend its turns trying to build an environment. Use "
            "--docker for a meaningful run."
        )

    agent = Agent(
        cwd=str(repo_path),
        model=args.model,
        max_turns=args.max_turns,
        verbose=not args.quiet,
        container=docker_env.container_name if docker_env else None,
        container_workdir=docker_env.container_workdir if docker_env else "/testbed",
        bash_init_commands=docker_env.bash_init_commands if docker_env else None,
        log_file=log_file,
    )
    task = f"{instance['problem_statement']}\n\nRepository: {instance['repo']}"
    result = agent.run(task)
    patch = agent.diff()

    if docker_env and not args.keep_container:
        docker_env.cleanup()

    # Saved next to (not inside) the repo checkout, so it doesn't show up in `git diff`.
    transcript_path = workdir / f"{instance['instance_id']}.trajectory.json"
    save_transcript(result.messages, transcript_path)

    status = f"submitted: {result.summary}" if result.submitted else "did not call submit"
    report(
        f"\n--- {args.model}: ran {result.turns} turn(s), "
        f"stop_reason={result.stop_reason}, {status} ---"
    )
    if result.verified_turn is not None:
        tail = result.turns - result.verified_turn
        report(f"verified green on turn {result.verified_turn}; {tail} turn(s) spent after that")
    if agent.diff_was_restored:
        report(
            "WARNING: the working tree was empty at the end of the run — submitting the "
            "last non-empty diff instead. The agent likely left a `git stash`/`checkout`/"
            "`reset` unreverted."
        )
    report(patch or "(no changes made)")
    report(
        f"\nusage: input={result.input_tokens} "
        f"cache_read={result.cache_read_input_tokens} "
        f"cache_creation={result.cache_creation_input_tokens} "
        f"output={result.output_tokens}"
    )
    total_input = result.input_tokens + result.cache_read_input_tokens + result.cache_creation_input_tokens
    if total_input:
        report(f"cache hit rate: {result.cache_read_input_tokens / total_input:.0%} of input tokens served from cache")
    if args.keep_workdir:
        report(f"\nrepo checkout: {repo_path}")
    else:
        import shutil

        shutil.rmtree(repo_path, ignore_errors=True)
        report(f"\nrepo checkout: {repo_path} (deleted — pass --keep-workdir to keep it)")
    report(f"full trajectory: {transcript_path}")
    report(f"(view it with: python scripts/show_transcript.py {transcript_path})")
    if docker_env and args.keep_container:
        report(f"container kept running: {docker_env.container_name}")

    if args.output:
        prediction = {
            "instance_id": instance["instance_id"],
            "model_patch": patch,
            "model_name_or_path": "agent-harness",
        }
        with open(args.output, "a") as f:
            f.write(json.dumps(prediction) + "\n")
        report(f"appended prediction to {args.output}")

    if log_file:
        log_file.close()
        print(f"full run log: {args.log}")


if __name__ == "__main__":
    main()
