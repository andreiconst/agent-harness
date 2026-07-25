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
    parser.add_argument("--output", default=None, help="path to append the prediction jsonl line to")
    parser.add_argument(
        "--quiet", action="store_true", help="don't print each step as the agent runs"
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
    args = parser.parse_args()

    instance = get_instance(args.instance_id, split=args.split, dataset=args.dataset)

    workdir = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="agent-harness-"))

    docker_env = None
    if args.docker:
        from agent_harness.swebench.docker_env import provision_container

        docker_env = provision_container(instance, workdir)
        repo_path = docker_env.repo_path
    else:
        repo_path = prepare_repo(instance, workdir)

    agent = Agent(
        cwd=str(repo_path),
        max_turns=args.max_turns,
        verbose=not args.quiet,
        container=docker_env.container_name if docker_env else None,
        container_workdir=docker_env.container_workdir if docker_env else "/testbed",
        bash_init_commands=docker_env.bash_init_commands if docker_env else None,
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
    print(f"\n--- ran {result.turns} turn(s), stop_reason={result.stop_reason}, {status} ---")
    print(patch or "(no changes made)")
    print(f"\nrepo checkout: {repo_path}")
    print(f"full trajectory: {transcript_path}")
    print(f"(view it with: python scripts/show_transcript.py {transcript_path})")
    if docker_env and args.keep_container:
        print(f"container kept running: {docker_env.container_name}")

    if args.output:
        prediction = {
            "instance_id": instance["instance_id"],
            "model_patch": patch,
            "model_name_or_path": "agent-harness",
        }
        with open(args.output, "a") as f:
            f.write(json.dumps(prediction) + "\n")
        print(f"appended prediction to {args.output}")


if __name__ == "__main__":
    main()
