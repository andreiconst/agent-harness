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


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance_id", help="e.g. astropy__astropy-12907")
    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Lite")
    parser.add_argument("--split", default="test")
    parser.add_argument("--workdir", default=None, help="defaults to a fresh temp dir")
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--output", default=None, help="path to append the prediction jsonl line to")
    args = parser.parse_args()

    instance = get_instance(args.instance_id, split=args.split, dataset=args.dataset)

    workdir = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="agent-harness-"))
    repo_path = prepare_repo(instance, workdir)

    agent = Agent(cwd=str(repo_path), max_turns=args.max_turns)
    task = f"{instance['problem_statement']}\n\nRepository: {instance['repo']}"
    result = agent.run(task)
    patch = agent.diff()

    print(f"--- ran {result.turns} turn(s), stop_reason={result.stop_reason} ---")
    print(patch or "(no changes made)")

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
