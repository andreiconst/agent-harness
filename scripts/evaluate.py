#!/usr/bin/env python3
"""Thin wrapper around the official SWE-bench docker-based evaluation harness.

Requires Docker running locally and the optional `eval` dependency group
(`uv pip install -e ".[eval]"`).

Usage:
    python scripts/evaluate.py \\
        --predictions_path preds.jsonl \\
        --dataset_name princeton-nlp/SWE-bench_Lite \\
        --run_id my_run

See `python -m swebench.harness.run_evaluation --help` for all options.
"""

from __future__ import annotations

import subprocess
import sys


def main() -> None:
    cmd = [sys.executable, "-m", "swebench.harness.run_evaluation", *sys.argv[1:]]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
