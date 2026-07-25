#!/usr/bin/env python3
"""Pretty-print a trajectory.json saved by run_instance.py.

Usage:
    python scripts/show_transcript.py /path/to/trajectory.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent_harness.transcript import render_transcript


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transcript_path")
    args = parser.parse_args()

    messages = json.loads(Path(args.transcript_path).read_text())
    print(render_transcript(messages))


if __name__ == "__main__":
    main()
