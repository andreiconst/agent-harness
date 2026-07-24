"""Check out a SWE-bench instance's repo at its base commit.

Note: this only gets you the source tree. The official SWE-bench harness
additionally builds a docker image per instance with the right language
runtime and dependencies installed so tests actually run; see
scripts/evaluate.py and docs/ROADMAP.md.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def prepare_repo(instance: dict, workdir: Path) -> Path:
    repo_url = f"https://github.com/{instance['repo']}.git"
    repo_path = Path(workdir) / instance["instance_id"]

    if not repo_path.exists():
        subprocess.run(["git", "clone", repo_url, str(repo_path)], check=True)

    subprocess.run(["git", "checkout", instance["base_commit"]], cwd=repo_path, check=True)
    return repo_path
