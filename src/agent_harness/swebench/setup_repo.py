"""Check out a SWE-bench instance's repo at its base commit.

Note: this only gets you the source tree. The official SWE-bench harness
additionally builds a docker image per instance with the right language
runtime and dependencies installed so tests actually run; see
scripts/evaluate.py and docs/ROADMAP.md.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def sanitize_git_history(repo_path: Path) -> None:
    """Drop every ref that isn't an ancestor of HEAD, then gc-prune the rest away.

    A full clone (or the SWE-bench team's prebuilt image, which is built from
    one) still contains every commit *after* the instance's base commit —
    including the real fix, fully reachable via `git log --all` / `git show
    <sha>` with no network involved. Only non-ancestor refs are removed
    (rather than all of them) because some repos resolve `__version__` from
    the nearest reachable tag at import time (setuptools_scm and similar) —
    deleting a past release tag that's a genuine ancestor would break that
    for reasons unrelated to the leak this is closing.
    """
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_path, check=True, capture_output=True, text=True
    ).stdout.strip()
    subprocess.run(["git", "checkout", "--detach", sha], cwd=repo_path, check=True)

    refs = subprocess.run(
        ["git", "for-each-ref", "--format=%(refname)"],
        cwd=repo_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    for ref in refs:
        is_ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", ref, sha], cwd=repo_path
        )
        if is_ancestor.returncode != 0:
            subprocess.run(["git", "update-ref", "-d", ref], cwd=repo_path, check=True)

    subprocess.run(["git", "remote", "remove", "origin"], cwd=repo_path, capture_output=True)
    subprocess.run(["git", "reflog", "expire", "--expire=now", "--all"], cwd=repo_path, check=True)
    subprocess.run(["git", "gc", "--prune=now", "--aggressive"], cwd=repo_path, check=True)


def prepare_repo(instance: dict, workdir: Path) -> Path:
    repo_url = f"https://github.com/{instance['repo']}.git"
    repo_path = Path(workdir) / instance["instance_id"]

    if not repo_path.exists():
        subprocess.run(["git", "clone", repo_url, str(repo_path)], check=True)

    subprocess.run(["git", "checkout", instance["base_commit"]], cwd=repo_path, check=True)
    sanitize_git_history(repo_path)
    return repo_path
