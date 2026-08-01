"""Provision a per-instance Docker environment for a SWE-bench instance.

`setup_repo.prepare_repo` only gets you a `git clone` + `git checkout` — no
Python environment, no compiled dependencies. Real repos (astropy, numpy,
lxml, ...) need the exact pinned environment the SWE-bench authors built, or
the agent burns turns fighting `pip: command not found`, wrong Python
versions, and C-extension build failures against a mismatched toolchain.

This reuses the official `swebench` harness's own image logic — with
`namespace="swebench"` (the same default `run_evaluation.py` itself uses) it
pulls the SWE-bench team's prebuilt image instead of building one from
scratch locally, sidestepping local compiler/toolchain mismatches entirely.

Strategy: build/pull the instance image, extract its `/testbed` to a host
directory via a throwaway container, then start a long-lived container with
that host directory bind-mounted back in at the same path. That way
`EditorTool` keeps doing plain host filesystem I/O unchanged, and `BashTool`
just needs to run commands via `docker exec` into the container instead of a
local shell.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import docker
from swebench.harness.constants import DOCKER_USER, DOCKER_WORKDIR
from swebench.harness.docker_build import build_container, setup_logger
from swebench.harness.test_spec.test_spec import make_test_spec

from .setup_repo import sanitize_git_history

RUN_ID = "agent-harness"
NAMESPACE = "swebench"  # matches run_evaluation.py's own default


@dataclass
class DockerEnv:
    container_name: str
    repo_path: Path
    container_workdir: str
    bash_init_commands: list[str]
    client: docker.DockerClient

    def cleanup(self) -> None:
        try:
            container = self.client.containers.get(self.container_name)
            container.remove(force=True)
        except docker.errors.NotFound:
            pass


def _bash_init_commands(test_spec) -> list[str]:
    """Extract the "activate this instance's environment" lines.

    `env_script_list` mixes one-time build steps (creating the conda env,
    pip-installing deps — already baked into the image) with the couple of
    lines needed on every invocation to activate that env. The real SWE-bench
    eval script re-runs exactly these same lines before every test run, so we
    do the same rather than hardcoding a path/env-manager assumption that
    might not hold for every repo.
    """
    return [
        line
        for line in test_spec.env_script_list
        if line.startswith("source ") or line.startswith("conda activate")
    ]


def _remove_if_exists(client: docker.DockerClient, name: str) -> None:
    """Drop a leftover container by name, if one is there."""
    try:
        client.containers.get(name).remove(force=True)
    except docker.errors.NotFound:
        pass


def provision_container(instance: dict, workdir: Path) -> DockerEnv:
    client = docker.from_env()
    test_spec = make_test_spec(instance, namespace=NAMESPACE)

    log_path = workdir / f"{instance['instance_id']}.docker-build.log"
    logger = setup_logger(instance["instance_id"], log_path, add_stdout=False)

    # A previous run that died between creating the extractor and removing it
    # (Ctrl-C, a crash, a failed `docker cp`) leaves the name taken, and
    # `build_container` then fails with a 409 name conflict that looks like an
    # image-build error. Clear it first, the same way the long-lived container
    # below is cleared.
    _remove_if_exists(client, test_spec.get_instance_container_name(RUN_ID))

    # Builds (or, since namespace is set, pulls) the instance image and
    # creates — but does not start — a container from it. We only need this
    # container to extract /testbed; `docker cp` works on stopped containers.
    extractor = build_container(test_spec, client, RUN_ID, logger, nocache=False)

    repo_path = workdir / instance["instance_id"]
    repo_path.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["docker", "cp", f"{extractor.name}:{DOCKER_WORKDIR}/.", str(repo_path)],
            check=True,
        )
    finally:
        # Even if the copy failed: leaving it behind only guarantees the next
        # run hits the name conflict this function just worked around.
        extractor.remove(force=True)
    sanitize_git_history(repo_path)

    container_name = f"agent-harness-{instance['instance_id']}"
    _remove_if_exists(client, container_name)

    cap_add = test_spec.docker_specs.get("run_args", {}).get("cap_add", [])
    container = client.containers.run(
        image=test_spec.instance_image_key,
        name=container_name,
        user=DOCKER_USER,
        detach=True,
        command="tail -f /dev/null",
        platform=test_spec.platform,
        cap_add=cap_add,
        volumes={str(repo_path.resolve()): {"bind": DOCKER_WORKDIR, "mode": "rw"}},
        # No network: an agent with internet access can look up the real fix
        # commit (GitHub search/commit APIs) and replay it instead of solving
        # the issue, which is exactly what happened here on a rerun. The
        # official SWE-bench eval harness runs test containers offline for
        # the same reason.
        network_mode="none",
    )
    return DockerEnv(
        container_name=container.name,
        repo_path=repo_path,
        container_workdir=DOCKER_WORKDIR,
        bash_init_commands=_bash_init_commands(test_spec),
        client=client,
    )
