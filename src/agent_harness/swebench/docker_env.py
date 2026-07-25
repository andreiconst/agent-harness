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

RUN_ID = "agent-harness"
NAMESPACE = "swebench"  # matches run_evaluation.py's own default


@dataclass
class DockerEnv:
    container_name: str
    repo_path: Path
    client: docker.DockerClient

    def cleanup(self) -> None:
        try:
            container = self.client.containers.get(self.container_name)
            container.remove(force=True)
        except docker.errors.NotFound:
            pass


def provision_container(instance: dict, workdir: Path) -> DockerEnv:
    client = docker.from_env()
    test_spec = make_test_spec(instance, namespace=NAMESPACE)

    log_path = workdir / f"{instance['instance_id']}.docker-build.log"
    logger = setup_logger(instance["instance_id"], log_path, add_stdout=False)

    # Builds (or, since namespace is set, pulls) the instance image and
    # creates — but does not start — a container from it. We only need this
    # container to extract /testbed; `docker cp` works on stopped containers.
    extractor = build_container(test_spec, client, RUN_ID, logger, nocache=False)

    repo_path = workdir / instance["instance_id"]
    repo_path.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["docker", "cp", f"{extractor.name}:{DOCKER_WORKDIR}/.", str(repo_path)],
        check=True,
    )
    extractor.remove(force=True)

    container_name = f"agent-harness-{instance['instance_id']}"
    try:
        client.containers.get(container_name).remove(force=True)
    except docker.errors.NotFound:
        pass

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
    )
    return DockerEnv(container_name=container.name, repo_path=repo_path, client=client)
