"""A persistent bash session, exposed as Anthropic's `bash_20250124` tool.

Persistence matters for a coding agent: `cd`, exported env vars, and activated
virtualenvs need to survive between tool calls, so each call re-uses the same
underlying `/bin/bash` process instead of spawning a fresh `subprocess.run`.
"""

from __future__ import annotations

import queue
import subprocess
import tempfile
import threading
import time
import uuid


class BashTool:
    name = "bash"
    tool_type = "bash_20250124"

    # Command output can be huge (build logs, verbose test runs) and gets
    # resent to the model on every subsequent turn, so it's capped hard. The
    # interesting part of a failure is often at the very end (the last error
    # line), so we keep both ends rather than just the head.
    _HEAD_CHARS = 1500
    _TAIL_CHARS = 1500

    def __init__(
        self,
        cwd: str | None = None,
        timeout: float = 60.0,
        container: str | None = None,
        container_workdir: str = "/testbed",
    ):
        self._cwd = cwd
        self._timeout = timeout
        # When `container` is set, commands run via `docker exec` into that
        # (already-running) container instead of a local /bin/bash — used for
        # SWE-bench instances, where the repo's real environment only exists
        # inside the container. `container_workdir` matches the SWE-bench
        # harness convention of checking the repo out at /testbed.
        self._container = container
        self._container_workdir = container_workdir
        self._proc: subprocess.Popen | None = None
        self._output_queue: queue.Queue[str | None] = queue.Queue()

    def start(self) -> None:
        if self._container:
            argv = ["docker", "exec", "-i", "-w", self._container_workdir, self._container, "/bin/bash"]
            popen_kwargs = {}
        else:
            argv = ["/bin/bash"]
            popen_kwargs = {"cwd": self._cwd}

        self._proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            **popen_kwargs,
        )
        threading.Thread(target=self._pump_output, daemon=True).start()

    def _pump_output(self) -> None:
        assert self._proc and self._proc.stdout
        for line in self._proc.stdout:
            self._output_queue.put(line)
        self._output_queue.put(None)  # signals EOF to any in-flight run()

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
        self._proc = None

    def _truncate_output(self, output: str) -> str:
        limit = self._HEAD_CHARS + self._TAIL_CHARS
        if len(output) <= limit:
            return output

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".log", prefix="agent-harness-bash-", delete=False
        ) as f:
            f.write(output)
            log_path = f.name

        omitted = len(output) - limit
        return (
            output[: self._HEAD_CHARS]
            + f"\n\n... [{omitted} chars omitted — full output saved to {log_path}; "
            + "use bash (e.g. sed -n, grep, tail) to inspect it if needed] ...\n\n"
            + output[-self._TAIL_CHARS :]
        )

    def run(self, command: str, restart: bool = False) -> str:
        if restart:
            self.stop()
            self.start()
            return "bash session restarted"

        if self._proc is None or self._proc.poll() is not None:
            self.start()
        assert self._proc and self._proc.stdin

        sentinel = f"__DONE_{uuid.uuid4().hex}__"
        self._proc.stdin.write(f"{command}\necho '{sentinel}' $?\n")
        self._proc.stdin.flush()

        lines: list[str] = []
        exit_code: str | None = None
        deadline = time.time() + self._timeout
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                lines.append(f"[bash tool] command timed out after {self._timeout}s\n")
                break
            try:
                line = self._output_queue.get(timeout=remaining)
            except queue.Empty:
                lines.append(f"[bash tool] command timed out after {self._timeout}s\n")
                break
            if line is None:
                break
            if sentinel in line:
                exit_code = line.strip().split()[-1]
                break
            lines.append(line)

        output = self._truncate_output("".join(lines))
        if exit_code is not None:
            output += f"\n[exit code: {exit_code}]"
        return output
