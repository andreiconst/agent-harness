"""A persistent bash session, exposed as Anthropic's `bash_20250124` tool.

Persistence matters for a coding agent: `cd`, exported env vars, and activated
virtualenvs need to survive between tool calls, so each call re-uses the same
underlying `/bin/bash` process instead of spawning a fresh `subprocess.run`.
"""

from __future__ import annotations

import queue
import subprocess
import threading
import time
import uuid


class BashTool:
    name = "bash"
    tool_type = "bash_20250124"

    def __init__(self, cwd: str | None = None, timeout: float = 60.0):
        self._cwd = cwd
        self._timeout = timeout
        self._proc: subprocess.Popen | None = None
        self._output_queue: queue.Queue[str | None] = queue.Queue()

    def start(self) -> None:
        self._proc = subprocess.Popen(
            ["/bin/bash"],
            cwd=self._cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
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

        output = "".join(lines)
        if exit_code is not None:
            output += f"\n[exit code: {exit_code}]"
        return output
