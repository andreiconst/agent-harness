"""The core tool-use loop: call Claude, execute any requested tools, repeat."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass

from anthropic import Anthropic
from rich.console import Console

from .llm import DEFAULT_MODEL, get_client
from .tools.bash import BashTool
from .tools.editor import EditorTool

_TRUNCATE_AT = 2000

SUBMIT_TOOL_NAME = "submit"
SUBMIT_TOOL = {
    "name": SUBMIT_TOOL_NAME,
    "description": (
        "Call this as soon as you've made the minimal fix and verified it works "
        "once. This ends the session immediately and submits the repo's current "
        "state as your final answer. Do not call it after only a partial fix, "
        "and do not delay calling it to re-verify multiple times or write "
        "extra documentation — those don't improve the submission."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "One or two sentences on what was changed and why.",
            }
        },
        "required": ["summary"],
    },
}

SYSTEM_PROMPT = """You are a software engineering agent. You have been given a
codebase and an issue to resolve. Use the bash and str_replace_based_edit_tool
tools to explore the repository, reproduce the problem, and make the minimal
changes needed to fix it.

Verify your fix by running the relevant test file with pytest exactly ONCE.
If it passes, call the `submit` tool immediately in that same turn. Do not:
- run the same test file again after it has already passed
- write additional standalone scripts to re-confirm a fix pytest already
  confirmed
- investigate or try to fix pre-existing test failures unrelated to your
  change — only failures your change caused matter
- gold-plate the fix or write summary documentation

Each of those costs turns and tokens without improving the submission.
"""


def _truncate(text: str, limit: int = _TRUNCATE_AT) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    omitted = len(text) - limit
    return f"{text[:half]}\n... [{omitted} more chars] ...\n{text[-half:]}"


@dataclass
class AgentResult:
    messages: list[dict]
    turns: int
    stop_reason: str | None
    submitted: bool = False
    summary: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class Agent:
    def __init__(
        self,
        cwd: str,
        model: str = DEFAULT_MODEL,
        max_turns: int = 40,
        client: Anthropic | None = None,
        verbose: bool = True,
        container: str | None = None,
        container_workdir: str = "/testbed",
        bash_init_commands: list[str] | None = None,
    ):
        self.cwd = cwd
        self.model = model
        self.max_turns = max_turns
        self.client = client or get_client()
        # `container`, if set, is a running Docker container name — bash
        # commands execute inside it via `docker exec` (see docker_env.py).
        # `cwd` is expected to be that same container's filesystem
        # bind-mounted onto the host at `container_workdir`: the editor
        # operates on the host path directly, but also needs
        # `container_workdir` so it can remap the container-absolute paths
        # the model naturally uses (since it explored via `bash`, which runs
        # *inside* the container) back onto that host directory.
        self.bash = BashTool(
            cwd=cwd,
            container=container,
            container_workdir=container_workdir,
            init_commands=bash_init_commands,
        )
        self.editor = EditorTool(cwd=cwd, container_workdir=container_workdir if container else None)
        self.verbose = verbose
        self._console = Console() if verbose else None

    @property
    def tools(self) -> list[dict]:
        return [
            {"type": self.bash.tool_type, "name": self.bash.name},
            {"type": self.editor.tool_type, "name": self.editor.name},
            SUBMIT_TOOL,
        ]

    def run(self, task: str) -> AgentResult:
        messages: list[dict] = [{"role": "user", "content": task}]
        stop_reason = None
        turn = 0
        summary: str | None = None
        usage_totals = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }

        for turn in range(1, self.max_turns + 1):
            if self.verbose:
                self._console.rule(f"turn {turn}")

            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=self.tools,
                messages=messages,
                # Caches everything up through the last message block — the
                # system prompt, tool defs, and the whole growing history are
                # byte-identical turn over turn (we only ever append), so this
                # is a straightforward prefix-cache win with no manual
                # bookkeeping: subsequent turns re-read the unchanged prefix
                # at ~10% of normal input cost instead of reprocessing it.
                cache_control={"type": "ephemeral"},
            )
            messages.append({"role": "assistant", "content": response.content})
            stop_reason = response.stop_reason

            for key in usage_totals:
                usage_totals[key] += getattr(response.usage, key, None) or 0
            if self.verbose:
                u = response.usage
                self._console.print(
                    f"[dim]usage: input={u.input_tokens} "
                    f"cache_read={u.cache_read_input_tokens or 0} "
                    f"cache_creation={u.cache_creation_input_tokens or 0} "
                    f"output={u.output_tokens}[/]"
                )

            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            if self.verbose:
                for block in response.content:
                    if block.type == "text" and block.text.strip():
                        self._console.print(f"[bold cyan]assistant:[/] {block.text.strip()}")
                    elif block.type == "tool_use":
                        args = _truncate(json.dumps(block.input))
                        self._console.print(f"[bold yellow]tool call:[/] {block.name}({args})")

            if response.stop_reason != "tool_use":
                break

            tool_results = []
            for block in tool_use_blocks:
                if block.name == SUBMIT_TOOL_NAME:
                    summary = block.input.get("summary", "")
                    tool_results.append(
                        {"type": "tool_result", "tool_use_id": block.id, "content": "submitted"}
                    )
                    if self.verbose:
                        self._console.print(f"[bold magenta]submitted:[/] {summary}")
                    continue
                result = self._execute_tool(block)
                if self.verbose:
                    style = "bold red" if result.get("is_error") else "green"
                    self._console.print(f"[{style}]tool result:[/] {_truncate(result['content'])}")
                tool_results.append(result)
            messages.append({"role": "user", "content": tool_results})

            if summary is not None:
                break

        self.bash.stop()
        return AgentResult(
            messages=messages,
            turns=turn,
            stop_reason=stop_reason,
            submitted=summary is not None,
            summary=summary,
            **usage_totals,
        )

    def _execute_tool(self, block) -> dict:
        try:
            if block.name == self.bash.name:
                output = self.bash.run(**block.input)
            elif block.name == self.editor.name:
                output = self.editor.run(**block.input)
            else:
                return {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": f"unknown tool: {block.name}",
                    "is_error": True,
                }
            return {"type": "tool_result", "tool_use_id": block.id, "content": output}
        except Exception as exc:
            return {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": str(exc),
                "is_error": True,
            }

    def diff(self) -> str:
        """Return the git diff of everything the agent changed in `cwd`."""
        result = subprocess.run(
            ["git", "diff"], cwd=self.cwd, capture_output=True, text=True
        )
        return result.stdout
