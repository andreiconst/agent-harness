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

Verify your fix once — e.g. by running the relevant test(s) or a small repro
script — and then immediately call the `submit` tool. Do not re-verify
multiple times, gold-plate the fix, or write summary documentation; none of
that improves the submission and it only spends turns and tokens.
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


class Agent:
    def __init__(
        self,
        cwd: str,
        model: str = DEFAULT_MODEL,
        max_turns: int = 40,
        client: Anthropic | None = None,
        verbose: bool = True,
        container: str | None = None,
    ):
        self.cwd = cwd
        self.model = model
        self.max_turns = max_turns
        self.client = client or get_client()
        # `container`, if set, is a running Docker container name — bash
        # commands execute inside it via `docker exec` (see docker_env.py),
        # while the editor still operates on `cwd`, which is expected to be
        # that same container's filesystem bind-mounted onto the host.
        self.bash = BashTool(cwd=cwd, container=container)
        self.editor = EditorTool(cwd=cwd)
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

        for turn in range(1, self.max_turns + 1):
            if self.verbose:
                self._console.rule(f"turn {turn}")

            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=self.tools,
                messages=messages,
            )
            messages.append({"role": "assistant", "content": response.content})
            stop_reason = response.stop_reason

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
