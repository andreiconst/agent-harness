"""The core tool-use loop: call Claude, execute any requested tools, repeat."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

from anthropic import Anthropic

from .llm import DEFAULT_MODEL, get_client
from .tools.bash import BashTool
from .tools.editor import EditorTool

SYSTEM_PROMPT = """You are a software engineering agent. You have been given a
codebase and an issue to resolve. Use the bash and str_replace_editor tools to
explore the repository, reproduce the problem, make the minimal changes
needed to fix it, and verify your fix (e.g. by running relevant tests).

When you are confident the issue is resolved, stop calling tools and reply
with a short summary of what you changed and why.
"""


@dataclass
class AgentResult:
    messages: list[dict]
    turns: int
    stop_reason: str | None


class Agent:
    def __init__(
        self,
        cwd: str,
        model: str = DEFAULT_MODEL,
        max_turns: int = 40,
        client: Anthropic | None = None,
    ):
        self.cwd = cwd
        self.model = model
        self.max_turns = max_turns
        self.client = client or get_client()
        self.bash = BashTool(cwd=cwd)
        self.editor = EditorTool(cwd=cwd)

    @property
    def tools(self) -> list[dict]:
        return [
            {"type": self.bash.tool_type, "name": self.bash.name},
            {"type": self.editor.tool_type, "name": self.editor.name},
        ]

    def run(self, task: str) -> AgentResult:
        messages: list[dict] = [{"role": "user", "content": task}]
        stop_reason = None
        turn = 0

        for turn in range(1, self.max_turns + 1):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=self.tools,
                messages=messages,
            )
            messages.append({"role": "assistant", "content": response.content})
            stop_reason = response.stop_reason

            if response.stop_reason != "tool_use":
                break

            tool_results = [
                self._execute_tool(block)
                for block in response.content
                if block.type == "tool_use"
            ]
            messages.append({"role": "user", "content": tool_results})

        self.bash.stop()
        return AgentResult(messages=messages, turns=turn, stop_reason=stop_reason)

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
