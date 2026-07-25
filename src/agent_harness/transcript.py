"""Serialize an agent's message history to disk and back to a readable log.

`Agent.run()` prints steps live (when verbose=True), but the raw
`AgentResult.messages` list also contains Anthropic SDK objects (pydantic
models for assistant turns), so it isn't directly JSON-serializable. This
module converts it to plain dicts for saving, and renders a saved (or live)
transcript as a flat step-by-step log for later review.
"""

from __future__ import annotations

import json
from pathlib import Path


def _to_plain(obj):
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, list):
        return [_to_plain(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    return obj


def to_plain_messages(messages: list[dict]) -> list[dict]:
    return [_to_plain(m) for m in messages]


def save_transcript(messages: list[dict], path: str | Path) -> None:
    Path(path).write_text(json.dumps(to_plain_messages(messages), indent=2))


def render_transcript(messages: list[dict]) -> str:
    """Flatten a transcript (plain dicts) into a readable step-by-step log."""
    lines: list[str] = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            lines.append(f"[{message['role']}] {content}")
            continue
        for block in content:
            block = _to_plain(block)
            btype = block.get("type")
            if btype == "text":
                lines.append(f"[assistant] {block['text']}")
            elif btype == "tool_use":
                lines.append(f"[tool call] {block['name']}({json.dumps(block['input'])})")
            elif btype == "tool_result":
                status = "ERROR" if block.get("is_error") else "ok"
                lines.append(f"[tool result: {status}] {block['content']}")
    return "\n".join(lines)
