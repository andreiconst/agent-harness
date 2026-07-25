"""A file-editing tool, exposed as Anthropic's `text_editor_20250728` tool.

Supports the four commands the model expects: view, create, str_replace, and
insert. (`undo_edit` was dropped from the 20250728 tool version and is no
longer sent by the model.)
"""

from __future__ import annotations

from pathlib import Path


class EditorTool:
    name = "str_replace_based_edit_tool"
    tool_type = "text_editor_20250728"

    def __init__(self, cwd: str | Path | None = None):
        self._cwd = Path(cwd) if cwd else Path.cwd()

    def run(self, command: str, path: str, **kwargs) -> str:
        p = self._resolve(path)
        if command == "view":
            return self._view(p, kwargs.get("view_range"))
        if command == "create":
            return self._create(p, kwargs["file_text"])
        if command == "str_replace":
            return self._str_replace(p, kwargs["old_str"], kwargs.get("new_str", ""))
        if command == "insert":
            return self._insert(p, kwargs["insert_line"], kwargs["new_str"])
        raise ValueError(f"unknown editor command: {command!r}")

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else self._cwd / p

    def _view(self, p: Path, view_range: list[int] | None) -> str:
        if p.is_dir():
            entries = sorted(
                str(c.relative_to(p)) for c in p.rglob("*") if ".git" not in c.parts
            )
            return "\n".join(entries) or "(empty directory)"

        lines = p.read_text().splitlines()
        start, end = 1, len(lines)
        if view_range:
            start, end = view_range
        numbered = [f"{i + 1}\t{lines[i]}" for i in range(start - 1, min(end, len(lines)))]
        return "\n".join(numbered)

    def _create(self, p: Path, file_text: str) -> str:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(file_text)
        return f"created {p}"

    def _str_replace(self, p: Path, old_str: str, new_str: str) -> str:
        text = p.read_text()
        count = text.count(old_str)
        if count == 0:
            raise ValueError(f"old_str not found in {p}")
        if count > 1:
            raise ValueError(f"old_str is not unique in {p} ({count} occurrences)")
        p.write_text(text.replace(old_str, new_str))
        return f"replaced 1 occurrence in {p}"

    def _insert(self, p: Path, insert_line: int, new_str: str) -> str:
        lines = p.read_text().splitlines(keepends=True)
        if new_str and not new_str.endswith("\n"):
            new_str += "\n"
        lines.insert(insert_line, new_str)
        p.write_text("".join(lines))
        return f"inserted after line {insert_line} in {p}"
