"""A file-editing tool, exposed as Anthropic's `text_editor_20250728` tool.

Supports the four commands the model expects: view, create, str_replace, and
insert. (`undo_edit` was dropped from the 20250728 tool version and is no
longer sent by the model.)
"""

from __future__ import annotations

from pathlib import Path

_MAX_VIEW_CHARS = 4000


def _on(turn: int | None) -> str:
    return f" on turn {turn}" if turn is not None else ""


class EditorTool:
    name = "str_replace_based_edit_tool"
    tool_type = "text_editor_20250728"

    def __init__(self, cwd: str | Path | None = None, container_workdir: str | None = None):
        self._cwd = (Path(cwd) if cwd else Path.cwd()).resolve()
        # When set, an absolute path starting with this prefix is remapped
        # onto `cwd` instead of being looked up literally. This matters when
        # `cwd` is a host directory bind-mounted into a container at
        # `container_workdir` (see docker_env.py): the model explores via
        # `bash`, which runs *inside* the container and sees paths like
        # `/testbed/foo.py`, so it naturally passes that same absolute path
        # to this tool too — but this tool operates on the host filesystem,
        # where `/testbed` doesn't exist. Without the remap, every absolute
        # path the model has actually seen fails with "No such file".
        self._container_workdir = container_workdir.rstrip("/") if container_workdir else None
        # Line ranges of each file already shown, and the turn that showed
        # them. Views are the largest thing in an agent's context — one
        # debugging run re-read a single file ten times, 23% of its whole
        # history — and every copy is permanent, since the transcript is
        # append-only. Crucially those ten reads were *overlapping*, not
        # identical, so this tracks coverage rather than exact repeats. See
        # `_view`.
        self._view_coverage: dict[str, list[tuple[int, int, int | None]]] = {}
        # Directory listings have no line ranges; dedupe those on content.
        self._dir_views: dict[str, tuple[str, int | None]] = {}
        # Set by the agent loop so a stub can point at the earlier turn.
        self.current_turn: int | None = None

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
        if self._container_workdir and (path == self._container_workdir or path.startswith(self._container_workdir + "/")):
            relative = path[len(self._container_workdir) :].lstrip("/")
            p = self._cwd / relative if relative else self._cwd
        else:
            raw = Path(path)
            p = raw if raw.is_absolute() else self._cwd / raw

        # Every operation is jailed to `cwd`, in every mode — not just
        # container mode. Without this, any absolute path (or a relative
        # path with enough `..`) is followed literally against the real
        # host filesystem: a model passing "/" gets the real disk root
        # `rglob`'d, "/etc/passwd" gets read straight off the host, etc.
        resolved = p.resolve()
        if resolved != self._cwd and self._cwd not in resolved.parents:
            raise ValueError(f"path {path!r} is outside the working directory ({self._cwd})")
        return resolved

    def _truncate(self, text: str) -> str:
        if len(text) <= _MAX_VIEW_CHARS:
            return text
        omitted = len(text) - _MAX_VIEW_CHARS
        return (
            text[:_MAX_VIEW_CHARS]
            + f"\n... [{omitted} more chars omitted — use view_range for a file, "
            + "or view a narrower subdirectory] ..."
        )

    def _render_lines(self, lines: list[str], start: int, end: int) -> tuple[str, int]:
        """Numbered text for [start, end], plus the last line it actually shows.

        Those differ when the body gets truncated: the caller must not record
        coverage for lines the model never received.
        """
        numbered = [f"{i + 1}\t{lines[i]}" for i in range(start - 1, end)]
        text = "\n".join(numbered)
        truncated = self._truncate(text)
        if truncated == text:
            return text, end
        # The cut lands mid-line; only whole lines before it were delivered.
        whole = len(truncated[:_MAX_VIEW_CHARS].split("\n")) - 1
        return truncated, start + max(whole, 0) - 1

    def _view(self, p: Path, view_range: list[int] | None) -> str:
        """Render the file, or a stub if these lines are already in context.

        Tracks which line ranges have been shown rather than which calls were
        repeated, because agents re-read overlapping windows around the code
        they're working on, not identical ones. Any edit to a file drops its
        coverage — line numbers shift and the old text is stale — so a re-read
        after a change always returns real content, which is the case that
        matters.
        """
        if p.is_dir():
            entries = sorted(
                str(c.relative_to(p)) for c in p.rglob("*") if ".git" not in c.parts
            )
            body = self._truncate("\n".join(entries) or "(empty directory)")
            previous, turn = self._dir_views.get(str(p), (None, None))
            if previous == body:
                return f"(identical to the listing you already have{_on(turn)}.)"
            self._dir_views[str(p)] = (body, self.current_turn)
            return body

        lines = p.read_text().splitlines()
        start = view_range[0] if view_range else 1
        end = view_range[1] if view_range else len(lines)
        if end in (-1, None) or end > len(lines):
            end = len(lines)
        start = max(start, 1)
        if start > end:
            return f"(no lines in range: {p} has {len(lines)} lines)"

        for shown_start, shown_end, turn in self._view_coverage.get(str(p), []):
            if shown_start <= start and end <= shown_end:
                return (
                    f"(lines {start}-{end} are already in your context: you viewed "
                    f"lines {shown_start}-{shown_end} of this file{_on(turn)} and "
                    "nothing has modified it since. Scroll up rather than re-reading.)"
                )

        body, covered_end = self._render_lines(lines, start, end)
        if covered_end >= start:
            self._view_coverage.setdefault(str(p), []).append(
                (start, covered_end, self.current_turn)
            )
        return body

    def _invalidate(self, p: Path) -> None:
        """Forget what's been shown of a file once its line numbers can shift."""
        self._view_coverage.pop(str(p), None)

    def _create(self, p: Path, file_text: str) -> str:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(file_text)
        self._invalidate(p)
        return f"created {p}"

    def _str_replace(self, p: Path, old_str: str, new_str: str) -> str:
        text = p.read_text()
        count = text.count(old_str)
        if count == 0:
            raise ValueError(f"old_str not found in {p}")
        if count > 1:
            raise ValueError(f"old_str is not unique in {p} ({count} occurrences)")
        p.write_text(text.replace(old_str, new_str))
        self._invalidate(p)
        return f"replaced 1 occurrence in {p}"

    def _insert(self, p: Path, insert_line: int, new_str: str) -> str:
        lines = p.read_text().splitlines(keepends=True)
        if new_str and not new_str.endswith("\n"):
            new_str += "\n"
        lines.insert(insert_line, new_str)
        p.write_text("".join(lines))
        self._invalidate(p)
        return f"inserted after line {insert_line} in {p}"
