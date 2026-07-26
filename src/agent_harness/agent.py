"""The core tool-use loop: call Claude, execute any requested tools, repeat."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass

from anthropic import Anthropic
from rich.console import Console
from rich.markup import escape

from .llm import DEFAULT_MODEL, get_client
from .tools.bash import BashTool
from .tools.editor import EditorTool

_TRUNCATE_AT = 2000

# Per-response output cap. On models with adaptive thinking (Sonnet 5, Opus 5,
# …) this budget covers thinking *and* the response text together, so a value
# tuned for a non-thinking model truncates mid-turn. 16K is the largest that
# comfortably stays under the SDK's non-streaming HTTP timeout; turns here
# average a few hundred output tokens, so the cap is headroom, not a target.
_MAX_TOKENS = 16000

# Positive evidence that pytest ran and everything it collected passed. The
# summary line lives at the very end of the output, which is why the bash tool
# biases its truncation toward the tail for test runs.
_PYTEST_PASSED_RE = re.compile(r"\b\d+ passed\b")
# Anything that means "this run does not vouch for the change", even alongside
# a passed count: partial failures, a suite that never got off the ground, or
# a pytest that refused the command line.
_PYTEST_TROUBLE_RE = re.compile(
    r"\b\d+ (failed|error)"
    r"|ImportError while loading conftest"
    r"|INTERNALERROR"
    r"|no tests ran"
    r"|unrecognized arguments",
    re.IGNORECASE,
)

SUBMIT_TOOL_NAME = "submit"
SUBMIT_TOOL = {
    "name": SUBMIT_TOOL_NAME,
    "description": (
        "Call this as soon as one pytest run has passed against your change. It "
        "ends the session immediately and submits the repository's current state "
        "as your final answer. The only reason not to call it is that the fix is "
        "still incomplete or a test is still red."
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

The repository is checked out at {workdir}. `bash` starts there and editor
paths resolve against it, so `pkg/module.py` and `{workdir}/pkg/module.py`
both reach the same file. The checkout exists nowhere else on the filesystem,
so don't go hunting for it.

Verify with the repository's own test suite, not with ad-hoc scripts. A
`python -c` snippet speaks to one case; the test file tells you whether you
broke anything else, and it is the only thing that counts as verification
here. Run it as soon as you have a candidate fix, rather than after several
rounds of editing.

Your work is finished the moment one such run passes, and the only useful
action left at that point is to call `submit`.

Only the repository diff is graded. Turns spent after that first passing run
cannot improve it: they spend budget you may need elsewhere, and they risk
leaving the tree in a worse state than you found it.

You are not finished while a test run is red, or while you have made an edit
that no test has exercised yet. Keep working, and submit when it next passes.

Failures that were already there before your change are not yours to fix; only
failures your change introduced matter.

Every tool result tells you which turn you are on, and — once your change has
passed — how long ago that was.
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
    # Turn on which a test run last passed against a dirty tree, or None if
    # that never happened (or a later run went red). `turns - verified_turn`
    # is the tail of work that happened after the change was already known
    # good — the thing worth measuring across a batch.
    verified_turn: int | None = None


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
        log_file=None,
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
        # The one path that means the same thing to both tools: in container
        # mode bash sees the checkout at `container_workdir` and the editor
        # remaps that prefix onto the host `cwd`, so naming the host path
        # instead would be actively wrong.
        self.workdir = container_workdir if container else cwd
        self.system_prompt = SYSTEM_PROMPT.format(workdir=self.workdir)
        self.verbose = verbose
        self._console = Console() if verbose else None
        # A second console mirroring everything to `log_file`, so a run leaves
        # a complete record without anyone having to scrape terminal
        # scrollback. Fixed width and no wrapping keep the file identical
        # regardless of the terminal it ran in and keep long lines greppable;
        # rich flushes on every write, so a run that crashes still leaves
        # everything up to the crash on disk.
        self._log_console = (
            Console(file=log_file, width=200, soft_wrap=True, no_color=True, highlight=False)
            if log_file is not None
            else None
        )
        # The most recent non-empty `git diff`, kept so a run that ends while
        # the tree is transiently clean doesn't submit an empty patch. See
        # `diff()`.
        self._last_nonempty_diff: str | None = None
        self.diff_was_restored = False
        self._verified_turn: int | None = None

    @property
    def tools(self) -> list[dict]:
        return [
            {"type": self.bash.tool_type, "name": self.bash.name},
            {"type": self.editor.tool_type, "name": self.editor.name},
            SUBMIT_TOOL,
        ]

    @property
    def _recording(self) -> bool:
        return self._console is not None or self._log_console is not None

    def _rule(self, title: str) -> None:
        for console in (self._console, self._log_console):
            if console is not None:
                console.rule(title)

    def _emit(self, markup: str, body: str = "", full_body: str | None = None) -> None:
        """Write one line to the terminal and, if configured, to the log file.

        `body` is what the terminal gets — capped, since nobody reads 40kB of
        pytest output in scrollback. The file gets `full_body` where it
        differs, because the point of having a file is that it is the complete
        record.

        Bodies are model- or file-authored, so they are escaped: rich parses
        square brackets as style markup and would otherwise silently reduce
        ordinary code like `np.hstack([cleft, cright])` to `np.hstack()`,
        making the log disagree with what the model actually saw.
        """
        for console, text in (
            (self._console, body),
            (self._log_console, body if full_body is None else full_body),
        ):
            if console is not None:
                console.print(f"{markup} {escape(text)}" if text else markup)

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

        # Only the file gets the task — it's what makes the log self-contained,
        # but it's already on screen for anyone watching live.
        if self._log_console is not None:
            self._log_console.rule("task")
            self._log_console.print(escape(task))

        for turn in range(1, self.max_turns + 1):
            self._rule(f"turn {turn}")
            self.editor.current_turn = turn

            response = self.client.messages.create(
                model=self.model,
                max_tokens=_MAX_TOKENS,
                system=self.system_prompt,
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
            u = response.usage
            self._emit(
                f"[dim]usage: input={u.input_tokens} "
                f"cache_read={u.cache_read_input_tokens or 0} "
                f"cache_creation={u.cache_creation_input_tokens or 0} "
                f"output={u.output_tokens}[/]"
            )

            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            if self._recording:
                for block in response.content:
                    if block.type == "text" and block.text.strip():
                        self._emit("[bold cyan]assistant:[/]", block.text.strip())
                    elif block.type == "tool_use":
                        args = json.dumps(block.input)
                        self._emit(
                            "[bold yellow]tool call:[/]",
                            f"{block.name}({_truncate(args)})",
                            f"{block.name}({args})",
                        )

            if response.stop_reason != "tool_use":
                break

            tool_results = []
            # Results from real tools, annotated only once every tool in the
            # turn has run — the footer depends on the tree state they leave
            # behind, which isn't known until then.
            executed: list[tuple[object, dict]] = []
            for block in tool_use_blocks:
                if block.name == SUBMIT_TOOL_NAME:
                    summary = block.input.get("summary", "")
                    tool_results.append(
                        {"type": "tool_result", "tool_use_id": block.id, "content": "submitted"}
                    )
                    self._emit("[bold magenta]submitted:[/]", summary)
                    continue
                result = self._execute_tool(block)
                executed.append((block, result))
                tool_results.append(result)

            self._update_turn_state(turn, executed)

            for _, result in executed:
                # Appending (rather than rewriting past results) keeps the
                # history byte-identical turn over turn, so the prefix cache
                # still hits.
                result["content"] += self._result_footer(turn)
                style = "bold red" if result.get("is_error") else "green"
                self._emit(
                    f"[{style}]tool result:[/]",
                    _truncate(result["content"]),
                    result["content"],
                )

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
            verified_turn=self._verified_turn,
            **usage_totals,
        )

    def _classify_test_run(self, block, result: dict) -> str | None:
        """"pass"/"fail" if this tool call was a test run, else None.

        A run counts as passing only when pytest itself says so. The shell
        exit code is not evidence and is deliberately ignored: agents chain
        cleanup after the test (`pytest ...; mv conftest.py.bak conftest.py`),
        and `$?` then reports the cleanup, so a run where pytest died on a
        conftest ImportError without collecting a single test still comes back
        as exit 0. Telling the model that passed is worse than saying nothing.

        Anything that mentions pytest but doesn't clearly pass is treated as a
        failure, which only ever clears the submit pressure — the safe
        direction to be wrong in.
        """
        if block.name != self.bash.name:
            return None
        command = (block.input or {}).get("command", "")
        if "pytest" not in command and "py.test" not in command:
            return None
        content = result.get("content", "")
        passed = _PYTEST_PASSED_RE.search(content)
        return "pass" if passed and not _PYTEST_TROUBLE_RE.search(content) else "fail"

    def _update_turn_state(self, turn: int, executed: list[tuple[object, dict]]) -> None:
        """Snapshot the diff and track when the change was last verified green.

        A single `git diff` per turn serves both jobs: the snapshot that keeps
        a transiently-clean tree from becoming an empty submission, and the
        dirty-tree check that stops a test run made *before* any edit from
        counting as verification of a change that doesn't exist yet.
        """
        outcomes = {self._classify_test_run(block, result) for block, result in executed}

        current_diff = self._raw_diff()
        if current_diff.strip():
            self._last_nonempty_diff = current_diff

        if "fail" in outcomes:
            # Still working. Clearing the flag means a fix that verifies one
            # module at a time is never pressured while a later module is
            # still red; the pressure re-arms on the next green run.
            self._verified_turn = None
        elif "pass" in outcomes and current_diff.strip() and self._verified_turn is None:
            # Deliberately sticky: re-running an already-green test must not
            # reset the counter, since doing exactly that is the behaviour
            # this pressure exists to interrupt.
            self._verified_turn = turn

    def _result_footer(self, turn: int) -> str:
        """Turn budget, plus submit pressure once the change is verified.

        This never ends the run on its own — an agent legitimately mid-way
        through a multi-module fix can keep working and simply ignore it.
        """
        lines = [f"[turn {turn}/{self.max_turns}]"]
        if self._verified_turn is not None:
            since = turn - self._verified_turn
            if since == 0:
                lines.append(
                    "[harness] A test run just passed against your change. "
                    "If the fix is complete, call `submit` now."
                )
            else:
                lines.append(
                    f"[harness] {since} turn(s) since a test run last passed against "
                    "your change. Call `submit` unless you have a specific change "
                    "that is still unverified."
                )
        return "\n" + "\n".join(lines)

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

    def _raw_diff(self) -> str:
        result = subprocess.run(
            ["git", "diff"], cwd=self.cwd, capture_output=True, text=True
        )
        return result.stdout

    def diff(self) -> str:
        """Return the git diff of everything the agent changed in `cwd`.

        The tree is read once, after the loop has already ended, so the patch
        is really "whatever was on disk when the run stopped" — and an agent
        can leave the tree transiently clean, e.g. between a `git stash` and
        its `git stash pop`. A run that ends inside that window (max_turns
        exhausted, an API error, a timeout) would otherwise submit an empty
        patch for work it had already finished, so fall back to the last
        non-empty diff we saw.

        Restricted to the *empty* case on purpose. A final diff that is merely
        smaller than an earlier one may well be the agent deliberately
        narrowing its fix, and overriding that would discard a real decision;
        an empty patch, by contrast, scores zero no matter what, so the
        fallback cannot do worse than the alternative.
        """
        current = self._raw_diff()
        if current.strip() or not self._last_nonempty_diff:
            return current
        self.diff_was_restored = True
        return self._last_nonempty_diff
