import io
import subprocess
from types import SimpleNamespace

from rich.console import Console

from agent_harness.agent import Agent


def _usage():
    return SimpleNamespace(
        input_tokens=1,
        output_tokens=1,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )


def _tool_use(name, tool_input, block_id="tu_1"):
    return SimpleNamespace(type="tool_use", name=name, input=tool_input, id=block_id)


def _text(text):
    return SimpleNamespace(type="text", text=text)


class StubClient:
    """Replays a canned list of responses in place of the real API."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **_kwargs):
        content = self._responses.pop(0)
        stop = "tool_use" if any(b.type == "tool_use" for b in content) else "end_turn"
        return SimpleNamespace(content=content, stop_reason=stop, usage=_usage())


def _git_repo(tmp_path):
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (tmp_path / "a.txt").write_text("original\n")
    (tmp_path / "b.txt").write_text("original\n")
    git("add", "-A")
    git("commit", "-qm", "init")


# Commands the classifier reads as test runs, without needing a real suite.
GREEN_RUN = "echo 'pytest: 12 passed in 0.15s'"
RED_RUN = "echo 'pytest: 1 failed, 11 passed'; false"  # `false`, not `exit`, or the shell dies

# Verbatim from a real run: pytest died on a conftest ImportError having
# collected nothing, but the agent had chained a cleanup `mv` after it, so the
# shell reported success. Trusting `$?` here told the model its change was
# verified when no test had run at all.
CONFTEST_CRASH = (
    "echo \"ImportError while loading conftest '/repo/astropy/conftest.py'.\n"
    "E   UserWarning: could not determine astropy package version\"; "
    "true  # pytest ... ; mv conftest.py.bak conftest.py"
)


def _tool_results(messages):
    return [
        block
        for message in messages
        if message["role"] == "user" and isinstance(message["content"], list)
        for block in message["content"]
        if block.get("type") == "tool_result"
    ]


def test_tool_results_carry_the_turn_budget(tmp_path):
    client = StubClient(
        [
            [_tool_use("bash", {"command": "echo one"})],
            [_tool_use("bash", {"command": "echo two"}, block_id="tu_2")],
            [_tool_use("submit", {"summary": "done"}, block_id="tu_3")],
        ]
    )
    agent = Agent(cwd=str(tmp_path), client=client, max_turns=40, verbose=False)
    result = agent.run("task")

    results = _tool_results(result.messages)
    assert results[0]["content"].endswith("[turn 1/40]")
    assert results[1]["content"].endswith("[turn 2/40]")
    # The marker is appended once, at creation, and never rewritten — that's
    # what keeps the cached prefix byte-identical across turns.
    assert results[0]["content"].count("[turn ") == 1


def test_log_does_not_swallow_square_brackets(tmp_path):
    code = "np.hstack([cleft, cright])"
    client = StubClient(
        [
            [_text(f"The real line is {code}"), _tool_use("bash", {"command": f"echo '{code}'"})],
            [_tool_use("submit", {"summary": f"fixed {code}"}, block_id="tu_2")],
        ]
    )
    agent = Agent(cwd=str(tmp_path), client=client, max_turns=40, verbose=True)
    buffer = io.StringIO()
    agent._console = Console(file=buffer, width=200, highlight=False)
    agent.run("task")

    log = buffer.getvalue()
    # Rich treats `[...]` as style markup; unescaped, each of these would be
    # logged as a bare `np.hstack()`.
    assert log.count(code) >= 3, log  # assistant text, tool call, tool result
    assert "np.hstack()" not in log


def test_log_file_records_the_run_untruncated(tmp_path):
    # Long enough that the console cap bites, short enough that the bash tool
    # passes it through — the log records what the model saw, not what the
    # command printed.
    big = "y" * 2500
    client = StubClient(
        [
            [_tool_use("bash", {"command": f"echo {big}"})],
            [_tool_use("submit", {"summary": "done"}, block_id="tu_2")],
        ]
    )
    log, terminal = io.StringIO(), io.StringIO()
    agent = Agent(
        cwd=str(tmp_path), client=client, max_turns=40, verbose=True, log_file=log
    )
    agent._console = Console(file=terminal, width=200, highlight=False)
    agent.run("THE TASK STATEMENT")

    log_text, terminal_text = log.getvalue(), terminal.getvalue()
    assert big in log_text
    assert big not in terminal_text
    assert "more chars" in terminal_text  # the terminal still gets the cap
    # The log is self-contained; the task is already on screen for anyone
    # watching live, so it isn't repeated there.
    assert "THE TASK STATEMENT" in log_text
    assert "THE TASK STATEMENT" not in terminal_text
    assert "submitted:" in log_text


def test_log_file_works_without_a_terminal(tmp_path):
    # `--quiet --log run.log`: nothing on screen, everything on disk.
    client = StubClient(
        [
            [_tool_use("bash", {"command": "echo hello"})],
            [_tool_use("submit", {"summary": "done"}, block_id="tu_2")],
        ]
    )
    log = io.StringIO()
    agent = Agent(
        cwd=str(tmp_path), client=client, max_turns=40, verbose=False, log_file=log
    )
    agent.run("task")

    log_text = log.getvalue()
    assert "hello" in log_text
    assert "turn 1" in log_text
    assert agent._console is None


def _run(tmp_path, commands, verbose=False):
    responses = [
        [_tool_use("bash", {"command": command}, block_id=f"tu_{i}")]
        for i, command in enumerate(commands)
    ]
    responses.append([_text("done")])  # ends the loop without calling submit
    agent = Agent(
        cwd=str(tmp_path), client=StubClient(responses), max_turns=40, verbose=verbose
    )
    return agent, agent.run("task")


def test_empty_final_diff_falls_back_to_the_last_non_empty_one(tmp_path):
    _git_repo(tmp_path)
    # The stash-window failure: work is done, then stashed, and the run ends
    # before the pop. Without the snapshot this submits an empty patch.
    agent, _ = _run(tmp_path, ["echo changed > a.txt", "git stash"])

    assert agent._raw_diff().strip() == ""
    patch = agent.diff()
    assert "changed" in patch
    assert agent.diff_was_restored


def test_a_deliberately_narrowed_diff_is_left_alone(tmp_path):
    _git_repo(tmp_path)
    # Two files edited, then one reverted. The final diff is smaller than an
    # earlier snapshot but not empty, so it must be submitted as-is.
    agent, _ = _run(
        tmp_path,
        ["echo changed > a.txt; echo changed > b.txt", "git checkout -- a.txt"],
    )

    patch = agent.diff()
    assert "b.txt" in patch
    assert "a.txt" not in patch
    assert not agent.diff_was_restored


def test_green_run_on_a_clean_tree_does_not_arm_the_pressure(tmp_path):
    _git_repo(tmp_path)
    # Tests passing before any edit says nothing about a change that does not
    # exist yet.
    agent, result = _run(tmp_path, [GREEN_RUN, "echo still exploring"])

    assert result.verified_turn is None
    assert all("[harness]" not in r["content"] for r in _tool_results(result.messages))


def test_pressure_escalates_after_a_green_run_on_a_dirty_tree(tmp_path):
    _git_repo(tmp_path)
    agent, result = _run(
        tmp_path,
        ["echo changed > a.txt", GREEN_RUN, "echo gold-plating", GREEN_RUN],
    )

    contents = [r["content"] for r in _tool_results(result.messages)]
    assert result.verified_turn == 2
    assert "[harness]" not in contents[0]
    assert "just passed" in contents[1]
    assert "1 turn(s) since" in contents[2]
    # Re-running an already-green test does not reset the counter — that is
    # exactly the behaviour the pressure exists to interrupt.
    assert "2 turn(s) since" in contents[3]


def test_a_zero_exit_without_a_passing_suite_is_not_verification(tmp_path):
    _git_repo(tmp_path)
    agent, result = _run(tmp_path, ["echo changed > a.txt", CONFTEST_CRASH, "echo next"])

    contents = [r["content"] for r in _tool_results(result.messages)]
    assert "[exit code: 0]" in contents[1]  # the shell really did report success
    assert result.verified_turn is None
    assert all("[harness]" not in c for c in contents)


def test_a_partial_pass_is_not_verification(tmp_path):
    # "6 failed, 57 passed" contains a passed count but vouches for nothing.
    _git_repo(tmp_path)
    agent, result = _run(
        tmp_path, ["echo changed > a.txt", "echo 'pytest: 6 failed, 57 passed in 1.31s'"]
    )
    assert result.verified_turn is None


def test_a_red_run_clears_the_pressure(tmp_path):
    _git_repo(tmp_path)
    # A fix spanning two modules: one verifies green, the next is still red.
    # The agent must be free to keep working without being nagged.
    agent, result = _run(
        tmp_path, ["echo changed > a.txt", GREEN_RUN, RED_RUN, "echo still fixing"]
    )

    contents = [r["content"] for r in _tool_results(result.messages)]
    assert result.verified_turn is None
    assert "just passed" in contents[1]
    assert "[harness]" not in contents[2]
    assert "[harness]" not in contents[3]
