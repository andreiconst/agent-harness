from agent_harness.tools.bash import BashTool


def test_run_command_captures_output_and_exit_code():
    bash = BashTool(timeout=10)
    try:
        output = bash.run("echo hi")
        assert "hi" in output
        assert "[exit code: 0]" in output
    finally:
        bash.stop()


def test_state_persists_across_commands(tmp_path):
    bash = BashTool(cwd=str(tmp_path), timeout=10)
    try:
        bash.run("export FOO=bar")
        output = bash.run("echo $FOO")
        assert "bar" in output
    finally:
        bash.stop()


def test_nonzero_exit_code_is_reported():
    bash = BashTool(timeout=10)
    try:
        output = bash.run("false")
        assert "[exit code: 1]" in output
    finally:
        bash.stop()


def test_long_output_is_truncated_at_both_ends():
    bash = BashTool(timeout=10)
    head_marker, tail_marker = "HEAD_MARKER", "TAIL_MARKER"
    filler = "x" * 5000
    output = bash._truncate_output(f"{head_marker}{filler}{tail_marker}")
    assert head_marker in output
    assert tail_marker in output
    assert "chars omitted" in output


def test_pytest_output_keeps_the_tail_over_the_banner():
    # The banner is longer than the default head budget, so an even split
    # would spend the whole head on it and truncate the summary line away.
    bash = BashTool(timeout=10)
    banner = "test session starts\n" + "platform linux -- plugins: many\n" * 60
    body = "collected 12 items\n" + "some_test PASSED\n" * 100 + "12 passed in 0.15s"
    output = bash._truncate_output(banner + body)

    assert "12 passed in 0.15s" in output
    assert "collected 12 items" in output
    # Same total budget as any other command, just divided differently.
    head, tail = bash._split_budget(banner + body)
    assert head + tail == bash._HEAD_CHARS + bash._TAIL_CHARS
    assert head < bash._HEAD_CHARS


def test_non_pytest_output_keeps_the_even_split():
    bash = BashTool(timeout=10)
    head, tail = bash._split_budget("a build log with no test run in it")
    assert (head, tail) == (bash._HEAD_CHARS, bash._TAIL_CHARS)


def test_init_commands_run_before_first_user_command_and_persist():
    # e.g. `conda activate testbed` in --docker mode: must complete before
    # any real command runs, its own output must not leak into the first
    # real command's result, and its effect (env activation) must persist.
    bash = BashTool(timeout=10, init_commands=["export FOO=from_init"])
    try:
        output = bash.run("echo $FOO")
        assert output.strip().startswith("from_init")
    finally:
        bash.stop()
