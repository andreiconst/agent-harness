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
