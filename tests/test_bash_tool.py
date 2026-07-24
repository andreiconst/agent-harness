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
