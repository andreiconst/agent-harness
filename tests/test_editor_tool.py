from agent_harness.tools.editor import EditorTool


def test_create_and_view(tmp_path):
    editor = EditorTool(cwd=tmp_path)
    editor.run(command="create", path="hello.txt", file_text="line1\nline2\n")
    assert (tmp_path / "hello.txt").read_text() == "line1\nline2\n"

    view = editor.run(command="view", path="hello.txt")
    assert "1\tline1" in view
    assert "2\tline2" in view


def test_str_replace(tmp_path):
    editor = EditorTool(cwd=tmp_path)
    editor.run(command="create", path="a.py", file_text="x = 1\n")
    editor.run(command="str_replace", path="a.py", old_str="x = 1", new_str="x = 2")
    assert (tmp_path / "a.py").read_text() == "x = 2\n"


def test_str_replace_requires_unique_match(tmp_path):
    editor = EditorTool(cwd=tmp_path)
    editor.run(command="create", path="a.py", file_text="x = 1\nx = 1\n")
    try:
        editor.run(command="str_replace", path="a.py", old_str="x = 1", new_str="x = 2")
        assert False, "expected a ValueError for a non-unique match"
    except ValueError:
        pass


def test_insert(tmp_path):
    editor = EditorTool(cwd=tmp_path)
    editor.run(command="create", path="a.py", file_text="line1\nline3\n")
    editor.run(command="insert", path="a.py", insert_line=1, new_str="line2")
    assert (tmp_path / "a.py").read_text() == "line1\nline2\nline3\n"


def test_unknown_command_raises(tmp_path):
    editor = EditorTool(cwd=tmp_path)
    editor.run(command="create", path="a.py", file_text="x = 1\n")
    try:
        editor.run(command="undo_edit", path="a.py")
        assert False, "expected a ValueError for an unsupported command"
    except ValueError:
        pass


def test_container_workdir_remaps_container_absolute_paths(tmp_path):
    # In --docker mode, `cwd` is a host directory bind-mounted into a
    # container at `container_workdir`. The model explores via `bash`
    # (which runs inside the container) and naturally passes back
    # container-absolute paths like "/testbed/a.py" — those must resolve
    # onto the host directory, not fail as "not found" against the host's
    # real filesystem where /testbed doesn't exist.
    editor = EditorTool(cwd=tmp_path, container_workdir="/testbed")
    editor.run(command="create", path="/testbed/a.py", file_text="x = 1\n")
    assert (tmp_path / "a.py").read_text() == "x = 1\n"

    view = editor.run(command="view", path="/testbed/a.py")
    assert "1\tx = 1" in view

    editor.run(command="str_replace", path="/testbed/a.py", old_str="x = 1", new_str="x = 2")
    assert (tmp_path / "a.py").read_text() == "x = 2\n"

    # The bare workdir itself (no trailing path) should resolve to `cwd`.
    listing = editor.run(command="view", path="/testbed")
    assert "a.py" in listing


def test_container_workdir_prefix_match_is_exact_not_substring(tmp_path):
    # "/testbed-other/..." merely shares a string prefix with "/testbed" —
    # it must NOT be remapped onto `cwd`. Since it's also outside `cwd`,
    # it's rejected by the containment check (see below), not silently
    # resolved against the real host filesystem.
    editor = EditorTool(cwd=tmp_path, container_workdir="/testbed")
    try:
        editor._resolve("/testbed-other/b.py")
        assert False, "expected a ValueError for a path outside the working directory"
    except ValueError:
        pass
    assert editor._resolve("/testbed/b.py") == (tmp_path / "b.py").resolve()


def test_rejects_paths_outside_cwd(tmp_path):
    # Real incident: the model passed path="/" to `view`, and the old
    # implementation followed it literally against the *host* filesystem —
    # rglobbing the entire real disk (hundreds of millions of chars) into a
    # single tool_result, which then blew past the API's request-size limit
    # on the next turn. Every absolute (or `..`-escaping) path outside `cwd`
    # must be rejected outright, in every mode, not just --docker.
    editor = EditorTool(cwd=tmp_path)
    for bad_path in ["/", "/etc/passwd", "..", "../outside.txt", str(tmp_path.parent)]:
        try:
            editor.run(command="view", path=bad_path)
            assert False, f"expected a ValueError for out-of-bounds path {bad_path!r}"
        except ValueError:
            pass


def test_rejects_paths_outside_cwd_in_docker_mode_too(tmp_path):
    # Same containment check must hold when container_workdir is set — an
    # absolute path that isn't under /testbed (or is testbed-prefixed but
    # escapes it) must not fall through to a literal host lookup.
    editor = EditorTool(cwd=tmp_path, container_workdir="/testbed")
    for bad_path in ["/", "/opt/miniconda3/envs/testbed", "/testbed/../../etc/passwd"]:
        try:
            editor.run(command="view", path=bad_path)
            assert False, f"expected a ValueError for out-of-bounds path {bad_path!r}"
        except ValueError:
            pass


def test_view_output_is_truncated(tmp_path):
    editor = EditorTool(cwd=tmp_path)
    editor.run(command="create", path="big.py", file_text="x = 1\n" * 2000)
    view = editor.run(command="view", path="big.py")
    assert len(view) < 6000
    assert "chars omitted" in view
