from agent_harness.tools.editor import EditorTool


def test_create_and_view(tmp_path):
    editor = EditorTool(cwd=tmp_path)
    editor.run(command="create", path="hello.txt", file_text="line1\nline2\n")
    assert (tmp_path / "hello.txt").read_text() == "line1\nline2\n"

    view = editor.run(command="view", path="hello.txt")
    assert "1\tline1" in view
    assert "2\tline2" in view


def _numbered_file(tmp_path, name="a.py", n=400):
    editor = EditorTool(cwd=tmp_path)
    editor.run(command="create", path=name, file_text="".join(f"line{i}\n" for i in range(n)))
    return editor


def test_a_range_already_shown_is_suppressed(tmp_path):
    # The real pattern from a debugging run: read a window, then re-read
    # narrower windows inside it over and over.
    editor = _numbered_file(tmp_path)
    editor.current_turn = 8
    wide = editor.run(command="view", path="a.py", view_range=[219, 350])
    assert "line219" in wide

    editor.current_turn = 21
    for narrow in ([219, 248], [234, 250], [236, 262]):
        result = editor.run(command="view", path="a.py", view_range=narrow)
        assert "you viewed lines 219-350 of this file on turn 8" in result
        assert "line240" not in result


def test_a_range_extending_past_what_was_shown_is_not_suppressed(tmp_path):
    editor = _numbered_file(tmp_path)
    editor.run(command="view", path="a.py", view_range=[219, 350])

    result = editor.run(command="view", path="a.py", view_range=[300, 400])
    assert "line380" in result


def test_an_edit_restores_the_full_view(tmp_path):
    # The case that must never be suppressed: line numbers shift after an
    # edit, so everything previously shown is stale.
    editor = _numbered_file(tmp_path)
    editor.run(command="view", path="a.py", view_range=[10, 20])
    editor.run(command="str_replace", path="a.py", old_str="line15\n", new_str="CHANGED\n")

    result = editor.run(command="view", path="a.py", view_range=[10, 20])
    assert "CHANGED" in result
    assert "already in your context" not in result


def test_a_truncated_view_only_covers_what_it_delivered(tmp_path):
    # A whole-file view of a big file gets cut off; the lines past the cut
    # were never sent, so asking for them must return them.
    editor = _numbered_file(tmp_path, n=2000)
    whole = editor.run(command="view", path="a.py")
    assert "more chars omitted" in whole

    result = editor.run(command="view", path="a.py", view_range=[1500, 1520])
    assert "line1510" in result


def test_a_repeated_directory_listing_is_suppressed(tmp_path):
    editor = _numbered_file(tmp_path)
    first = editor.run(command="view", path=".")
    assert "a.py" in first
    assert "identical to the listing" in editor.run(command="view", path=".")


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
