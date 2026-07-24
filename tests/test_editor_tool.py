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


def test_undo(tmp_path):
    editor = EditorTool(cwd=tmp_path)
    editor.run(command="create", path="a.py", file_text="x = 1\n")
    editor.run(command="str_replace", path="a.py", old_str="x = 1", new_str="x = 2")
    editor.run(command="undo_edit", path="a.py")
    assert (tmp_path / "a.py").read_text() == "x = 1\n"
