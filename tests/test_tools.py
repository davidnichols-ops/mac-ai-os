from app.tools.builtin import FileReadTool, FileWriteTool, ShellTool
from app.tools.registry import ToolRouter


def test_shell_tool_runs():
    tool = ShellTool()
    res = tool.execute({"command": "echo hello"})
    assert res.ok
    assert "hello" in res.output


def test_shell_tool_permission_denied():
    router = ToolRouter(granted_permissions=set())  # no permissions granted
    res = router.execute("shell", {"command": "echo hi"})
    assert not res.ok
    assert "permission" in res.error.lower()


def test_file_read_write(tmp_path):
    f = tmp_path / "out.txt"
    write = FileWriteTool()
    r = write.execute({"path": str(f), "content": "hi"})
    assert r.ok
    read = FileReadTool()
    r2 = read.execute({"path": str(f)})
    assert r2.ok and r2.output == "hi"


def test_router_unknown_tool():
    router = ToolRouter(granted_permissions={"shell.exec"})
    res = router.execute("nope", {})
    assert not res.ok
    assert "unknown" in res.error


def test_router_schemas():
    router = ToolRouter(granted_permissions={"shell.exec"})
    schemas = router.schemas()
    names = {s["name"] for s in schemas}
    assert "shell" in names
