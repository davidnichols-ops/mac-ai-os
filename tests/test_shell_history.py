
from app.db import Database
from app.observers.shell import import_shell_history, parse_zsh_history


def test_parse_zsh_history(tmp_path):
    hist = tmp_path / ".zsh_history"
    hist.write_text(
        ": 1705312800:0;git status\n"
        ": 1705312805:0;uv run pytest\n"
        ": 1705312810:0;echo hello world\n"
    )

    commands = parse_zsh_history(hist)
    assert len(commands) == 3
    assert commands[0]["command"] == "git status"
    assert commands[0]["timestamp"] == "1705312800"
    assert commands[1]["command"] == "uv run pytest"


def test_parse_zsh_multiline(tmp_path):
    hist = tmp_path / ".zsh_history"
    hist.write_text(
        ": 1705312800:0;python -c 'print(\nhello\n)'\n"
        ": 1705312810:0;git commit\n"
    )

    commands = parse_zsh_history(hist)
    assert len(commands) == 2
    assert "python -c" in commands[0]["command"]
    assert commands[1]["command"] == "git commit"


def test_import_shell_history(tmp_path):
    hist = tmp_path / ".zsh_history"
    hist.write_text(
        ": 1705312800:0;git status\n"
        ": 1705312805:0;uv run pytest\n"
        ": 1705312810:0;echo hello\n"
    )

    db = Database(str(tmp_path / "test.db"))
    count = import_shell_history(db, history_path=str(hist))
    assert count == 3
    assert db.shell_command_count() == 3

    # Second import should skip already-imported commands.
    count2 = import_shell_history(db, history_path=str(hist))
    assert count2 == 0
    db.close()


def test_search_history(tmp_path):
    hist = tmp_path / ".zsh_history"
    hist.write_text(
        ": 1705312800:0;git status\n"
        ": 1705312805:0;uv run pytest\n"
        ": 1705312810:0;python script.py\n"
    )

    db = Database(str(tmp_path / "test.db"))
    import_shell_history(db, history_path=str(hist))

    from app.observers.shell import search_history

    results = search_history(db, "pytest")
    assert len(results) == 1
    assert "pytest" in results[0]["command"]

    results_git = search_history(db, "git")
    assert len(results_git) >= 1
    db.close()
