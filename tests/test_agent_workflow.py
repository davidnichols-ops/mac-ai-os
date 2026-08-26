from pathlib import Path

from app.agents.workflow import build_agent_brief, doctor_report


def _write_env(path: Path, values: dict[str, str]) -> None:
    path.write_text("".join(f"{key}={value}\n" for key, value in values.items()))
    path.chmod(0o600)


def test_doctor_reports_readiness_without_secret_values(tmp_path):
    (tmp_path / "app" / "agents").mkdir(parents=True)
    (tmp_path / "AGENTS.md").write_text("repo workflow\n")
    (tmp_path / "app" / "agents" / "AGENTS.md").write_text("agent workflow\n")
    (tmp_path / "AGENTS_WORKFLOW.md").write_text("device workflow\n")
    (tmp_path / ".grok").mkdir()
    (tmp_path / ".grok" / "config.toml").write_text("[mcp]\n")
    (tmp_path / ".grok" / "rules").mkdir()
    (tmp_path / ".grok" / "rules" / "04-personal-ecosystem.md").write_text("rules\n")
    secrets = {
        "OPENROUTER_API_KEY": "secret-openrouter-value",
        "SLACK_BOT_TOKEN": "secret-bot-value",
        "SLACK_APP_TOKEN": "secret-app-value",
        "SLACK_USER_TOKEN": "secret-user-value",
        "SLACK_TEAM_ID": "T123",
    }
    _write_env(tmp_path / ".env", secrets)

    report = doctor_report(
        "grokhack",
        root=tmp_path,
        home=tmp_path,
        environ={},
        which=lambda name: f"/bin/{name}",
    )

    assert report["healthy"] is True
    rendered = str(report)
    assert all(value not in rendered for value in secrets.values())
    assert report["credentials"]["OPENROUTER_API_KEY"]["configured"] is True
    assert report["workflow_files"]["environment"]["mode"] == "600"


def test_doctor_fails_when_required_provider_key_is_missing(tmp_path):
    (tmp_path / "app" / "agents").mkdir(parents=True)
    for path in (
        tmp_path / "AGENTS.md",
        tmp_path / "AGENTS_WORKFLOW.md",
        tmp_path / "app" / "agents" / "AGENTS.md",
    ):
        path.write_text("workflow\n")
    (tmp_path / ".grok").mkdir()
    (tmp_path / ".grok" / "config.toml").write_text("[mcp]\n")
    (tmp_path / ".grok" / "rules").mkdir()
    (tmp_path / ".grok" / "rules" / "04-personal-ecosystem.md").write_text("rules\n")
    _write_env(
        tmp_path / ".env",
        {
            "SLACK_BOT_TOKEN": "bot",
            "SLACK_APP_TOKEN": "app",
            "SLACK_USER_TOKEN": "user",
            "SLACK_TEAM_ID": "team",
        },
    )

    report = doctor_report(
        "grokhack",
        root=tmp_path,
        home=tmp_path,
        environ={},
        which=lambda name: f"/bin/{name}",
    )

    assert report["healthy"] is False
    assert report["credentials"]["OPENROUTER_API_KEY"]["configured"] is False


def test_doctor_fails_when_provider_configuration_is_missing(tmp_path):
    (tmp_path / "app" / "agents").mkdir(parents=True)
    for path in (
        tmp_path / "AGENTS.md",
        tmp_path / "AGENTS_WORKFLOW.md",
        tmp_path / "app" / "agents" / "AGENTS.md",
    ):
        path.write_text("workflow\n")
    _write_env(
        tmp_path / ".env",
        {
            "OPENROUTER_API_KEY": "openrouter",
            "SLACK_BOT_TOKEN": "bot",
            "SLACK_APP_TOKEN": "app",
            "SLACK_USER_TOKEN": "user",
            "SLACK_TEAM_ID": "team",
        },
    )

    report = doctor_report(
        "grokhack",
        root=tmp_path,
        home=tmp_path,
        environ={},
        which=lambda name: f"/bin/{name}",
    )

    assert report["healthy"] is False
    assert all(not item["exists"] for item in report["providers"]["grokhack"]["config_files"])


def test_agent_brief_is_provider_specific_and_secret_safe():
    brief = build_agent_brief("devin", "repair the pipeline")

    assert "Provider: devin" in brief
    assert "agent doctor --provider devin" in brief
    assert "agent start --provider devin" in brief
    assert "Never print" in brief
