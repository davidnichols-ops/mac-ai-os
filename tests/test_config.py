from app.config.hardware import detect
from app.config.settings import load_settings


def test_hardware_detect():
    hw = detect()
    assert hw.memory_gb >= 0
    assert hw.arch in {"arm64", "x86_64"}


def test_load_settings_fills_defaults(tmp_path):
    settings = load_settings(detect_hardware=True)
    # Empty model fields should be filled from hardware.
    assert settings.planner.model != ""
    assert settings.utility.model != ""
    assert settings.embeddings.model != ""


def test_load_settings_user_override(tmp_path):
    cfg = tmp_path / "user.yaml"
    cfg.write_text("planner:\n  provider: ollama\n  model: qwen3:14b\n")
    settings = load_settings(str(cfg), detect_hardware=True)
    assert settings.planner.provider == "ollama"
    assert settings.planner.model == "qwen3:14b"
