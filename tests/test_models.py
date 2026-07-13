from app.models.types import Message
from tests.conftest import FakeModel


def test_fake_model_generate():
    m = FakeModel()
    resp = m.generate("hello")
    assert resp.text.startswith("FAKE:")
    assert "hello" in resp.text


def test_fake_model_chat():
    m = FakeModel()
    resp = m.chat([Message("user", "hi there")])
    assert "hi there" in resp.text
    assert resp.provider == "fake"
