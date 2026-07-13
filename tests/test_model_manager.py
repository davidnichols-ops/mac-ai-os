"""Tests for the model lifecycle manager."""

from app.models.base import Model
from app.models.manager import ModelManager
from app.models.types import ModelResponse


class FakeModel(Model):
    """A fake model that tracks load/unload."""

    provider = "fake"

    def __init__(self, name="fake"):
        self.name = name
        self._model = None
        self._load_count = 0

    def _load(self):
        self._model = "loaded"
        self._load_count += 1

    def generate(self, prompt, *, system=None, **kwargs):
        self._load()
        return ModelResponse(text="generated", model=self.name)

    def chat(self, messages, **kwargs):
        self._load()
        return ModelResponse(text="chat response", model=self.name)


def test_model_manager_register():
    manager = ModelManager(max_memory_gb=12.0)
    model = FakeModel("test")
    manager.register("test", model, estimated_memory_gb=2.0)

    status = manager.status()
    assert len(status) == 1
    assert status[0]["name"] == "test"
    assert status[0]["loaded"] is False


def test_model_manager_activate():
    manager = ModelManager(max_memory_gb=12.0)
    model = FakeModel("test")
    manager.register("test", model, estimated_memory_gb=2.0)

    manager.activate("test")
    assert model._model == "loaded"
    assert model._load_count == 1

    status = manager.status()
    assert status[0]["loaded"] is True


def test_model_manager_deactivate():
    manager = ModelManager(max_memory_gb=12.0)
    model = FakeModel("test")
    manager.register("test", model, estimated_memory_gb=2.0)

    manager.activate("test")
    assert model._model == "loaded"

    manager.deactivate("test")
    assert model._model is None


def test_model_manager_memory_pressure():
    """When memory is tight, activating a large model should unload others."""
    manager = ModelManager(max_memory_gb=10.0)

    small = FakeModel("small")
    large = FakeModel("large")
    manager.register("small", small, estimated_memory_gb=4.0)
    manager.register("large", large, estimated_memory_gb=8.0)

    # Load small first.
    manager.activate("small")
    assert small._model == "loaded"

    # Loading large (4+8=12 > 10) should unload small.
    manager.activate("large")
    assert large._model == "loaded"
    assert small._model is None  # unloaded


def test_model_manager_deactivate_all():
    manager = ModelManager(max_memory_gb=20.0)
    m1 = FakeModel("m1")
    m2 = FakeModel("m2")
    manager.register("m1", m1, estimated_memory_gb=2.0)
    manager.register("m2", m2, estimated_memory_gb=2.0)

    manager.activate("m1")
    manager.activate("m2")
    assert m1._model == "loaded"
    assert m2._model == "loaded"

    manager.deactivate_all()
    assert m1._model is None
    assert m2._model is None
