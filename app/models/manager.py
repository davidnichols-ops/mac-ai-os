"""Model lifecycle manager — tracks loaded models and manages memory.

On a 16 GB Mac, loading Qwen3-14B (~9 GB) plus a utility model (~2 GB) leaves
little headroom. The ModelManager tracks which models are loaded, how much
memory they use, and can unload models that haven't been used recently.

Usage:
    manager = ModelManager()
    manager.register("planner", planner_model)
    manager.register("utility", utility_model)
    manager.activate("planner")  # loads on demand
    # ... use planner ...
    manager.deactivate("utility")  # unloads if loaded
"""

from __future__ import annotations

import time
from typing import Any

from app.log import get_logger
from app.models.base import Model

log = get_logger(__name__)


class ManagedModel:
    """A model wrapper that tracks usage and memory."""

    def __init__(self, name: str, model: Model, *, estimated_memory_gb: float = 0.0):
        self.name = name
        self.model = model
        self.estimated_memory_gb = estimated_memory_gb
        self.loaded = False
        self.last_used = 0.0
        self.use_count = 0

    def activate(self) -> None:
        """Ensure the model is loaded."""
        if not self.loaded:
            # For MLX models, loading happens on first generate/chat call.
            # We trigger it by checking if the underlying model has _load.
            loader = getattr(self.model, "_load", None)
            if loader:
                loader()
                self.loaded = True
                log.info("model_manager.activated", name=self.name,
                         memory_gb=self.estimated_memory_gb)
        self.last_used = time.time()
        self.use_count += 1

    def deactivate(self) -> None:
        """Unload the model from memory."""
        if not self.loaded:
            return
        # For MLX models, we release the references so Python can GC them.
        # mlx-lm doesn't have an explicit unload, but dropping references works.
        if hasattr(self.model, "_model"):
            self.model._model = None
        if hasattr(self.model, "_tokenizer"):
            self.model._tokenizer = None
        self.loaded = False
        log.info("model_manager.deactivated", name=self.name,
                 memory_gb=self.estimated_memory_gb, uses=self.use_count)

    def age_seconds(self) -> float:
        """Seconds since last use."""
        return time.time() - self.last_used if self.last_used else float("inf")


class ModelManager:
    """Manages model lifecycle and memory pressure."""

    def __init__(self, *, max_memory_gb: float = 12.0, idle_timeout: float = 300.0):
        self.models: dict[str, ManagedModel] = {}
        self.max_memory_gb = max_memory_gb
        self.idle_timeout = idle_timeout

    def register(self, name: str, model: Model, *, estimated_memory_gb: float = 0.0) -> None:
        self.models[name] = ManagedModel(name, model, estimated_memory_gb=estimated_memory_gb)
        log.info("model_manager.registered", name=name,
                 memory_gb=estimated_memory_gb)

    def activate(self, name: str) -> None:
        """Activate a model, unloading others if memory is tight."""
        if name not in self.models:
            raise KeyError(f"model not registered: {name}")

        target = self.models[name]
        loaded_memory = sum(
            m.estimated_memory_gb for m in self.models.values() if m.loaded
        )

        # If loading this model would exceed memory, unload idle models.
        if not target.loaded:
            needed = loaded_memory + target.estimated_memory_gb
            if needed > self.max_memory_gb:
                self._unload_idle(exclude=name)
                # Recheck.
                loaded_memory = sum(
                    m.estimated_memory_gb for m in self.models.values() if m.loaded
                )
                if loaded_memory + target.estimated_memory_gb > self.max_memory_gb:
                    # Unload the least recently used loaded model.
                    self._unload_lru(exclude=name)

        target.activate()

    def deactivate(self, name: str) -> None:
        if name in self.models:
            self.models[name].deactivate()

    def deactivate_all(self) -> None:
        for m in self.models.values():
            m.deactivate()

    def cleanup_idle(self) -> int:
        """Unload models that have been idle for longer than the timeout."""
        unloaded = 0
        for _name, m in self.models.items():
            if m.loaded and m.age_seconds() > self.idle_timeout:
                m.deactivate()
                unloaded += 1
        if unloaded:
            log.info("model_manager.cleanup", unloaded=unloaded)
        return unloaded

    def status(self) -> list[dict[str, Any]]:
        """Return status of all managed models."""
        return [
            {
                "name": m.name,
                "loaded": m.loaded,
                "memory_gb": m.estimated_memory_gb,
                "use_count": m.use_count,
                "idle_seconds": int(m.age_seconds()) if m.last_used else None,
            }
            for m in self.models.values()
        ]

    def total_memory_gb(self) -> float:
        return sum(m.estimated_memory_gb for m in self.models.values() if m.loaded)

    def _unload_idle(self, exclude: str = "") -> None:
        for name, m in self.models.items():
            if name == exclude or not m.loaded:
                continue
            if m.age_seconds() > self.idle_timeout:
                m.deactivate()

    def _unload_lru(self, exclude: str = "") -> None:
        """Unload the least recently used model."""
        candidates = [
            (name, m) for name, m in self.models.items()
            if m.loaded and name != exclude
        ]
        if not candidates:
            return
        candidates.sort(key=lambda x: x[1].last_used)
        name, m = candidates[0]
        m.deactivate()
