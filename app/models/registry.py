"""Provider registry and factory.

Keeps provider construction in one place so config strings map to objects
without the rest of the codebase knowing about concrete classes.
"""

from __future__ import annotations

from app.config.settings import Settings
from app.log import get_logger
from app.models.base import EmbeddingModel, Model
from app.models.chroma_embedding import ChromaDefaultEmbeddingModel
from app.models.mlx_provider import MLXEmbeddingModel, MLXModel
from app.models.ollama_provider import OllamaEmbeddingModel, OllamaModel

log = get_logger(__name__)

_MODEL_FACTORIES = {
    "mlx": lambda cfg, settings: MLXModel(
        cfg.model, temperature=cfg.temperature, max_tokens=cfg.max_tokens
    ),
    "ollama": lambda cfg, settings: OllamaModel(
        cfg.model,
        base_url=settings.ollama.base_url,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    ),
}

_EMBEDDING_FACTORIES = {
    "mlx": lambda cfg, settings: MLXEmbeddingModel(cfg.model),
    "ollama": lambda cfg, settings: OllamaEmbeddingModel(
        cfg.model, base_url=settings.ollama.base_url
    ),
    "chroma_default": lambda cfg, settings: ChromaDefaultEmbeddingModel(),
}


def build_model(role: str, settings: Settings) -> Model:
    """role is 'planner' or 'utility'."""
    cfg = getattr(settings, role)
    factory = _MODEL_FACTORIES.get(cfg.provider)
    if factory is None:
        raise ValueError(f"Unknown model provider: {cfg.provider}")
    log.info("model.build", role=role, provider=cfg.provider, model=cfg.model)
    return factory(cfg, settings)


def build_embedding_model(settings: Settings) -> EmbeddingModel:
    cfg = settings.embeddings
    factory = _EMBEDDING_FACTORIES.get(cfg.provider)
    if factory is None:
        raise ValueError(f"Unknown embedding provider: {cfg.provider}")
    log.info("embedding.build", provider=cfg.provider, model=cfg.model)
    return factory(cfg, settings)


def register_model_provider(name: str, factory) -> None:
    _MODEL_FACTORIES[name] = factory


def register_embedding_provider(name: str, factory) -> None:
    _EMBEDDING_FACTORIES[name] = factory
