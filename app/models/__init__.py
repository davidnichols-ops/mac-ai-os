"""Model abstraction package."""

from app.models.base import EmbeddingModel, Model
from app.models.registry import build_embedding_model, build_model
from app.models.types import EmbeddingResponse, Message, ModelResponse

__all__ = [
    "Model",
    "EmbeddingModel",
    "Message",
    "ModelResponse",
    "EmbeddingResponse",
    "build_model",
    "build_embedding_model",
]
