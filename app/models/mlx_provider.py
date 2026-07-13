"""MLX provider (Apple Silicon) backed by `mlx-lm`.

mlx-lm is imported lazily so the rest of the project remains importable on
non-Apple platforms (and in CI without the extra).
"""

from __future__ import annotations

from app.log import get_logger
from app.models.base import EmbeddingModel, Model
from app.models.types import EmbeddingResponse, Message, ModelResponse

log = get_logger(__name__)


def _import_mlx_lm():
    try:
        import mlx_lm  # type: ignore
        return mlx_lm
    except ImportError as e:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "mlx-lm is not installed. Install with: uv sync --extra mlx"
        ) from e


class MLXModel(Model):
    """A chat/generation model served by mlx-lm."""

    provider = "mlx"

    def __init__(self, model: str, *, temperature: float = 0.7, max_tokens: int = 1024):
        self.name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._mlx = _import_mlx_lm()
        self._model = None
        self._tokenizer = None
        log.info("mlx.model.requested", model=model)

    def _load(self):
        if self._model is not None:
            return
        log.info("mlx.model.loading", model=self.name)
        self._model, self._tokenizer = self._mlx.load(self.name)
        log.info("mlx.model.loaded", model=self.name)

    def generate(self, prompt: str, *, system: str | None = None, **kwargs) -> ModelResponse:
        self._load()
        messages: list[Message] = []
        if system:
            messages.append(Message("system", system))
        messages.append(Message("user", prompt))
        return self.chat(messages, **kwargs)

    def chat(self, messages: list[Message], **kwargs) -> ModelResponse:
        self._load()
        temp = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        if hasattr(self._tokenizer, "apply_chat_template"):
            prompt = self._tokenizer.apply_chat_template(
                [m.to_dict() for m in messages], tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = "\n".join(f"{m.role}: {m.content}" for m in messages)

        # mlx-lm 0.31+ uses a `sampler` callable instead of a `temp` kwarg.
        gen_kwargs: dict = {"max_tokens": max_tokens, "verbose": False}
        try:
            from mlx_lm.sample_utils import make_sampler  # type: ignore

            gen_kwargs["sampler"] = make_sampler(temp=temp)
        except Exception:
            # Older mlx-lm versions accept `temp` directly.
            gen_kwargs["temp"] = temp

        response = self._mlx.generate(self._model, self._tokenizer, prompt=prompt, **gen_kwargs)
        text = response.text if hasattr(response, "text") else str(response)
        return ModelResponse(text=text, model=self.name, provider=self.provider, raw=response)

    def health(self) -> bool:
        try:
            self._load()
            return self._model is not None
        except Exception as e:
            log.warning("mlx.health.failed", model=self.name, error=str(e))
            return False


class MLXEmbeddingModel(EmbeddingModel):
    """Embeddings via mlx-lm using an embedding model from mlx-community."""

    provider = "mlx"

    def __init__(self, model: str):
        self.name = model
        self._mlx = _import_mlx_lm()
        self._model = None
        self._tokenizer = None
        log.info("mlx.embedding.requested", model=model)

    def _load(self):
        if self._model is not None:
            return
        log.info("mlx.embedding.loading", model=self.name)
        self._model, self._tokenizer = self._mlx.load(self.name)
        log.info("mlx.embedding.loaded", model=self.name)

    def embed(self, text: str) -> EmbeddingResponse:
        self._load()
        # mlx-lm exposes `embed` at module level for embedding-capable models.
        if hasattr(self._mlx, "embed"):
            vec = self._mlx.embed(self._model, self._tokenizer, text)
        elif hasattr(self._model, "embed"):
            vec = self._model.embed(self._tokenizer, text)
        else:
            raise RuntimeError(
                f"{self.name} does not expose an embed() API via mlx-lm; "
                "use an embedding model (e.g. mlx-community/bge-m3-4bit) or "
                "switch embeddings.provider to 'ollama'."
            )
        # mlx-lm may return an mlx array / numpy array / list.
        vec_list = list(vec) if not hasattr(vec, "tolist") else list(vec.tolist())
        return EmbeddingResponse(vector=vec_list, model=self.name, provider=self.provider)

    def health(self) -> bool:
        try:
            self._load()
            return self._model is not None
        except Exception as e:
            log.warning("mlx.embedding.health.failed", model=self.name, error=str(e))
            return False
