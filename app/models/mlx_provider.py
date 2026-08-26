"""MLX provider (Apple Silicon) backed by `mlx-lm`.

mlx-lm is imported lazily so the rest of the project remains importable on
non-Apple platforms (and in CI without the extra).
"""

from __future__ import annotations

import json
import re
from typing import Any

from app.log import get_logger
from app.models.base import EmbeddingModel, Model
from app.models.types import EmbeddingResponse, Message, ModelResponse, ToolCall

log = get_logger(__name__)

TC_OPEN = chr(60) + "tool_call" + chr(62)
TC_CLOSE = chr(60) + "/tool_call" + chr(62)


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
        self._mlx = None  # deferred to _load() — saves ~3s on commands that never call the model
        self._model = None
        self._tokenizer = None
        log.info("mlx.model.requested", model=model)

    def _load(self):
        if self._model is not None:
            return
        if self._mlx is None:
            self._mlx = _import_mlx_lm()
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

    def chat(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> ModelResponse:
        self._load()
        temp = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        if hasattr(self._tokenizer, "apply_chat_template"):
            # Convert messages to dicts, ensuring tool_calls arguments are
            # dicts (not JSON strings) — Qwen3.5's template requires this.
            msg_dicts = []
            for m in messages:
                d = m.to_dict()
                if d.get("tool_calls"):
                    for tc in d["tool_calls"]:
                        args = tc.get("function", {}).get("arguments")
                        if isinstance(args, str):
                            try:
                                tc["function"]["arguments"] = json.loads(args)
                            except (json.JSONDecodeError, ValueError):
                                pass
                msg_dicts.append(d)
            prompt = self._tokenizer.apply_chat_template(
                msg_dicts, tokenize=False, add_generation_prompt=True
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

        # Ensure the tokenizer's eos_token is in eos_token_ids.
        # Some models (e.g. Phi-3.5) have eos_token=<|end|> (id 32007) but
        # eos_token_ids={32000} — a different token. mlx-lm checks
        # eos_token_ids to stop generation, so without this fix the model
        # generates past its answer and fills context with repeated junk.
        eos_id = getattr(self._tokenizer, "eos_token_id", None)
        eos_ids = getattr(self._tokenizer, "eos_token_ids", set())
        if eos_id is not None and eos_id not in eos_ids:
            if hasattr(self._tokenizer, "add_eos_token"):
                self._tokenizer.add_eos_token(self._tokenizer.eos_token)
            else:
                # Fallback: directly patch the set if possible
                try:
                    self._tokenizer.eos_token_ids = eos_ids | {eos_id}
                except Exception:
                    pass

        response = self._mlx.generate(self._model, self._tokenizer, prompt=prompt, **gen_kwargs)
        # mlx_lm 0.31+ generate() returns str directly; older versions return
        # an object with .text. Handle both cleanly.
        if isinstance(response, str):
            text = response
        elif hasattr(response, "text"):
            text = response.text
        else:
            text = str(response)
        tool_calls = self._parse_tool_calls(text) if tools else []
        finish = "tool_calls" if tool_calls else "stop"
        return ModelResponse(
            text=text, model=self.name, provider=self.provider, raw=response,
            tool_calls=tool_calls, finish_reason=finish,
        )

    @staticmethod
    def _parse_tool_calls(text: str) -> list[ToolCall]:
        """Parse <tool_call> JSON blocks from model output."""
        tool_calls = []
        pat = TC_OPEN + r"\s*(\{.*?\})\s*(?:" + TC_CLOSE + r"|\n\n|$)"
        for i, match in enumerate(re.findall(pat, text, re.DOTALL)):
            try:
                data = json.loads(match)
                name = data.get("name", "")
                args = data.get("arguments", data.get("parameters", {}))
                if name:
                    tool_calls.append(ToolCall(id=f"call_{i}", name=name, arguments=args))
            except (json.JSONDecodeError, ValueError):
                continue
        return tool_calls

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
        self._mlx = None  # deferred to _load()
        self._model = None
        self._tokenizer = None
        log.info("mlx.embedding.requested", model=model)

    def _load(self):
        if self._model is not None:
            return
        if self._mlx is None:
            self._mlx = _import_mlx_lm()
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
