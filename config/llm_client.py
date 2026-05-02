"""LLM client — supports Ollama and LM Studio backends.

Backend selection (set once via env var before starting):
  LLM_BACKEND=ollama      Use Ollama (default). OLLAMA_HOST overrides server URL.
  LLM_BACKEND=lmstudio    Use LM Studio. LMSTUDIO_HOST overrides server URL.

LM Studio setup:
  1. Open LM Studio → Local Server tab → Start Server
  2. Default address: http://localhost:1234/v1
  3. Load models in LM Studio's model library before running
  4. Set lmstudio_model on each ModelSpec (LM Studio IDs differ from Ollama tags)

Model loading in LM Studio:
  LM Studio loads one model at a time by default. When the pipeline switches to an
  agent with a different model ID, LM Studio unloads the current model and loads the
  new one (~10–30s). To avoid this penalty, either:
    a) Use the same model ID for all agents (simplest), or
    b) Enable "Multi-Model" in LM Studio settings (keeps multiple models in RAM).
"""

import asyncio
import os
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

from ui.console import console

if TYPE_CHECKING:
    from config.models import ModelSpec


def _get_backend() -> str:
    return os.getenv("LLM_BACKEND", "ollama").lower()


# =============================================================================
#  OLLAMA BACKEND
# =============================================================================

class OllamaBackend:
    """Inference backend using Ollama."""

    def _host(self) -> str:
        return os.getenv("OLLAMA_HOST", "http://localhost:11434")

    def chat(
        self,
        model: str,
        messages: List[Dict],
        temperature: float,
        max_tokens: int,
        n_ctx: int,
    ) -> Tuple[str, int, int]:
        import ollama as _ollama
        client = _ollama.Client(host=self._host())
        console.info(f"[Ollama] {model} (ctx={n_ctx})")
        response = client.chat(
            model=model,
            messages=messages,
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": n_ctx,
            },
        )
        text = response["message"]["content"] or ""
        input_tokens = response.get("prompt_eval_count", 0)
        output_tokens = response.get("eval_count", 0)
        return text, input_tokens, output_tokens

    def chat_stream(
        self,
        model: str,
        messages: List[Dict],
        temperature: float,
        max_tokens: int,
        callback: Callable[[str], None],
        n_ctx: int,
    ) -> Tuple[str, int, int]:
        import ollama as _ollama
        client = _ollama.Client(host=self._host())
        console.info(f"[Ollama] {model} streaming (ctx={n_ctx})")
        stream = client.chat(
            model=model,
            messages=messages,
            stream=True,
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": n_ctx,
            },
        )
        full_response = ""
        input_tokens = 0
        output_tokens = 0
        for chunk in stream:
            delta = chunk["message"]["content"] or ""
            if delta:
                full_response += delta
                callback(delta)
            if chunk.get("done"):
                input_tokens = chunk.get("prompt_eval_count", 0)
                output_tokens = chunk.get("eval_count", 0)
        return full_response, input_tokens, output_tokens


# =============================================================================
#  LM STUDIO BACKEND
# =============================================================================

class LMStudioBackend:
    """Inference backend using LM Studio's OpenAI-compatible local server.

    LM Studio loads models on demand: if the requested model_id is not currently
    loaded, LM Studio will unload whatever is in memory and load the new one.
    This causes a ~10–30s delay on first call per unique model.

    To reduce swaps: assign the same lmstudio_model to all agents that can share
    one model, or enable Multi-Model mode in LM Studio settings.
    """

    def __init__(self):
        from openai import OpenAI
        host = os.getenv("LMSTUDIO_HOST", "http://localhost:1234/v1")
        # LM Studio does not require a real API key — any non-empty string works
        self._client = OpenAI(base_url=host, api_key="lm-studio")
        console.info(f"[LMStudio] backend initialised at {host}")

    def chat(
        self,
        model: str,
        messages: List[Dict],
        temperature: float,
        max_tokens: int,
        n_ctx: int,  # LM Studio ignores this; context set per-model in its UI
    ) -> Tuple[str, int, int]:
        console.info(f"[LMStudio] {model}")
        response = self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = response.choices[0].message.content or ""
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        return text, input_tokens, output_tokens

    def chat_stream(
        self,
        model: str,
        messages: List[Dict],
        temperature: float,
        max_tokens: int,
        callback: Callable[[str], None],
        n_ctx: int,
    ) -> Tuple[str, int, int]:
        console.info(f"[LMStudio] {model} streaming")
        stream = self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        full_response = ""
        input_tokens = 0
        output_tokens = 0
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                full_response += delta
                callback(delta)
            # LM Studio emits usage on the final chunk
            if hasattr(chunk, "usage") and chunk.usage:
                input_tokens = chunk.usage.prompt_tokens or 0
                output_tokens = chunk.usage.completion_tokens or 0
        return full_response, input_tokens, output_tokens


# =============================================================================
#  UNIFIED CLIENT
# =============================================================================

class LLMClient:
    """Unified LLM client — picks Ollama or LM Studio based on LLM_BACKEND env var."""

    def __init__(self):
        self._ollama: Optional[OllamaBackend] = None
        self._lmstudio: Optional[LMStudioBackend] = None

    @property
    def _backend(self) -> str:
        # Resolve backend on every access so late env-var changes (e.g. --backend
        # flag setting LLM_BACKEND after the singleton was created) take effect.
        return _get_backend()

    def _ollama_backend(self) -> OllamaBackend:
        if not self._ollama:
            self._ollama = OllamaBackend()
        return self._ollama

    def _lmstudio_backend(self) -> LMStudioBackend:
        if not self._lmstudio:
            self._lmstudio = LMStudioBackend()
        return self._lmstudio

    def _resolve_model_id(self, model_spec: "ModelSpec") -> str:
        return model_spec.model_id(self._backend)

    def chat(
        self,
        model_spec: "ModelSpec",
        messages: List[Dict],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> Tuple[str, int, int]:
        """Returns (response_text, input_tokens, output_tokens)."""
        model_id = self._resolve_model_id(model_spec)
        if self._backend != "ollama":
            return self._lmstudio_backend().chat(
                model_id, messages, temperature, max_tokens, model_spec.context_window
            )
        return self._ollama_backend().chat(
            model_id, messages, temperature, max_tokens, model_spec.context_window
        )

    def chat_stream(
        self,
        model_spec: "ModelSpec",
        messages: List[Dict],
        callback: Callable[[str], None],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> Tuple[str, int, int]:
        """Streaming chat. Returns (full_response, input_tokens, output_tokens)."""
        model_id = self._resolve_model_id(model_spec)
        if self._backend != "ollama":
            return self._lmstudio_backend().chat_stream(
                model_id, messages, temperature, max_tokens, callback, model_spec.context_window
            )
        return self._ollama_backend().chat_stream(
            model_id, messages, temperature, max_tokens, callback, model_spec.context_window
        )

    async def chat_async(
        self,
        model_spec: "ModelSpec",
        messages: List[Dict],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> Tuple[str, int, int]:
        """Async wrapper — runs the sync call in a thread pool."""
        return await asyncio.to_thread(self.chat, model_spec, messages, temperature, max_tokens)


# Global singleton
_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Return the global LLMClient instance."""
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
