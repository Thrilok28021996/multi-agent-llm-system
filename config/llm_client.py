"""LLM client using Ollama.

Ollama must be running locally. Install: https://ollama.ai
Pull models:
  ollama pull qwen3:8b
  ollama pull mistral:8b-instruct-2410-q4_K_M
  ollama pull qwen2.5-coder:7b

Override server address with OLLAMA_HOST env var (default: http://localhost:11434).
"""

import asyncio
import os
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

from ui.console import console

if TYPE_CHECKING:
    from config.models import ModelSpec


def get_ollama_host() -> str:
    """Return the Ollama server base URL."""
    return os.getenv("OLLAMA_HOST", "http://localhost:11434")


class OllamaBackend:
    """Inference backend using Ollama."""

    def chat(
        self,
        model: str,
        messages: List[Dict],
        temperature: float,
        max_tokens: int,
        n_ctx: int,
    ) -> Tuple[str, int, int]:
        import ollama as _ollama
        host = get_ollama_host()
        client = _ollama.Client(host=host)
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
        host = get_ollama_host()
        client = _ollama.Client(host=host)
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


class LLMClient:
    """Unified LLM client — Ollama backend."""

    def __init__(self):
        self._ollama: Optional[OllamaBackend] = None

    def _resolve(self) -> str:
        return "ollama"

    def _ollama_backend(self) -> OllamaBackend:
        if not self._ollama:
            self._ollama = OllamaBackend()
        return self._ollama

    def chat(
        self,
        model_spec: "ModelSpec",
        messages: List[Dict],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> Tuple[str, int, int]:
        """Returns (response_text, input_tokens, output_tokens)."""
        return self._ollama_backend().chat(
            model_spec.ollama_model, messages, temperature, max_tokens, model_spec.context_window
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
        return self._ollama_backend().chat_stream(
            model_spec.ollama_model, messages, temperature, max_tokens, callback, model_spec.context_window
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
