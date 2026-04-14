"""Thin, pluggable LLM provider interface.

The detector treats any callable matching :class:`LLMProvider` as a provider.
This keeps the core logic independent of which SDK you use and lets you mix
Anthropic, OpenAI, Gemini, or a local stub.

A provider returns a :class:`LLMResponse` with the generated text plus
optional per-token logprobs used by the uncertainty layer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class LLMResponse:
    text: str
    # Per-token log probabilities for the selected tokens, if available.
    token_logprobs: list[float] = field(default_factory=list)
    tokens: list[str] = field(default_factory=list)
    # Raw response payload, provider-specific, for debugging.
    raw: Any = None
    model: str = ""


class LLMProvider(Protocol):
    name: str
    model: str

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        logprobs: bool = False,
    ) -> LLMResponse: ...


# --------------------------------------------------------------------------- #
# Implementations
# --------------------------------------------------------------------------- #


class AnthropicProvider:
    """Anthropic Claude provider. Claude does not currently expose logprobs,
    so :attr:`LLMResponse.token_logprobs` will be empty."""

    name = "anthropic"

    def __init__(self, model: str = "claude-opus-4-6", api_key: str | None = None):
        try:
            import anthropic  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "anthropic SDK not installed. pip install anthropic"
            ) from e
        from anthropic import Anthropic

        self.model = model
        self._client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        logprobs: bool = False,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            # Use prompt caching on the system prompt - it's typically reused
            # across many detection runs (RAG context, judge instructions).
            kwargs["system"] = [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ]
        resp = self._client.messages.create(**kwargs)
        text = "".join(block.text for block in resp.content if block.type == "text")
        return LLMResponse(text=text, raw=resp, model=self.model)


class OpenAIProvider:
    """OpenAI provider. Supports ``logprobs=True`` for uncertainty scoring."""

    name = "openai"

    def __init__(self, model: str = "gpt-4o", api_key: str | None = None):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("openai SDK not installed. pip install openai") from e
        self.model = model
        self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        logprobs: bool = False,
    ) -> LLMResponse:
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=logprobs,
            top_logprobs=5 if logprobs else None,
        )
        choice = resp.choices[0]
        text = choice.message.content or ""
        tok_logprobs: list[float] = []
        tokens: list[str] = []
        if logprobs and choice.logprobs and choice.logprobs.content:
            for t in choice.logprobs.content:
                tokens.append(t.token)
                tok_logprobs.append(t.logprob)
        return LLMResponse(
            text=text,
            token_logprobs=tok_logprobs,
            tokens=tokens,
            raw=resp,
            model=self.model,
        )


class GeminiProvider:
    """Google Gemini provider."""

    name = "gemini"

    def __init__(self, model: str = "gemini-1.5-pro", api_key: str | None = None):
        try:
            import google.generativeai as genai
        except ImportError as e:
            raise ImportError(
                "google-generativeai not installed. pip install google-generativeai"
            ) from e
        genai.configure(api_key=api_key or os.environ.get("GOOGLE_API_KEY"))
        self.model = model
        self._genai = genai
        self._model = genai.GenerativeModel(model)

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        logprobs: bool = False,
    ) -> LLMResponse:
        full = prompt if not system else f"{system}\n\n{prompt}"
        resp = self._model.generate_content(
            full,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
        )
        return LLMResponse(text=resp.text or "", raw=resp, model=self.model)


class StubProvider:
    """Deterministic stub used in tests and offline demos.

    Pass a ``responder`` callable that maps (prompt, system) -> text.
    """

    name = "stub"

    def __init__(self, responder, model: str = "stub-1", logprob_fn=None):
        self.model = model
        self._responder = responder
        self._logprob_fn = logprob_fn

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        logprobs: bool = False,
    ) -> LLMResponse:
        text = self._responder(prompt, system)
        token_logprobs: list[float] = []
        tokens: list[str] = []
        if logprobs and self._logprob_fn is not None:
            tokens, token_logprobs = self._logprob_fn(text)
        return LLMResponse(
            text=text,
            token_logprobs=token_logprobs,
            tokens=tokens,
            model=self.model,
        )
