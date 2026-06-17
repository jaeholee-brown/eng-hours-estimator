"""Async judge clients for the three ensemble providers.

Each provider exposes the same coroutine:

    await provider.complete(system, user, max_tokens=..., temperature=..., reasoning="off")
        -> Result(text, tokens_in, tokens_out)

`reasoning="off"` disables extended thinking / reasoning everywhere:
  * Anthropic Haiku 4.5 — no `thinking` param (thinking is off by default).
  * OpenAI gpt-5.4-mini — `reasoning_effort="minimal"`.
  * Google Gemini 3 Flash — `thinking_config` set to the lowest setting.

SDKs are imported lazily so a missing one only disables its own provider.
Provider-specific param quirks (e.g. models that reject `temperature`) are handled
by retrying once with the offending parameter dropped.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import ModelCfg, env


@dataclass
class Result:
    text: str
    tokens_in: int
    tokens_out: int


class AnthropicProvider:
    def __init__(self, model: str):
        from anthropic import AsyncAnthropic

        self.model = model
        self.client = AsyncAnthropic(api_key=env("ANTHROPIC_API_KEY"), max_retries=4)

    async def complete(self, system, user, *, max_tokens, temperature, reasoning="off") -> Result:
        kwargs = dict(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        if temperature is not None:
            kwargs["temperature"] = temperature
        # reasoning "off" -> simply do not pass a `thinking` param (default is off).
        resp = await self.client.messages.create(**kwargs)
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        return Result(text, resp.usage.input_tokens, resp.usage.output_tokens)


class OpenAIProvider:
    def __init__(self, model: str):
        from openai import AsyncOpenAI

        self.model = model
        self.client = AsyncOpenAI(api_key=env("OPENAI_API_KEY"), max_retries=4)

    async def complete(self, system, user, *, max_tokens, temperature, reasoning="off") -> Result:
        from openai import BadRequestError

        base = dict(
            model=self.model,
            max_completion_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        kwargs = dict(base)
        if reasoning == "off":
            # gpt-5.x reasoning models expose "none" for no reasoning ("minimal"
            # is rejected by gpt-5.4-mini). If a model rejects "none", the
            # bad-param retry below drops it and falls back to the default.
            kwargs["reasoning_effort"] = "none"
        if temperature is not None:
            kwargs["temperature"] = temperature

        # Some gpt-5-family models reject `temperature` and/or `reasoning_effort`.
        # Drop unsupported params and retry rather than failing the whole run.
        for _ in range(3):
            try:
                resp = await self.client.chat.completions.create(**kwargs)
                break
            except BadRequestError as e:
                msg = str(e).lower()
                dropped = False
                for param in ("temperature", "reasoning_effort"):
                    if param in kwargs and param in msg:
                        kwargs.pop(param)
                        dropped = True
                if not dropped:
                    raise
        else:
            resp = await self.client.chat.completions.create(**base)

        text = resp.choices[0].message.content or ""
        usage = resp.usage
        return Result(text, usage.prompt_tokens, usage.completion_tokens)


class GoogleProvider:
    def __init__(self, model: str):
        from google import genai

        self.model = model
        self.client = genai.Client(api_key=env("GEMINI_API_KEY"))

    def _config(self, system, max_tokens, temperature, reasoning):
        from google.genai import types

        thinking = None
        if reasoning == "off":
            # Gemini 3 prefers `thinking_level`; older flashes accept `thinking_budget=0`.
            try:
                thinking = types.ThinkingConfig(thinking_level="low")
            except (TypeError, ValueError):
                try:
                    thinking = types.ThinkingConfig(thinking_budget=0)
                except (TypeError, ValueError):
                    thinking = None
        cfg_kwargs = dict(
            system_instruction=system,
            max_output_tokens=max_tokens,
            temperature=temperature if temperature is not None else 0.0,
        )
        if thinking is not None:
            cfg_kwargs["thinking_config"] = thinking
        return types.GenerateContentConfig(**cfg_kwargs)

    async def complete(self, system, user, *, max_tokens, temperature, reasoning="off") -> Result:
        try:
            config = self._config(system, max_tokens, temperature, reasoning)
            resp = await self.client.aio.models.generate_content(
                model=self.model, contents=user, config=config
            )
        except Exception:
            # Fall back to a config with no thinking control if the above is rejected.
            from google.genai import types

            config = types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=max_tokens,
                temperature=temperature if temperature is not None else 0.0,
            )
            resp = await self.client.aio.models.generate_content(
                model=self.model, contents=user, config=config
            )

        text = resp.text or ""
        um = resp.usage_metadata
        tin = getattr(um, "prompt_token_count", 0) or 0
        tout = (getattr(um, "candidates_token_count", 0) or 0) + (
            getattr(um, "thoughts_token_count", 0) or 0
        )
        return Result(text, tin, tout)


def make_provider(m: ModelCfg):
    if m.provider == "anthropic":
        return AnthropicProvider(m.model)
    if m.provider == "openai":
        return OpenAIProvider(m.model)
    if m.provider == "google":
        return GoogleProvider(m.model)
    raise ValueError(f"unknown provider {m.provider!r}")
