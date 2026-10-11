from __future__ import annotations

from typing import Any

from openai import OpenAI

from workflow.common.llm_config import RuntimeConfig


def create_openai_client(config: RuntimeConfig, *, timeout_seconds: float) -> OpenAI:
    return OpenAI(api_key=config.api_key, base_url=config.base_url, timeout=timeout_seconds)


def apply_chat_options(
    request_kwargs: dict[str, Any],
    *,
    temperature: float | None,
    thinking: str | None,
    stream: bool,
    provider_name: str | None = None,
    include_usage: bool = False,
    max_tokens: int | None = None,
) -> None:
    if temperature is not None:
        request_kwargs["temperature"] = temperature
    if thinking is not None and thinking != "none":
        if provider_name == "openrouter":
            request_kwargs["extra_body"] = {
                "reasoning": {"enabled": True}
                if thinking == "enabled"
                else {"effort": "none"}
            }
        else:
            request_kwargs["extra_body"] = {"thinking": {"type": thinking}}
    if max_tokens is not None:
        request_kwargs["max_tokens"] = max_tokens
    if stream:
        request_kwargs["stream"] = True
        if include_usage:
            request_kwargs["stream_options"] = {"include_usage": True}
