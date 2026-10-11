from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable

from openai import OpenAI


@dataclass(frozen=True)
class AvailabilityResult:
    kind: str
    model_id: str
    ok: bool
    message: str


ClientFactory = Callable[..., Any]


def _openai_client_factory(*, api_key: str, base_url: str, timeout_seconds: float) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds)


def _sanitize_error(error: Exception, secrets: list[str]) -> str:
    message = str(error) or error.__class__.__name__
    for secret in secrets:
        if secret:
            message = message.replace(secret, "[redacted]")
    return re.sub(r"sk-[A-Za-z0-9_-]+", "[redacted]", message)


def _provider(config: dict, provider_id: str) -> dict:
    providers = config.get("providers", {})
    provider = providers[provider_id]
    if not isinstance(provider, dict):
        raise ValueError(f"Provider {provider_id!r} is not a JSON object.")
    return provider


def check_chat_model(
    config: dict,
    model_id: str,
    *,
    client_factory: ClientFactory = _openai_client_factory,
    timeout_seconds: float = 20.0,
) -> AvailabilityResult:
    model = config.get("models", {})[model_id]
    provider = _provider(config, str(model["provider"]))
    api_key = str(provider["api_key"])
    try:
        client = client_factory(
            api_key=api_key,
            base_url=str(provider["base_url"]),
            timeout_seconds=timeout_seconds,
        )
        client.chat.completions.create(
            model=str(model["model_name"]),
            messages=[{"role": "user", "content": "Reply with OK."}],
            temperature=0,
            max_tokens=16,
        )
    except Exception as exc:
        return AvailabilityResult(
            kind="chat",
            model_id=model_id,
            ok=False,
            message=_sanitize_error(exc, [api_key]),
        )
    return AvailabilityResult(kind="chat", model_id=model_id, ok=True, message="ok")


def check_embedding_model(
    config: dict,
    model_id: str,
    *,
    client_factory: ClientFactory = _openai_client_factory,
    timeout_seconds: float = 20.0,
) -> AvailabilityResult:
    model = config.get("embedding_models", {})[model_id]
    provider = _provider(config, str(model["provider"]))
    api_key = str(provider["api_key"])
    try:
        client = client_factory(
            api_key=api_key,
            base_url=str(provider["base_url"]),
            timeout_seconds=timeout_seconds,
        )
        client.embeddings.create(
            model=str(model["model_name"]),
            input="tc-norm availability check",
        )
    except Exception as exc:
        return AvailabilityResult(
            kind="embedding",
            model_id=model_id,
            ok=False,
            message=_sanitize_error(exc, [api_key]),
        )
    return AvailabilityResult(kind="embedding", model_id=model_id, ok=True, message="ok")


def check_llm_availability(
    config: dict,
    *,
    client_factory: ClientFactory = _openai_client_factory,
    timeout_seconds: float = 20.0,
) -> list[AvailabilityResult]:
    results: list[AvailabilityResult] = []
    for model_id in sorted(config.get("models", {}).keys()):
        results.append(
            check_chat_model(
                config,
                model_id,
                client_factory=client_factory,
                timeout_seconds=timeout_seconds,
            )
        )
    for model_id in sorted(config.get("embedding_models", {}).keys()):
        results.append(
            check_embedding_model(
                config,
                model_id,
                client_factory=client_factory,
                timeout_seconds=timeout_seconds,
            )
        )
    return results
