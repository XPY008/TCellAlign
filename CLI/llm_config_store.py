from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "llm.json"
THINKING_CHOICES = {"enabled", "disabled", "none"}


def read_llm_config(path: Path | str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{config_path} must contain a JSON object.")
    return payload


def write_llm_config(config: dict[str, Any], path: Path | str = DEFAULT_CONFIG_PATH) -> None:
    config_path = Path(path)
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_llm_config(
    *,
    provider_id: str,
    api_key: str,
    base_url: str,
    chat_model_id: str,
    chat_model_name: str,
    supports_thinking: bool,
    default_thinking: str | bool,
    stream: bool = True,
    embedding_model_id: str | None = None,
    embedding_model_name: str | None = None,
    semantic_scholar_api_key: str | None = None,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "defaults": {
            "model": chat_model_id,
            "thinking": default_thinking,
            "stream": stream,
        },
        "providers": {
            provider_id: {
                "api_key": api_key,
                "base_url": base_url,
            }
        },
        "models": {
            chat_model_id: {
                "provider": provider_id,
                "model_name": chat_model_name,
                "supports_thinking": supports_thinking,
            }
        },
        "embedding_models": {},
    }
    if embedding_model_id and embedding_model_name:
        config["defaults"]["embedding_model"] = embedding_model_id
        config["embedding_models"][embedding_model_id] = {
            "provider": provider_id,
            "model_name": embedding_model_name,
        }
    if semantic_scholar_api_key:
        config["services"] = {
            "semantic_scholar_api_key": semantic_scholar_api_key,
        }
    return config


def _object(config: dict[str, Any], key: str, errors: list[str]) -> dict[str, Any]:
    value = config.get(key)
    if isinstance(value, dict):
        return value
    errors.append(f"{key} must be a JSON object.")
    return {}


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_llm_config(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    defaults = _object(config, "defaults", errors)
    providers = _object(config, "providers", errors)
    models = _object(config, "models", errors)
    embedding_models = _object(config, "embedding_models", errors)

    default_model = defaults.get("model")
    if not _non_empty_string(default_model) or default_model not in models:
        errors.append("defaults.model must name an existing chat model.")

    default_embedding = defaults.get("embedding_model")
    if default_embedding is not None and (
        not _non_empty_string(default_embedding) or default_embedding not in embedding_models
    ):
        errors.append("defaults.embedding_model must name an existing embedding model.")

    if "thinking" in defaults:
        thinking = defaults["thinking"]
        if not isinstance(thinking, bool) and thinking not in THINKING_CHOICES:
            errors.append("defaults.thinking must be true, false, enabled, disabled, or none.")

    if "stream" in defaults and not isinstance(defaults["stream"], bool):
        errors.append("defaults.stream must be a JSON boolean.")

    for provider_id, provider in providers.items():
        if not isinstance(provider, dict):
            errors.append(f"providers.{provider_id} must be a JSON object.")
            continue
        if not _non_empty_string(provider.get("api_key")):
            errors.append(f"providers.{provider_id}.api_key is required.")
        if not _non_empty_string(provider.get("base_url")):
            errors.append(f"providers.{provider_id}.base_url is required.")

    for model_id, model in models.items():
        if not isinstance(model, dict):
            errors.append(f"models.{model_id} must be a JSON object.")
            continue
        provider = model.get("provider")
        if not _non_empty_string(provider) or provider not in providers:
            errors.append(f"models.{model_id}.provider must name an existing provider.")
        if not _non_empty_string(model.get("model_name")):
            errors.append(f"models.{model_id}.model_name is required.")
        if not isinstance(model.get("supports_thinking"), bool):
            errors.append(f"models.{model_id}.supports_thinking must be a JSON boolean.")

    for embedding_id, embedding_model in embedding_models.items():
        if not isinstance(embedding_model, dict):
            errors.append(f"embedding_models.{embedding_id} must be a JSON object.")
            continue
        provider = embedding_model.get("provider")
        if not _non_empty_string(provider) or provider not in providers:
            errors.append(f"embedding_models.{embedding_id}.provider must name an existing provider.")
        if not _non_empty_string(embedding_model.get("model_name")):
            errors.append(f"embedding_models.{embedding_id}.model_name is required.")

    return errors


def set_provider(config: dict[str, Any], provider_id: str, *, api_key: str, base_url: str) -> None:
    config.setdefault("providers", {})[provider_id] = {
        "api_key": api_key,
        "base_url": base_url,
    }


def delete_provider(config: dict[str, Any], provider_id: str) -> None:
    for model_id, model in config.get("models", {}).items():
        if isinstance(model, dict) and model.get("provider") == provider_id:
            raise ValueError(f"Provider {provider_id!r} is used by model {model_id!r}.")
    for embedding_id, embedding_model in config.get("embedding_models", {}).items():
        if isinstance(embedding_model, dict) and embedding_model.get("provider") == provider_id:
            raise ValueError(f"Provider {provider_id!r} is used by embedding model {embedding_id!r}.")
    config.get("providers", {}).pop(provider_id, None)


def set_chat_model(
    config: dict[str, Any],
    model_id: str,
    *,
    provider: str,
    model_name: str,
    supports_thinking: bool,
) -> None:
    if provider not in config.get("providers", {}):
        raise ValueError(f"Unknown provider: {provider}")
    config.setdefault("models", {})[model_id] = {
        "provider": provider,
        "model_name": model_name,
        "supports_thinking": supports_thinking,
    }


def delete_chat_model(config: dict[str, Any], model_id: str) -> None:
    if config.get("defaults", {}).get("model") == model_id:
        raise ValueError(f"Cannot delete the default chat model {model_id!r}.")
    config.get("models", {}).pop(model_id, None)


def set_embedding_model(
    config: dict[str, Any],
    embedding_model_id: str,
    *,
    provider: str,
    model_name: str,
) -> None:
    if provider not in config.get("providers", {}):
        raise ValueError(f"Unknown provider: {provider}")
    config.setdefault("embedding_models", {})[embedding_model_id] = {
        "provider": provider,
        "model_name": model_name,
    }


def delete_embedding_model(config: dict[str, Any], embedding_model_id: str) -> None:
    if config.get("defaults", {}).get("embedding_model") == embedding_model_id:
        raise ValueError(f"Cannot delete the default embedding model {embedding_model_id!r}.")
    config.get("embedding_models", {}).pop(embedding_model_id, None)


def set_default_model(config: dict[str, Any], model_id: str) -> None:
    if model_id not in config.get("models", {}):
        raise ValueError(f"Unknown chat model: {model_id}")
    config.setdefault("defaults", {})["model"] = model_id


def set_default_embedding_model(config: dict[str, Any], embedding_model_id: str) -> None:
    if embedding_model_id not in config.get("embedding_models", {}):
        raise ValueError(f"Unknown embedding model: {embedding_model_id}")
    config.setdefault("defaults", {})["embedding_model"] = embedding_model_id


def set_default_thinking(config: dict[str, Any], thinking: str | bool) -> None:
    if not isinstance(thinking, bool) and thinking not in THINKING_CHOICES:
        raise ValueError("thinking must be true, false, enabled, disabled, or none.")
    config.setdefault("defaults", {})["thinking"] = thinking


def set_semantic_scholar_api_key(config: dict[str, Any], api_key: str) -> None:
    config.setdefault("services", {})["semantic_scholar_api_key"] = api_key


def clear_semantic_scholar_api_key(config: dict[str, Any]) -> None:
    config.setdefault("services", {}).pop("semantic_scholar_api_key", None)
