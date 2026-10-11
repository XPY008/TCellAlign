from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Mapping


class ConfigError(RuntimeError):
    """Raised when runtime configuration is missing or invalid."""


@dataclass(frozen=True)
class RuntimeConfig:
    model_id: str
    provider_name: str
    api_key: str
    base_url: str
    model_name: str
    thinking: str = "enabled"
    stream: bool = True
    supports_thinking: bool = True


@dataclass(frozen=True)
class EmbeddingConfig:
    embedding_model_id: str
    provider_name: str
    api_key: str
    base_url: str
    model_name: str


def default_llm_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "llm.json"


def _read_config(config_path: Path) -> dict[str, Any] | None:
    if not config_path.exists():
        return None
    try:
        raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{config_path} is not valid JSON.") from exc
    if not isinstance(raw_config, dict):
        raise ConfigError(f"{config_path} must contain a JSON object.")
    return raw_config


def _required_object(value: Any, *, field_name: str, config_path: Path) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    raise ConfigError(f"{config_path} field {field_name!r} must be a JSON object.")


def _required_string(value: Any, *, field_name: str, config_path: Path) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ConfigError(f"{config_path} is missing required field {field_name!r}.")


def _optional_bool(
    value: Any,
    *,
    field_name: str,
    config_path: Path,
    default: bool,
) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise ConfigError(f"{config_path} field {field_name!r} must be a JSON boolean.")


def _normalize_thinking(
    value: Any,
    *,
    field_name: str,
    config_path: Path,
    default: str,
) -> str:
    if value is None:
        return default
    if isinstance(value, bool):
        return "enabled" if value else "disabled"
    if isinstance(value, str) and value in {"enabled", "disabled", "none"}:
        return value
    raise ConfigError(
        f"{config_path} field {field_name!r} must be true, false, "
        "'enabled', 'disabled', or 'none'."
    )


def _provider_config(
    *,
    raw_config: Mapping[str, Any],
    provider_name: str,
    config_path: Path,
) -> dict[str, str]:
    providers = _required_object(raw_config.get("providers"), field_name="providers", config_path=config_path)
    raw_provider = providers.get(provider_name)
    if not isinstance(raw_provider, dict):
        raise ConfigError(f"{config_path} does not define provider {provider_name!r}.")
    return {
        "api_key": _required_string(
            raw_provider.get("api_key"),
            field_name=f"providers.{provider_name}.api_key",
            config_path=config_path,
        ),
        "base_url": _required_string(
            raw_provider.get("base_url"),
            field_name=f"providers.{provider_name}.base_url",
            config_path=config_path,
        ),
    }


def load_runtime_config(
    environ: Mapping[str, str] | None = None,
    config_path: str | Path | None = None,
    model_id: str | None = None,
    thinking: str | bool | None = None,
    stream: bool | None = None,
) -> RuntimeConfig:
    env = os.environ if environ is None else environ
    llm_config_path = default_llm_config_path() if config_path is None else Path(config_path)
    raw_config = _read_config(llm_config_path)

    if raw_config is None:
        api_key = env.get("OPENAI_API_KEY")
        base_url = env.get("OPENAI_BASE_URL")
        model = env.get("OPENAI_MODEL")
        if not api_key or not base_url or not model:
            raise ConfigError(
                f"{llm_config_path} was not found. Set OPENAI_API_KEY, OPENAI_BASE_URL, and OPENAI_MODEL."
            )
        resolved_thinking = _normalize_thinking(
            thinking,
            field_name="thinking",
            config_path=llm_config_path,
            default="enabled",
        )
        return RuntimeConfig(
            model_id="environment",
            provider_name="environment",
            api_key=api_key,
            base_url=base_url,
            model_name=model,
            thinking=resolved_thinking,
            stream=True if stream is None else stream,
        )

    defaults = _required_object(raw_config.get("defaults"), field_name="defaults", config_path=llm_config_path)
    models = _required_object(
        raw_config.get("models"),
        field_name="models",
        config_path=llm_config_path,
    )
    selected_model = model_id or _required_string(
        defaults.get("model"),
        field_name="defaults.model",
        config_path=llm_config_path,
    )
    raw_model = models.get(selected_model)
    if not isinstance(raw_model, dict):
        raise ConfigError(f"{llm_config_path} does not define model {selected_model!r}.")

    provider_name = _required_string(
        raw_model.get("provider"),
        field_name=f"models.{selected_model}.provider",
        config_path=llm_config_path,
    )
    provider = _provider_config(
        raw_config=raw_config,
        provider_name=provider_name,
        config_path=llm_config_path,
    )
    supports_thinking = _optional_bool(
        raw_model.get("supports_thinking"),
        field_name=f"models.{selected_model}.supports_thinking",
        config_path=llm_config_path,
        default=True,
    )
    default_thinking = _normalize_thinking(
        defaults.get("thinking"),
        field_name="defaults.thinking",
        config_path=llm_config_path,
        default="enabled",
    )
    resolved_thinking = _normalize_thinking(
        thinking,
        field_name="thinking",
        config_path=llm_config_path,
        default=default_thinking,
    )
    if resolved_thinking == "enabled" and not supports_thinking:
        raise ConfigError(f"{llm_config_path} model {selected_model!r} does not support thinking.")

    default_stream = _optional_bool(
        defaults.get("stream"),
        field_name="defaults.stream",
        config_path=llm_config_path,
        default=True,
    )
    return RuntimeConfig(
        model_id=selected_model,
        provider_name=provider_name,
        api_key=provider["api_key"],
        base_url=provider["base_url"],
        model_name=_required_string(
            raw_model.get("model_name"),
            field_name=f"models.{selected_model}.model_name",
            config_path=llm_config_path,
        ),
        thinking=resolved_thinking,
        stream=default_stream if stream is None else stream,
        supports_thinking=supports_thinking,
    )


def load_embedding_config(
    *,
    environ: Mapping[str, str] | None = None,
    config_path: str | Path | None = None,
    embedding_model_id: str | None = None,
    embedding_model_name: str | None = None,
) -> EmbeddingConfig:
    env = os.environ if environ is None else environ
    llm_config_path = default_llm_config_path() if config_path is None else Path(config_path)
    raw_config = _read_config(llm_config_path)

    if raw_config is None:
        api_key = env.get("OPENAI_API_KEY")
        base_url = env.get("OPENAI_BASE_URL")
        model_name = (
            embedding_model_name
            or env.get("QWEN_EMBEDDING_MODEL")
            or env.get("EMBEDDING_MODEL")
            or "Qwen/Qwen3-Embedding-8B"
        )
        if not api_key or not base_url or not model_name:
            raise ConfigError(
                f"{llm_config_path} was not found. Set OPENAI_API_KEY, OPENAI_BASE_URL, and "
                "QWEN_EMBEDDING_MODEL, or pass --llm-config and --embedding-model-name."
            )
        return EmbeddingConfig(
            embedding_model_id="environment",
            provider_name="environment",
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
        )

    defaults = _required_object(raw_config.get("defaults"), field_name="defaults", config_path=llm_config_path)
    embedding_models = _required_object(
        raw_config.get("embedding_models"),
        field_name="embedding_models",
        config_path=llm_config_path,
    )
    selected_embedding_model = embedding_model_id or _required_string(
        defaults.get("embedding_model"),
        field_name="defaults.embedding_model",
        config_path=llm_config_path,
    )
    raw_embedding_model = embedding_models.get(selected_embedding_model)
    if not isinstance(raw_embedding_model, dict):
        raise ConfigError(f"{llm_config_path} does not define embedding model {selected_embedding_model!r}.")

    provider_name = _required_string(
        raw_embedding_model.get("provider"),
        field_name=f"embedding_models.{selected_embedding_model}.provider",
        config_path=llm_config_path,
    )
    provider = _provider_config(
        raw_config=raw_config,
        provider_name=provider_name,
        config_path=llm_config_path,
    )
    model_name = embedding_model_name or _required_string(
        raw_embedding_model.get("model_name"),
        field_name=f"embedding_models.{selected_embedding_model}.model_name",
        config_path=llm_config_path,
    )
    return EmbeddingConfig(
        embedding_model_id=selected_embedding_model,
        provider_name=provider_name,
        api_key=provider["api_key"],
        base_url=provider["base_url"],
        model_name=model_name,
    )


def load_semantic_scholar_api_key(
    *,
    cli_api_key: str | None = None,
    environ: Mapping[str, str] | None = None,
    config_path: str | Path | None = None,
) -> str | None:
    if cli_api_key and cli_api_key.strip():
        return cli_api_key.strip()
    env = os.environ if environ is None else environ
    env_api_key = env.get("SEMANTIC_SCHOLAR_API_KEY")
    if env_api_key and env_api_key.strip():
        return env_api_key.strip()

    llm_config_path = default_llm_config_path() if config_path is None else Path(config_path)
    raw_config = _read_config(llm_config_path)
    if raw_config is None:
        return None
    raw_services = raw_config.get("services")
    if raw_services is None:
        return None
    services = _required_object(raw_services, field_name="services", config_path=llm_config_path)
    value = services.get("semantic_scholar_api_key")
    if value is None:
        return None
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ConfigError(f"{llm_config_path} field 'services.semantic_scholar_api_key' must be a non-empty string.")
