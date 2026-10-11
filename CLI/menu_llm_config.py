from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from CLI import ui
from CLI.llm_availability import check_llm_availability
from CLI.llm_config_store import (
    DEFAULT_CONFIG_PATH,
    build_llm_config,
    clear_semantic_scholar_api_key,
    delete_chat_model,
    delete_embedding_model,
    delete_provider,
    read_llm_config,
    set_chat_model,
    set_default_embedding_model,
    set_default_model,
    set_default_thinking,
    set_embedding_model,
    set_provider,
    set_semantic_scholar_api_key,
    validate_llm_config,
    write_llm_config,
)


THINKING_DEFAULT_CHOICES = ["enabled", "disabled", "none"]


@dataclass(frozen=True)
class ConfigStep:
    key: str
    ask: Callable[[dict[str, object]], object]


def llm_config_menu() -> None:
    if not _ensure_config_exists():
        return
    while True:
        ui.clear_screen()
        ui.header("tc-norm / LLM config")
        _print_config_summary(_load())
        choice = ui.ask_select_navigation(
            "Choose an action",
            [
                "Initialize llm.json",
                "View current config",
                "Manage providers",
                "Manage chat models",
                "Manage embedding models",
                "Manage defaults",
                "Manage services",
                "Check model availability",
                "Validate llm.json",
                "Back",
            ],
        )
        if isinstance(choice, ui.Navigation) or choice == "Back":
            return
        actions: dict[str, Callable[[], None]] = {
            "Initialize llm.json": initialize_llm_config,
            "View current config": view_current_config,
            "Manage providers": manage_providers,
            "Manage chat models": manage_chat_models,
            "Manage embedding models": manage_embedding_models,
            "Manage defaults": manage_defaults,
            "Manage services": manage_services,
            "Check model availability": check_model_availability,
            "Validate llm.json": validate_current_config,
        }
        actions[choice]()


def _ensure_config_exists() -> bool:
    if DEFAULT_CONFIG_PATH.exists():
        return True
    ui.clear_screen()
    ui.header("tc-norm / LLM config")
    ui.console.print("llm.json was not found.")
    choice = ui.ask_select_navigation(
        "Choose an action",
        ["Initialize llm.json", "Back"],
    )
    if isinstance(choice, ui.Navigation) or choice == "Back":
        return False
    initialize_llm_config()
    return DEFAULT_CONFIG_PATH.exists()


def _load() -> dict:
    return read_llm_config()


def _save(config: dict) -> None:
    write_llm_config(config)


def _print_refreshed_table(config: dict, section: str) -> None:
    if section == "providers":
        ui.console.print(ui.providers_table(config))
    elif section == "chat_models":
        ui.console.print(ui.models_table(config))
    elif section == "embedding_models":
        ui.console.print(ui.embedding_models_table(config))
    elif section == "defaults":
        defaults = config.get("defaults", {})
        ui.console.print(f"Default chat model: {defaults.get('model')}")
        ui.console.print(f"Default embedding model: {defaults.get('embedding_model')}")
        ui.console.print(f"Default thinking: {defaults.get('thinking')}")
    elif section == "services":
        ui.console.print(
            f"Semantic Scholar key: "
            f"{'set' if config.get('services', {}).get('semantic_scholar_api_key') else 'not set'}"
        )


def _print_config_summary(config: dict) -> None:
    ui.console.print(ui.providers_table(config))
    ui.console.print(ui.models_table(config))
    ui.console.print(ui.embedding_models_table(config))
    _print_refreshed_table(config, "defaults")
    _print_refreshed_table(config, "services")


def _run_config_wizard(
    build_steps: Callable[[dict[str, object]], list[ConfigStep]],
) -> dict[str, object] | None:
    values: dict[str, object] = {}
    index = 0
    while True:
        steps = build_steps(values)
        if index >= len(steps):
            return values

        answer = steps[index].ask(values)
        if answer is ui.Navigation.CANCEL:
            return None
        if answer is ui.Navigation.BACK:
            if index == 0:
                return None
            index -= 1
            continue

        key = steps[index].key
        old_answer = values.get(key, object())
        values[key] = answer
        if old_answer != answer:
            for later_step in steps[index + 1 :]:
                values.pop(later_step.key, None)
        index += 1


def _text_step(key: str, message: str, *, default: str = "", required: bool = False) -> ConfigStep:
    def ask(values: dict[str, object]) -> object:
        while True:
            answer = ui.ask_text_navigation(message, default=str(values.get(key, default)))
            if isinstance(answer, ui.Navigation) or answer or not required:
                return answer
            ui.console.print(f"[red]{message} is required.[/red]")

    return ConfigStep(key, ask)


def _select_step(
    key: str,
    message: str,
    choices: Sequence[str],
    *,
    default: str | None = None,
) -> ConfigStep:
    def ask(values: dict[str, object]) -> object:
        selected_default = values.get(key, default)
        return ui.ask_select_navigation(
            message,
            list(choices),
            default=str(selected_default) if selected_default is not None else None,
        )

    return ConfigStep(key, ask)


def _confirm_step(key: str, message: str, *, default: bool = True) -> ConfigStep:
    return ConfigStep(
        key,
        lambda values: ui.ask_confirm_navigation(
            message,
            default=bool(values.get(key, default)),
        ),
    )


def _select_value(message: str, choices: Sequence[str], *, default: str | None = None) -> str | None:
    if not choices:
        ui.console.print(f"[red]No choices available for {message}.[/red]")
        ui.pause()
        return None
    answer = ui.ask_select_navigation(message, list(choices), default=default)
    if isinstance(answer, ui.Navigation):
        return None
    return str(answer)


def _has_entries(config: dict, key: str, label: str) -> bool:
    if config.get(key):
        return True
    ui.console.print(f"[red]No {label} configured.[/red]")
    ui.pause()
    return False


def _default_thinking(config: dict) -> str:
    value = config.get("defaults", {}).get("thinking", "enabled")
    if isinstance(value, bool):
        return "enabled" if value else "disabled"
    return value if value in THINKING_DEFAULT_CHOICES else "enabled"


def view_current_config() -> None:
    ui.clear_screen()
    _print_config_summary(_load())
    ui.pause()


def initialize_llm_config() -> None:
    ui.clear_screen()
    ui.header("Initialize llm.json")
    if DEFAULT_CONFIG_PATH.exists():
        replace = ui.ask_confirm_navigation("Replace existing llm.json?", default=False)
        if isinstance(replace, ui.Navigation) or not replace:
            return

    values = _run_config_wizard(_build_initial_config_steps)
    if values is None:
        return

    embedding_model_id = None
    embedding_model_name = None
    if values.get("add_embedding_model") is True:
        embedding_model_id = str(values["embedding_model_id"])
        embedding_model_name = str(values["embedding_model_name"])

    semantic_scholar_key = str(values.get("semantic_scholar_api_key", "")).strip() or None
    config = build_llm_config(
        provider_id=str(values["provider_id"]),
        api_key=str(values["api_key"]),
        base_url=str(values["base_url"]),
        chat_model_id=str(values["chat_model_id"]),
        chat_model_name=str(values["chat_model_name"]),
        supports_thinking=bool(values["supports_thinking"]),
        default_thinking=str(values["default_thinking"]),
        stream=bool(values["stream"]),
        embedding_model_id=embedding_model_id,
        embedding_model_name=embedding_model_name,
        semantic_scholar_api_key=semantic_scholar_key,
    )
    _save(config)
    errors = validate_llm_config(config)
    if errors:
        ui.console.print("[yellow]llm.json was created, but it still has validation errors.[/yellow]")
        ui.print_errors(errors)
    else:
        ui.console.print("[green]llm.json was created.[/green]")
    ui.pause()


def _build_initial_config_steps(values: dict[str, object]) -> list[ConfigStep]:
    steps = [
        _text_step("provider_id", "Provider id", default="openai_compatible", required=True),
        _text_step("api_key", "API key", required=True),
        _text_step("base_url", "Base URL", required=True),
        _text_step("chat_model_id", "Chat model id", default="default_chat", required=True),
        _text_step("chat_model_name", "API chat model name", required=True),
        _confirm_step("supports_thinking", "Does this chat model support thinking?", default=False),
    ]
    if values.get("supports_thinking") is True:
        steps.append(
            _select_step(
                "default_thinking",
                "Default thinking",
                THINKING_DEFAULT_CHOICES,
                default="enabled",
            )
        )
    else:
        steps.append(
            _select_step(
                "default_thinking",
                "Default thinking",
                ["disabled", "none"],
                default="disabled",
            )
        )
    steps.extend(
        [
            _confirm_step("stream", "Stream chat responses?", default=True),
            _confirm_step("add_embedding_model", "Add an embedding model now?", default=False),
        ]
    )
    if values.get("add_embedding_model") is True:
        steps.extend(
            [
                _text_step(
                    "embedding_model_id",
                    "Embedding model id",
                    default="default_embedding",
                    required=True,
                ),
                _text_step("embedding_model_name", "API embedding model name", required=True),
            ]
        )
    steps.append(
        _text_step(
            "semantic_scholar_api_key",
            "Semantic Scholar API key (optional)",
        )
    )
    return steps


def validate_current_config() -> None:
    ui.clear_screen()
    errors = validate_llm_config(_load())
    if errors:
        ui.console.print("[red]llm.json has validation errors.[/red]")
        ui.print_errors(errors)
    else:
        ui.console.print("[green]llm.json is valid.[/green]")
    ui.pause()


def check_model_availability() -> None:
    ui.clear_screen()
    config = _load()
    errors = validate_llm_config(config)
    if errors:
        ui.console.print("[red]llm.json has validation errors. Fix these before checking availability.[/red]")
        ui.print_errors(errors)
        ui.pause()
        return

    ui.console.print("[cyan]Checking configured chat and embedding models...[/cyan]")
    results = check_llm_availability(config)
    ui.console.print(ui.availability_table(results))
    if all(result.ok for result in results):
        ui.console.print("[green]All configured models are reachable.[/green]")
    else:
        ui.console.print("[red]Some configured models failed the availability probe.[/red]")
    ui.pause()


def manage_providers() -> None:
    while True:
        ui.clear_screen()
        config = _load()
        _print_refreshed_table(config, "providers")
        choice = ui.ask_select_navigation(
            "Manage providers",
            ["Add provider", "Edit provider", "Delete provider", "Back"],
        )
        if isinstance(choice, ui.Navigation) or choice == "Back":
            return
        providers = list(config.get("providers", {}).keys())

        if choice == "Delete provider":
            provider_id = _select_value("Provider", providers)
            if provider_id is None:
                continue
            try:
                delete_provider(config, provider_id)
            except ValueError as exc:
                ui.console.print(f"[red]{exc}[/red]")
                ui.pause()
                continue
            _save(config)
            continue

        if choice == "Edit provider" and not _has_entries(config, "providers", "providers"):
            continue

        values = _run_config_wizard(
            lambda current_values: _build_provider_steps(str(choice), config, current_values)
        )
        if values is None:
            continue
        set_provider(
            config,
            str(values["provider_id"]),
            api_key=str(values["api_key"]),
            base_url=str(values["base_url"]),
        )
        _save(config)


def _build_provider_steps(action: str, config: dict, values: dict[str, object]) -> list[ConfigStep]:
    if action == "Add provider":
        steps = [_text_step("provider_id", "Provider id", required=True)]
    else:
        steps = [_select_step("provider_id", "Provider", list(config.get("providers", {}).keys()))]

    provider_id = str(values.get("provider_id", ""))
    current = config.get("providers", {}).get(provider_id, {})
    steps.extend(
        [
            _text_step("api_key", "API key", default=str(current.get("api_key", ""))),
            _text_step("base_url", "Base URL", default=str(current.get("base_url", ""))),
        ]
    )
    return steps


def manage_chat_models() -> None:
    while True:
        ui.clear_screen()
        config = _load()
        _print_refreshed_table(config, "chat_models")
        choice = ui.ask_select_navigation(
            "Manage chat models",
            ["Add chat model", "Edit chat model", "Delete chat model", "Back"],
        )
        if isinstance(choice, ui.Navigation) or choice == "Back":
            return
        model_ids = list(config.get("models", {}).keys())

        if choice == "Delete chat model":
            model_id = _select_value("Chat model", model_ids)
            if model_id is None:
                continue
            try:
                delete_chat_model(config, model_id)
            except ValueError as exc:
                ui.console.print(f"[red]{exc}[/red]")
                ui.pause()
                continue
            _save(config)
            continue

        if choice == "Edit chat model" and not _has_entries(config, "models", "chat models"):
            continue
        if choice == "Add chat model" and not _has_entries(config, "providers", "providers"):
            continue

        values = _run_config_wizard(
            lambda current_values: _build_chat_model_steps(str(choice), config, current_values)
        )
        if values is None:
            continue
        set_chat_model(
            config,
            str(values["model_id"]),
            provider=str(values["provider"]),
            model_name=str(values["model_name"]),
            supports_thinking=bool(values["supports_thinking"]),
        )
        _save(config)


def _build_chat_model_steps(action: str, config: dict, values: dict[str, object]) -> list[ConfigStep]:
    if action == "Add chat model":
        steps = [_text_step("model_id", "Chat model id", required=True)]
    else:
        steps = [_select_step("model_id", "Chat model", list(config.get("models", {}).keys()))]

    model_id = str(values.get("model_id", ""))
    current = config.get("models", {}).get(model_id, {})
    steps.extend(
        [
            _select_step(
                "provider",
                "Provider",
                list(config.get("providers", {}).keys()),
                default=current.get("provider"),
            ),
            _text_step("model_name", "API model name", default=str(current.get("model_name", ""))),
            _confirm_step(
                "supports_thinking",
                "Supports thinking?",
                default=bool(current.get("supports_thinking", True)),
            ),
        ]
    )
    return steps


def manage_embedding_models() -> None:
    while True:
        ui.clear_screen()
        config = _load()
        _print_refreshed_table(config, "embedding_models")
        choice = ui.ask_select_navigation(
            "Manage embedding models",
            ["Add embedding model", "Edit embedding model", "Delete embedding model", "Back"],
        )
        if isinstance(choice, ui.Navigation) or choice == "Back":
            return
        model_ids = list(config.get("embedding_models", {}).keys())

        if choice == "Delete embedding model":
            model_id = _select_value("Embedding model", model_ids)
            if model_id is None:
                continue
            try:
                delete_embedding_model(config, model_id)
            except ValueError as exc:
                ui.console.print(f"[red]{exc}[/red]")
                ui.pause()
                continue
            _save(config)
            continue

        if choice == "Edit embedding model" and not _has_entries(config, "embedding_models", "embedding models"):
            continue
        if choice == "Add embedding model" and not _has_entries(config, "providers", "providers"):
            continue

        values = _run_config_wizard(
            lambda current_values: _build_embedding_model_steps(str(choice), config, current_values)
        )
        if values is None:
            continue
        set_embedding_model(
            config,
            str(values["model_id"]),
            provider=str(values["provider"]),
            model_name=str(values["model_name"]),
        )
        _save(config)


def _build_embedding_model_steps(action: str, config: dict, values: dict[str, object]) -> list[ConfigStep]:
    if action == "Add embedding model":
        steps = [_text_step("model_id", "Embedding model id", required=True)]
    else:
        steps = [
            _select_step(
                "model_id",
                "Embedding model",
                list(config.get("embedding_models", {}).keys()),
            )
        ]

    model_id = str(values.get("model_id", ""))
    current = config.get("embedding_models", {}).get(model_id, {})
    steps.extend(
        [
            _select_step(
                "provider",
                "Provider",
                list(config.get("providers", {}).keys()),
                default=current.get("provider"),
            ),
            _text_step("model_name", "API model name", default=str(current.get("model_name", ""))),
        ]
    )
    return steps


def manage_defaults() -> None:
    while True:
        ui.clear_screen()
        config = _load()
        _print_refreshed_table(config, "defaults")
        choice = ui.ask_select_navigation(
            "Manage defaults",
            ["Set default chat model", "Set default embedding model", "Set default thinking", "Back"],
        )
        if isinstance(choice, ui.Navigation) or choice == "Back":
            return
        if choice == "Set default chat model":
            model_id = _select_value(
                "Default chat model",
                list(config.get("models", {}).keys()),
                default=config.get("defaults", {}).get("model"),
            )
            if model_id is None:
                continue
            set_default_model(config, model_id)
        elif choice == "Set default embedding model":
            model_id = _select_value(
                "Default embedding model",
                list(config.get("embedding_models", {}).keys()),
                default=config.get("defaults", {}).get("embedding_model"),
            )
            if model_id is None:
                continue
            set_default_embedding_model(config, model_id)
        elif choice == "Set default thinking":
            thinking = _select_value(
                "Default thinking",
                THINKING_DEFAULT_CHOICES,
                default=_default_thinking(config),
            )
            if thinking is None:
                continue
            set_default_thinking(config, thinking)
        _save(config)


def manage_services() -> None:
    while True:
        ui.clear_screen()
        config = _load()
        _print_refreshed_table(config, "services")
        choice = ui.ask_select_navigation(
            "Manage services",
            ["Set Semantic Scholar API key", "Clear Semantic Scholar API key", "Back"],
        )
        if isinstance(choice, ui.Navigation) or choice == "Back":
            return
        if choice == "Set Semantic Scholar API key":
            api_key = ui.ask_text_navigation("Semantic Scholar API key")
            if isinstance(api_key, ui.Navigation):
                continue
            set_semantic_scholar_api_key(config, api_key)
        elif choice == "Clear Semantic Scholar API key":
            clear_semantic_scholar_api_key(config)
        _save(config)
