from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from typing import Any

import questionary
from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings
from prompt_toolkit.keys import Keys
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


console = Console()


class Navigation(Enum):
    BACK = "back"
    CANCEL = "cancel"


def _required_answer(value: Any) -> Any:
    if value is None:
        raise KeyboardInterrupt
    return value


def _ask_navigation(question: Any) -> Any:
    navigation_bindings = KeyBindings()

    @navigation_bindings.add(Keys.Escape, Keys.Escape, eager=True)
    def cancel(event) -> None:
        event.app.exit(result=Navigation.CANCEL, style="class:aborting")

    @navigation_bindings.add(Keys.Escape)
    def back(event) -> None:
        event.app.exit(result=Navigation.BACK, style="class:aborting")

    question.application.key_bindings = merge_key_bindings(
        [question.application.key_bindings, navigation_bindings]
    )
    question.application.ttimeoutlen = 0.05
    question.application.timeoutlen = 0.3
    return _required_answer(question.ask())


def header(title: str) -> None:
    console.print(Panel(title, style="bold cyan"))


def clear_screen() -> None:
    console.clear()


def ask_text(message: str, *, default: str = "") -> str:
    return str(_required_answer(questionary.text(message, default=default).ask())).strip()


def ask_float(message: str, *, default: float = 0.0) -> float:
    while True:
        value = ask_text(message, default=f"{default:g}")
        try:
            return float(value)
        except ValueError:
            console.print("[red]Please enter a number.[/red]")


def ask_confirm(message: str, *, default: bool = True) -> bool:
    return bool(_required_answer(questionary.confirm(message, default=default).ask()))


def ask_confirm_navigation(
    message: str,
    *,
    default: bool = True,
    **kwargs: Any,
) -> bool | Navigation:
    result = _ask_navigation(questionary.confirm(message, default=default, **kwargs))
    return result if isinstance(result, Navigation) else bool(result)


def ask_select(message: str, choices: Sequence[str], *, default: str | None = None) -> str:
    if not choices:
        raise ValueError(f"No choices available for: {message}")
    return str(_required_answer(questionary.select(message, choices=list(choices), default=default).ask()))


def ask_select_navigation(
    message: str,
    choices: Sequence[Any],
    *,
    default: str | None = None,
    **kwargs: Any,
) -> str | Navigation:
    if not choices:
        raise ValueError(f"No choices available for: {message}")
    question = questionary.select(message, choices=list(choices), default=default, **kwargs)
    result = _ask_navigation(question)
    return result if isinstance(result, Navigation) else str(result)


def ask_text_navigation(
    message: str,
    *,
    default: str = "",
    **kwargs: Any,
) -> str | Navigation:
    result = _ask_navigation(questionary.text(message, default=default, **kwargs))
    return result if isinstance(result, Navigation) else str(result).strip()


def ask_float_navigation(
    message: str,
    *,
    default: float = 0.0,
    **kwargs: Any,
) -> float | Navigation:
    while True:
        result = ask_text_navigation(message, default=f"{default:g}", **kwargs)
        if isinstance(result, Navigation):
            return result
        try:
            return float(result)
        except ValueError:
            console.print("[red]Please enter a number.[/red]")


def ask_int_navigation(
    message: str,
    *,
    default: int = 0,
    **kwargs: Any,
) -> int | Navigation:
    while True:
        result = ask_text_navigation(message, default=str(default), **kwargs)
        if isinstance(result, Navigation):
            return result
        try:
            return int(result)
        except ValueError:
            console.print("[red]Please enter an integer.[/red]")


def ask_checkbox_navigation(
    message: str,
    choices: Sequence[Any],
    **kwargs: Any,
) -> list[str] | Navigation:
    if not choices:
        raise ValueError(f"No choices available for: {message}")
    result = _ask_navigation(questionary.checkbox(message, choices=list(choices), **kwargs))
    return result if isinstance(result, Navigation) else list(result)


def ask_checkbox(message: str, choices: Sequence[str]) -> list[str]:
    if not choices:
        raise ValueError(f"No choices available for: {message}")
    return list(_required_answer(questionary.checkbox(message, choices=list(choices)).ask()))


def pause() -> None:
    questionary.press_any_key_to_continue("Press any key to continue...").ask()


def print_errors(errors: Sequence[str]) -> None:
    for error in errors:
        console.print(f"[red]- {error}[/red]")


def print_command_preview(commands: list[list[str]], formatter) -> None:
    rendered = "\n".join(formatter(command) for command in commands)
    console.print(Panel(rendered, title="Command preview", border_style="green"))


def models_table(config: dict[str, Any]) -> Table:
    table = Table(title="Chat models")
    table.add_column("ID")
    table.add_column("Provider")
    table.add_column("Model name")
    table.add_column("Thinking")
    default_model = config.get("defaults", {}).get("model")
    for model_id, model in config.get("models", {}).items():
        label = f"{model_id} *" if model_id == default_model else model_id
        table.add_row(
            label,
            str(model.get("provider", "")),
            str(model.get("model_name", "")),
            "yes" if model.get("supports_thinking") else "no",
        )
    return table


def providers_table(config: dict[str, Any]) -> Table:
    table = Table(title="Providers")
    table.add_column("ID")
    table.add_column("Base URL")
    table.add_column("API key")
    for provider_id, provider in config.get("providers", {}).items():
        api_key = str(provider.get("api_key", ""))
        table.add_row(provider_id, str(provider.get("base_url", "")), "set" if api_key else "missing")
    return table


def embedding_models_table(config: dict[str, Any]) -> Table:
    table = Table(title="Embedding models")
    table.add_column("ID")
    table.add_column("Provider")
    table.add_column("Model name")
    default_model = config.get("defaults", {}).get("embedding_model")
    for model_id, model in config.get("embedding_models", {}).items():
        label = f"{model_id} *" if model_id == default_model else model_id
        table.add_row(label, str(model.get("provider", "")), str(model.get("model_name", "")))
    return table


def availability_table(results: Sequence[Any]) -> Table:
    table = Table(title="Model availability")
    table.add_column("Type")
    table.add_column("ID")
    table.add_column("Status")
    table.add_column("Message")
    for result in results:
        status = "[green]ok[/green]" if result.ok else "[red]failed[/red]"
        table.add_row(result.kind, result.model_id, status, result.message)
    return table
