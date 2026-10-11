from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import questionary

from CLI.command_runner import (
    JudgeSelection,
    SearchRunConfig,
    IERunConfig,
    MappingRunConfig,
    JudgeRunConfig,
    build_search_command,
    build_ie_command,
    build_mapping_command,
    build_judge_command,
    format_command,
    launch_background,
    run_foreground,
)
from CLI.llm_config_store import read_llm_config, validate_llm_config
from CLI import ui


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IE_PROMPT = "workflow/ie_agent/prompt.md"
DEFAULT_MAPPING_RULE = "workflow/mapping_agent/prompt.md"
THINKING_CHOICES = ["enabled", "disabled", "none"]
SEARCH_INPUT_CHOICES = [
    questionary.Choice("Local PDF directory", value="local"),
    questionary.Choice("Semantic Scholar search", value="search"),
]


@dataclass(frozen=True)
class FileChoice:
    label: str
    value: str


@dataclass(frozen=True)
class WizardStep:
    key: str
    ask: Callable[[dict[str, object]], object]


def _prompt_file_choices(path: Path, *, repo_root: Path = REPO_ROOT) -> list[FileChoice]:
    return [FileChoice(path.name, path.relative_to(repo_root).as_posix())] if path.is_file() else []


def _run_wizard(
    build_steps: Callable[[dict[str, object]], list[WizardStep]],
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
        values[steps[index].key] = answer
        index += 1


def _text_step(key: str, message: str, *, default: str = "", required: bool = False) -> WizardStep:
    def ask(values: dict[str, object]) -> object:
        while True:
            answer = ui.ask_text_navigation(message, default=str(values.get(key, default)))
            if isinstance(answer, ui.Navigation) or answer or not required:
                return answer
            ui.console.print(f"[red]{message} is required.[/red]")

    return WizardStep(key, ask)


def _select_step(
    key: str,
    message: str,
    choices: list[object],
    *,
    default: str | None = None,
) -> WizardStep:
    return WizardStep(
        key,
        lambda values: ui.ask_select_navigation(
            message,
            choices,
            default=str(values.get(key, default)) if values.get(key, default) is not None else None,
        ),
    )


def _float_step(key: str, message: str, *, default: float = 0.0) -> WizardStep:
    return WizardStep(
        key,
        lambda values: ui.ask_float_navigation(
            message,
            default=float(values.get(key, default)),
        ),
    )


def _int_step(key: str, message: str, *, default: int) -> WizardStep:
    return WizardStep(
        key,
        lambda values: ui.ask_int_navigation(
            message,
            default=int(values.get(key, default)),
        ),
    )


def _confirm_step(key: str, message: str, *, default: bool = False) -> WizardStep:
    return WizardStep(
        key,
        lambda values: ui.ask_confirm_navigation(
            message,
            default=bool(values.get(key, default)),
        ),
    )


def _judge_temperature_step(judge_number: int, *, default: float = 0.0) -> WizardStep:
    key = f"judge_judge_{judge_number}_temperature"

    def ask(values: dict[str, object]) -> object:
        while True:
            answer = ui.ask_float_navigation(
                f"Judge {judge_number} temperature",
                default=float(values.get(key, default)),
            )
            if isinstance(answer, ui.Navigation):
                return answer
            candidate = (
                str(values[f"judge_judge_{judge_number}_model"]),
                str(values[f"judge_judge_{judge_number}_thinking"]),
                float(answer),
            )
            existing = {
                (
                    str(values[f"judge_judge_{index}_model"]),
                    str(values[f"judge_judge_{index}_thinking"]),
                    float(values[f"judge_judge_{index}_temperature"]),
                )
                for index in range(1, judge_number)
            }
            if candidate not in existing:
                return answer
            ui.console.print(
                f"[red]Judge {judge_number} duplicates an existing judge configuration.[/red]"
            )

    return WizardStep(key, ask)


def _file_step(
    key: str,
    message: str,
    choices: list[FileChoice],
    *,
    default: str,
) -> WizardStep:
    options = [questionary.Choice(choice.label, value=choice.value) for choice in choices]
    available_values = {choice.value for choice in choices}
    selected_default = default if default in available_values else choices[0].value
    return _select_step(key, message, options, default=selected_default)


def _build_run_steps(
    values: dict[str, object],
    *,
    config: dict,
    agents: tuple[int, ...],
    ie_choices: list[FileChoice],
    mapping_choices: list[FileChoice],
) -> list[WizardStep]:
    models = _model_choices(config)
    default_model = config.get("defaults", {}).get("model")
    default_thinking = _default_thinking(config)
    steps = [_text_step("run_name", "Run name", required=True)]

    if 0 in agents:
        steps.append(
            _select_step(
                "search_input_mode",
                "Search Agent input mode",
                SEARCH_INPUT_CHOICES,
                default="local",
            )
        )
        input_mode = values.get("search_input_mode")
        if input_mode is None:
            return steps
        if input_mode == "local":
            steps.append(
                _text_step(
                    "search_input_dir",
                    "Search Agent input directory",
                    default="data/papers/raw",
                    required=True,
                )
            )
        elif input_mode == "search":
            steps.extend(
                [
                    _text_step(
                        "search_query",
                        "Search Agent search query",
                        required=True,
                    ),
                    _int_step("search_max_results", "Search Agent max results", default=50),
                ]
            )
        else:
            raise ValueError(f"Unknown Search Agent input mode: {input_mode}")
        steps.extend(
            [
                _select_step("search_model", "Search Agent model", models, default=default_model),
                _select_step(
                    "search_thinking",
                    "Search Agent thinking",
                    THINKING_CHOICES,
                    default=default_thinking,
                ),
                _float_step("search_temperature", "Search Agent temperature"),
            ]
        )

    if 1 in agents:
        if not ie_choices:
            raise ValueError("No IE Agent prompt files found.")
        steps.extend(
            [
                _file_step(
                    "ie_prompt",
                    "IE Agent prompt",
                    ie_choices,
                    default=DEFAULT_IE_PROMPT,
                ),
                _select_step("ie_model", "IE Agent model", models, default=default_model),
                _select_step(
                    "ie_thinking",
                    "IE Agent thinking",
                    THINKING_CHOICES,
                    default=default_thinking,
                ),
                _float_step("ie_temperature", "IE Agent temperature"),
            ]
        )

    if 2 in agents:
        if not mapping_choices:
            raise ValueError("No Mapping Agent rule files found.")
        steps.extend(
            [
                _file_step(
                    "mapping_rule",
                    "Mapping Agent rule",
                    mapping_choices,
                    default=DEFAULT_MAPPING_RULE,
                ),
                _select_step("mapping_model", "Mapping Agent model", models, default=default_model),
                _select_step(
                    "mapping_thinking",
                    "Mapping Agent thinking",
                    THINKING_CHOICES,
                    default=default_thinking,
                ),
                _float_step("mapping_temperature", "Mapping Agent temperature"),
            ]
        )

    if 3 in agents:
        judge_number = 1
        while True:
            add_another_key = f"judge_judge_{judge_number}_add_another"
            steps.extend(
                [
                    _select_step(
                        f"judge_judge_{judge_number}_model",
                        f"Judge {judge_number} model",
                        models,
                        default=default_model,
                    ),
                    _select_step(
                        f"judge_judge_{judge_number}_thinking",
                        f"Judge {judge_number} thinking",
                        THINKING_CHOICES,
                        default=default_thinking,
                    ),
                    _judge_temperature_step(judge_number),
                    _confirm_step(
                        add_another_key,
                        "Add another judge?",
                    ),
                ]
            )
            if values.get(add_another_key) is not True:
                break
            judge_number += 1

    return steps


def _build_commands_from_values(
    agents: tuple[int, ...],
    values: dict[str, object],
) -> list[list[str]]:
    run_name = str(values["run_name"])
    commands: list[list[str]] = []

    if 0 in agents:
        input_mode = str(values["search_input_mode"])
        if input_mode == "local":
            input_value = str(values["search_input_dir"])
            max_results = 50
        elif input_mode == "search":
            input_value = str(values["search_query"])
            max_results = int(values["search_max_results"])
        else:
            raise ValueError(f"Unknown Search Agent input mode: {input_mode}")
        commands.append(
            build_search_command(
                SearchRunConfig(
                    run_name=run_name,
                    input_mode=input_mode,
                    input_value=input_value,
                    max_results=max_results,
                    model=str(values["search_model"]),
                    thinking=str(values["search_thinking"]),
                    temperature=float(values["search_temperature"]),
                )
            )
        )
    if 1 in agents:
        commands.append(
            build_ie_command(
                IERunConfig(
                    run_name=run_name,
                    prompt=str(values["ie_prompt"]),
                    model=str(values["ie_model"]),
                    thinking=str(values["ie_thinking"]),
                    temperature=float(values["ie_temperature"]),
                )
            )
        )
    if 2 in agents:
        commands.append(
            build_mapping_command(
                MappingRunConfig(
                    run_name=run_name,
                    rule=str(values["mapping_rule"]),
                    model=str(values["mapping_model"]),
                    thinking=str(values["mapping_thinking"]),
                    temperature=float(values["mapping_temperature"]),
                )
            )
        )
    if 3 in agents:
        judges: list[JudgeSelection] = []
        judge_number = 1
        while True:
            judges.append(
                JudgeSelection(
                    model_id=str(values[f"judge_judge_{judge_number}_model"]),
                    thinking=str(values[f"judge_judge_{judge_number}_thinking"]),
                    temperature=float(values[f"judge_judge_{judge_number}_temperature"]),
                )
            )
            if values.get(f"judge_judge_{judge_number}_add_another") is not True:
                break
            judge_number += 1
        commands.append(build_judge_command(JudgeRunConfig(run_name=run_name, judges=judges)))

    return commands


def run_menu() -> None:
    while True:
        ui.clear_screen()
        ui.header("tc-norm / Run")
        choice = ui.ask_select_navigation(
            "Choose a run action",
            ["Run all agents", "Search Agent", "IE Agent", "Mapping Agent", "Judge Agent", "Back"],
        )
        if isinstance(choice, ui.Navigation) or choice == "Back":
            return
        if choice == "Run all agents":
            _run_agents((0, 1, 2, 3))
        elif choice == "Search Agent":
            _run_agents((0,))
        elif choice == "IE Agent":
            _run_agents((1,))
        elif choice == "Mapping Agent":
            _run_agents((2,))
        elif choice == "Judge Agent":
            _run_agents((3,))


def _load_config() -> dict | None:
    try:
        config = read_llm_config()
    except FileNotFoundError:
        ui.console.print("[red]llm.json was not found.[/red]")
        ui.console.print("Use tc-norm -> LLM config -> Initialize llm.json first.")
        ui.pause()
        return None
    errors = validate_llm_config(config)
    if errors:
        ui.console.print("[red]llm.json has validation errors.[/red]")
        ui.print_errors(errors)
        ui.pause()
    return config


def _default_thinking(config: dict) -> str:
    value = config.get("defaults", {}).get("thinking", "enabled")
    if isinstance(value, bool):
        return "enabled" if value else "disabled"
    return value if value in THINKING_CHOICES else "enabled"


def _model_choices(config: dict) -> list[str]:
    return list(config.get("models", {}).keys())


def _run_agents(agents: tuple[int, ...]) -> None:
    config = _load_config()
    if config is None:
        return
    ie_choices = (
        _prompt_file_choices(REPO_ROOT / DEFAULT_IE_PROMPT)
        if 1 in agents
        else []
    )
    mapping_choices = (
        _prompt_file_choices(REPO_ROOT / DEFAULT_MAPPING_RULE)
        if 2 in agents
        else []
    )
    if 1 in agents and not ie_choices:
        ui.console.print("[red]No IE Agent prompt files found.[/red]")
        ui.pause()
        return
    if 2 in agents and not mapping_choices:
        ui.console.print("[red]No Mapping Agent rule files found.[/red]")
        ui.pause()
        return

    values = _run_wizard(
        lambda current_values: _build_run_steps(
            current_values,
            config=config,
            agents=agents,
            ie_choices=ie_choices,
            mapping_choices=mapping_choices,
        )
    )
    if values is None:
        return
    commands = _build_commands_from_values(agents, values)
    _preview_and_execute(commands, run_name=str(values["run_name"]))


def _preview_and_execute(commands: list[list[str]], *, run_name: str) -> None:
    while True:
        ui.clear_screen()
        ui.print_command_preview(commands, format_command)
        action = ui.ask_select_navigation(
            "How should tc-norm handle this command?",
            ["Run now", "Launch in background with one log", "Print commands only", "Back"],
        )
        if isinstance(action, ui.Navigation) or action == "Back":
            return
        if action == "Print commands only":
            for command in commands:
                ui.console.print(format_command(command))
            ui.pause()
            return
        if action == "Run now":
            return_code = run_foreground(commands)
            if return_code == 0:
                ui.console.print("[green]Command sequence finished.[/green]")
            else:
                ui.console.print(f"[red]Command sequence stopped with exit code {return_code}.[/red]")
            ui.pause()
            return
        if action == "Launch in background with one log":
            pid, log_path = launch_background(commands, run_name=run_name)
            ui.console.print(f"[green]Started PID {pid}.[/green]")
            ui.console.print(f"Log: {log_path}")
            ui.pause()
            return
