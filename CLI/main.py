from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CLI import ui
from CLI.llm_config_store import read_llm_config, validate_llm_config
from CLI.menu_llm_config import llm_config_menu
from CLI.menu_run import run_menu


def validate_environment() -> None:
    ui.clear_screen()
    errors: list[str] = []
    for path in [
        "llm.json",
        "workflow/search_agent/run_raw_pdf_screening.py",
        "workflow/ie_agent/run_extraction.py",
        "workflow/mapping_agent/run_mapping.py",
        "workflow/judge_agent/run_judge.py",
    ]:
        if not (REPO_ROOT / path).exists():
            errors.append(f"Missing {path}")
    try:
        config = read_llm_config()
    except Exception as exc:  # pragma: no cover - interactive status path
        errors.append(f"Cannot read llm.json: {exc}")
    else:
        errors.extend(validate_llm_config(config))

    if errors:
        ui.console.print("[red]Environment validation found issues.[/red]")
        ui.print_errors(errors)
    else:
        ui.console.print("[green]Environment validation passed.[/green]")
    ui.pause()


def main() -> None:
    try:
        while True:
            ui.clear_screen()
            ui.header("tc-norm")
            choice = ui.ask_select_navigation(
                "Choose an action",
                ["Run", "LLM config", "Validate environment", "Exit"],
            )
            if isinstance(choice, ui.Navigation) or choice == "Exit":
                return
            if choice == "Run":
                run_menu()
            elif choice == "LLM config":
                llm_config_menu()
            elif choice == "Validate environment":
                validate_environment()
    except KeyboardInterrupt:
        ui.console.print("\n[yellow]Exiting tc-norm.[/yellow]")


if __name__ == "__main__":
    main()
