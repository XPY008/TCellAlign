from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import shlex
import subprocess
import sys
from time import perf_counter


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SearchRunConfig:
    run_name: str
    input_mode: str
    input_value: str
    max_results: int
    model: str
    thinking: str
    temperature: float


@dataclass(frozen=True)
class IERunConfig:
    run_name: str
    prompt: str
    model: str
    thinking: str
    temperature: float


@dataclass(frozen=True)
class MappingRunConfig:
    run_name: str
    rule: str
    model: str
    thinking: str
    temperature: float


@dataclass(frozen=True)
class JudgeSelection:
    model_id: str
    thinking: str
    temperature: float


@dataclass(frozen=True)
class JudgeRunConfig:
    run_name: str
    judges: list[JudgeSelection]


def _base_command(script_path: str) -> list[str]:
    return [sys.executable, script_path]


def _float_arg(value: float) -> str:
    return f"{value:g}"


def build_search_command(config: SearchRunConfig) -> list[str]:
    if config.input_mode == "local":
        input_args = ["--input-dir", config.input_value]
    elif config.input_mode == "search":
        input_args = [
            "--query",
            config.input_value,
            "--max-results",
            str(config.max_results),
        ]
    else:
        raise ValueError(f"Unknown Search Agent input mode: {config.input_mode}")

    return [
        *_base_command("workflow/search_agent/run_raw_pdf_screening.py"),
        "--run-name",
        config.run_name,
        *input_args,
        "--model",
        config.model,
        "--thinking",
        config.thinking,
        "--temperature",
        _float_arg(config.temperature),
    ]


def build_ie_command(config: IERunConfig) -> list[str]:
    return [
        *_base_command("workflow/ie_agent/run_extraction.py"),
        "--run-name",
        config.run_name,
        "--prompt",
        config.prompt,
        "--model",
        config.model,
        "--thinking",
        config.thinking,
        "--temperature",
        _float_arg(config.temperature),
    ]


def build_mapping_command(config: MappingRunConfig) -> list[str]:
    return [
        *_base_command("workflow/mapping_agent/run_mapping.py"),
        "--run-name",
        config.run_name,
        "--rule",
        config.rule,
        "--model",
        config.model,
        "--thinking",
        config.thinking,
        "--temperature",
        _float_arg(config.temperature),
    ]


def build_judge_command(config: JudgeRunConfig) -> list[str]:
    command = [
        *_base_command("workflow/judge_agent/run_judge.py"),
        "--run-name",
        config.run_name,
    ]
    for judge in config.judges:
        command.extend(
            [
                "--judge",
                f"{judge.model_id}:thinking={judge.thinking},temperature={_float_arg(judge.temperature)}",
            ]
        )
    return command


def build_all_agent_commands(
    *,
    search: SearchRunConfig,
    ie: IERunConfig,
    mapping: MappingRunConfig,
    judge: JudgeRunConfig,
) -> list[list[str]]:
    return [
        build_search_command(search),
        build_ie_command(ie),
        build_mapping_command(mapping),
        build_judge_command(judge),
    ]


def format_command(command: list[str]) -> str:
    return shlex.join(command)


def format_command_sequence(commands: list[list[str]]) -> str:
    return "\n".join(format_command(command) for command in commands)


def _agent_label(command: list[str]) -> str:
    names = {
        "search_agent": "Search Agent",
        "ie_agent": "IE Agent",
        "mapping_agent": "Mapping Agent",
        "judge_agent": "Judge Agent",
    }
    for argument in command:
        for part in Path(argument).parts:
            if part in names:
                return names[part]
    return "Command"


def run_foreground(commands: list[list[str]]) -> int:
    total = len(commands)
    for index, command in enumerate(commands, start=1):
        agent_label = _agent_label(command)
        prefix = f"[{index}/{total}]"
        print(f"{prefix} Starting {agent_label}", flush=True)
        print(f"Command: {format_command(command)}", flush=True)
        started_at = perf_counter()
        completed = subprocess.run(command, cwd=REPO_ROOT)
        elapsed = perf_counter() - started_at
        if completed.returncode != 0:
            print(
                f"{prefix} {agent_label} failed with exit code "
                f"{completed.returncode} after {elapsed:.1f}s",
                flush=True,
            )
            return completed.returncode
        print(f"{prefix} {agent_label} finished in {elapsed:.1f}s", flush=True)
    return 0


def launch_background(commands: list[list[str]], *, run_name: str) -> tuple[int, Path]:
    logs_dir = REPO_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = logs_dir / f"tc-norm-{run_name}-{timestamp}.log"
    shell_command = " && ".join(format_command(command) for command in commands)
    log_file = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        ["nohup", "bash", "-lc", shell_command],
        cwd=REPO_ROOT,
        stdin=subprocess.DEVNULL,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_file.close()
    return process.pid, log_path
