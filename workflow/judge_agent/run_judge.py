from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence

from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflow.common.llm_chat import apply_chat_options, create_openai_client
from workflow.common.llm_config import RuntimeConfig, load_runtime_config
from workflow.common.paths import paper_agent_path, run_agent_dir
from workflow.common.run_metadata import runtime_config_metadata, update_run_metadata
from workflow.common.agent_io import MappingPaper, read_mapping_paper
from workflow.common.json_utils import parse_json_array
from workflow.mapping_agent.mapping_task import PROMPT_PATH as MAPPING_PROMPT_PATH, read_mapping_rule
from workflow.judge_agent.judge_task import build_user_prompt, mapping_records, validate_feedback

LLM_TIMEOUT_SECONDS = 3600
THINKING_CHOICES = {"enabled", "disabled", "none"}


@dataclass(frozen=True)
class JudgeConfig:
    model_id: str
    thinking: str
    temperature: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "thinking": self.thinking,
            "temperature": self.temperature,
        }


def parse_judge_config(value: str) -> JudgeConfig:
    if ":" not in value:
        raise argparse.ArgumentTypeError(
            "Judge config must be '<model_id>:thinking=<enabled|disabled|none>,temperature=<float>'."
        )

    model_id, raw_options = value.split(":", 1)
    model_id = model_id.strip()
    if not model_id:
        raise argparse.ArgumentTypeError("Judge config model_id cannot be empty.")

    options: dict[str, str] = {}
    for raw_part in raw_options.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "=" not in part:
            raise argparse.ArgumentTypeError(f"Invalid judge option {part!r}; expected key=value.")
        key, option_value = part.split("=", 1)
        key = key.strip()
        option_value = option_value.strip()
        if key in options:
            raise argparse.ArgumentTypeError(f"Duplicate judge option {key!r}.")
        options[key] = option_value

    missing = {"thinking", "temperature"} - options.keys()
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise argparse.ArgumentTypeError(f"Judge config missing required option(s): {missing_text}.")

    extra = set(options) - {"thinking", "temperature"}
    if extra:
        extra_text = ", ".join(sorted(extra))
        raise argparse.ArgumentTypeError(f"Unknown judge option(s): {extra_text}.")

    thinking = options["thinking"]
    if thinking not in THINKING_CHOICES:
        raise argparse.ArgumentTypeError("Judge thinking must be one of: enabled, disabled, none.")

    try:
        temperature = float(options["temperature"])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Judge temperature must be a float.") from exc

    return JudgeConfig(model_id=model_id, thinking=thinking, temperature=temperature)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_rule_path(mapping_paper: MappingPaper, rule_path: Path | None) -> Path:
    return MAPPING_PROMPT_PATH if rule_path is None else rule_path


def build_judge_messages(*, rule_text: str, record: dict) -> list[dict[str, str]]:
    return [{"role": "user", "content": build_user_prompt(rule_text=rule_text, records=[{**record, "case_id": 0}])}]


def run_chat(
    *,
    client: OpenAI,
    config: RuntimeConfig,
    messages: Sequence[dict[str, str]],
    timeout_seconds: float,
    temperature: float,
) -> tuple[str, dict[str, Any]]:
    request_kwargs: dict[str, Any] = {
        "model": config.model_name,
        "messages": list(messages),
    }
    apply_chat_options(
        request_kwargs,
        temperature=temperature,
        thinking=config.thinking,
        provider_name=config.provider_name,
        stream=config.stream,
        include_usage=True,
    )

    started = time.time()
    content_parts: list[str] = []
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    response_id: str | None = None
    model_returned: str | None = None

    response = client.chat.completions.create(**request_kwargs, timeout=timeout_seconds)
    if config.stream:
        for chunk in response:
            response_id = response_id or getattr(chunk, "id", None)
            model_returned = model_returned or getattr(chunk, "model", None)
            if getattr(chunk, "usage", None) is not None:
                usage = chunk.usage.model_dump(mode="json")
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            finish_reason = choice.finish_reason or finish_reason
            content_part = getattr(choice.delta, "content", None)
            if content_part:
                content_parts.append(content_part)
    else:
        choice = response.choices[0]
        content_parts.append(choice.message.content or "")
        usage = response.usage.model_dump(mode="json") if response.usage is not None else None
        response_id = response.id
        model_returned = response.model
        finish_reason = choice.finish_reason

    content = "".join(content_parts)
    metadata = {
        "completed_at": utc_now(),
        "elapsed_seconds": round(time.time() - started, 3),
        "model_id": config.model_id,
        "provider": config.provider_name,
        "model": config.model_name,
        "model_returned": model_returned,
        "thinking": config.thinking,
        "stream": config.stream,
        "temperature": temperature,
        "response_id": response_id,
        "finish_reason": finish_reason,
        "usage": usage,
    }
    return content, metadata


def judge_item(*, record: dict, rule_text: str, judge_configs: Sequence[JudgeConfig]) -> dict[str, Any]:
    judgments = []
    for judge_config in judge_configs:
        config = load_runtime_config(model_id=judge_config.model_id, thinking=judge_config.thinking, stream=True)
        with create_openai_client(config, timeout_seconds=LLM_TIMEOUT_SECONDS) as client:
            content, metadata = run_chat(
                client=client, config=config,
                messages=build_judge_messages(rule_text=rule_text, record=record),
                timeout_seconds=LLM_TIMEOUT_SECONDS, temperature=judge_config.temperature,
            )
        if metadata.get("finish_reason") != "stop":
            raise ValueError("Judge response did not finish normally")
        parsed, error = parse_json_array(content)
        if error:
            raise ValueError(f"Invalid Judge JSON: {error}")
        feedback = validate_feedback(parsed, 1)[0]
        judgments.append({"judge_model": config.model_id, "provider": config.provider_name,
                          "model": config.model_name, "thinking": config.thinking,
                          "temperature": judge_config.temperature,
                          "decision": feedback["decision"], "feedback": feedback["feedback"],
                          "response_metadata": metadata})
    return {**record, "judge_feedback": judgments}


def judged_items(*, paper: MappingPaper, rule_text: str, judge_configs: Sequence[JudgeConfig]) -> list[dict[str, Any]]:
    return [judge_item(record=record, rule_text=rule_text, judge_configs=judge_configs)
            for record in mapping_records(paper)]


def run_judge(
    *,
    mapping_input: Path,
    output_json: Path,
    judge_configs: Sequence[JudgeConfig],
    rule_path: Path | None,
) -> dict[str, Any]:
    if not judge_configs:
        raise ValueError("At least one --judge is required.")

    mapping_paper = read_mapping_paper(mapping_input)
    resolved_rule_path = resolve_rule_path(mapping_paper, rule_path)
    if not resolved_rule_path.exists():
        raise FileNotFoundError(f"Missing rule file: {resolved_rule_path}")

    rule_text = read_mapping_rule(resolved_rule_path)
    payload = {
        "title": mapping_paper.title,
        "rule_version": mapping_paper.rule_version,
        "judges": [judge_config.as_dict() for judge_config in judge_configs],
        "items": judged_items(
            paper=mapping_paper,
            rule_text=rule_text,
            judge_configs=judge_configs,
        ),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "output_json": str(output_json),
        "mapping_input": str(mapping_input),
        "rule_path": str(resolved_rule_path),
        "item_count": len(mapping_paper.items),
        "judges": [judge_config.as_dict() for judge_config in judge_configs],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--judge",
        action="append",
        required=True,
        type=parse_judge_config,
        help="Judge config: <model_id>:thinking=<enabled|disabled|none>,temperature=<float>.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    mapping_dir = run_agent_dir(args.run_name, "mapping_agent")
    if not mapping_dir.is_dir():
        raise SystemExit(f"Missing Mapping Agent output directory: {mapping_dir}")
    mapping_files = sorted(mapping_dir.glob("*.json"))
    if not mapping_files:
        raise SystemExit(f"No Mapping Agent paper outputs found under: {mapping_dir}")

    judge_runtime_configs = [
        load_runtime_config(model_id=judge_config.model_id, thinking=judge_config.thinking, stream=True)
        for judge_config in args.judge
    ]
    statuses: list[dict[str, Any]] = []
    for mapping_input in mapping_files:
        statuses.append(
            run_judge(
                mapping_input=mapping_input,
                output_json=paper_agent_path(args.run_name, "judge_agent", mapping_input.stem),
                judge_configs=args.judge,
                rule_path=None,
            )
        )
    update_run_metadata(
        args.run_name,
        "judge_agent",
        {
            "input_dir": str(mapping_dir),
            "output_dir": str(run_agent_dir(args.run_name, "judge_agent")),
            "judges": [
                {
                    **runtime_config_metadata(config),
                    "temperature": judge_config.temperature,
                }
                for judge_config, config in zip(args.judge, judge_runtime_configs)
            ],
            "llm_timeout_seconds": LLM_TIMEOUT_SECONDS,
            "paper_count": len(statuses),
            "item_count": sum(int(status["item_count"]) for status in statuses),
            "rule_paths": sorted({status["rule_path"] for status in statuses}),
        },
    )
    print(
        json.dumps(
            {"judge_dir": str(run_agent_dir(args.run_name, "judge_agent")), "paper_count": len(statuses), "runs": statuses},
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
