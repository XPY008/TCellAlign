from __future__ import annotations

import argparse
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

from workflow.common.json_utils import parse_json_array
from workflow.common.llm_chat import apply_chat_options, create_openai_client
from workflow.common.llm_config import RuntimeConfig, load_runtime_config
from workflow.common.paths import paper_agent_path, run_agent_dir
from workflow.common.run_metadata import runtime_config_metadata, update_run_metadata
from workflow.common.agent_io import (
    MappedLabel,
    IEPaper,
    MappingPaper,
    read_ie_paper,
    write_mapping_paper,
)
from workflow.mapping_agent.mapping_task import MappingCase, build_mapping_prompt, read_mapping_rule, PROMPT_PATH
from workflow.mapping_agent.slot_schema import build_mapped_label, normalize_mapped_slots

LLM_TIMEOUT_SECONDS = 3600


def evaluate_predictions(payload: list[Any] | None, expected_cases: Sequence[MappingCase]) -> dict[str, Any]:
    problems: list[str] = []
    if payload is None:
        return {
            "schema_ok": False,
            "item_count": 0,
            "problems": ["No parsed prediction list."],
        }

    if len(payload) != len(expected_cases):
        problems.append(f"Expected {len(expected_cases)} items, got {len(payload)}.")

    expected_case_ids = {case.case_id for case in expected_cases}
    seen_case_ids: set[int] = set()

    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            problems.append(f"[{index}] is not an object.")
            continue
        extra_fields = sorted(set(item) - {"case_id", "mapped_slots"})
        if extra_fields:
            problems.append(f"[{index}] has unexpected field(s): {extra_fields}.")

        case_id = item.get("case_id")
        if type(case_id) is not int:
            problems.append(f"[{index}].case_id is missing or not an integer.")
        elif case_id in seen_case_ids:
            problems.append(f"[{index}].case_id duplicates an earlier prediction.")
        elif case_id not in expected_case_ids:
            problems.append(f"[{index}].case_id does not match any input case.")
        else:
            seen_case_ids.add(case_id)

        _mapped_slots, slot_problems = normalize_mapped_slots(item.get("mapped_slots"))
        problems.extend(f"[{index}].{problem}" for problem in slot_problems)

    for case in expected_cases:
        if case.case_id not in seen_case_ids:
            problems.append(f"Missing prediction for case_id {case.case_id}.")

    return {
        "schema_ok": not problems,
        "item_count": len(payload),
        "problems": problems,
    }


def mapping_cases_from_ie_papers(papers: list[IEPaper]) -> list[MappingCase]:
    cases: list[MappingCase] = []
    for paper in papers:
        for item in paper.items:
            cases.append(
                MappingCase(
                    case_id=len(cases),
                    original_label=item.original_label,
                    evidence=item.evidence,
                )
            )
    return cases


def mapping_papers_from_predictions(
    ie_papers: list[IEPaper],
    predictions: list[Any],
    *,
    rule_version: str,
) -> list[MappingPaper]:
    expected_cases = mapping_cases_from_ie_papers(ie_papers)
    if len(predictions) != len(expected_cases):
        raise ValueError(f"Expected {len(expected_cases)} predictions, got {len(predictions)}.")

    predictions_by_case_id: dict[int, dict[str, Any]] = {}
    expected_case_ids = {case.case_id for case in expected_cases}
    for index, prediction in enumerate(predictions):
        if not isinstance(prediction, dict):
            raise ValueError(f"Prediction {index + 1} must be an object.")
        case_id = prediction.get("case_id")
        if type(case_id) is not int:
            raise ValueError(f"Prediction {index + 1} case_id must be an integer.")
        if case_id not in expected_case_ids:
            raise ValueError(f"Prediction {index + 1} case_id does not match any IE Agent input.")
        if case_id in predictions_by_case_id:
            raise ValueError(f"Prediction {index + 1} case_id duplicates an earlier prediction.")
        mapped_slots, slot_problems = normalize_mapped_slots(prediction.get("mapped_slots"))
        if slot_problems:
            raise ValueError(f"Prediction {index + 1} mapped_slots is invalid: {slot_problems}")
        predictions_by_case_id[case_id] = {"mapped_slots": mapped_slots}

    prediction_index = 0
    papers: list[MappingPaper] = []
    for paper in ie_papers:
        mapped_items: list[MappedLabel] = []
        for item in paper.items:
            expected = expected_cases[prediction_index]
            prediction_index += 1
            prediction = predictions_by_case_id[expected.case_id]
            mapped_slots = prediction["mapped_slots"]
            mapped_label = build_mapped_label(mapped_slots)
            mapped_items.append(
                MappedLabel(
                    original_label=item.original_label,
                    evidence=item.evidence,
                    mapped_label=mapped_label,
                    mapped_slots=mapped_slots,
                )
            )
        papers.append(MappingPaper(title=paper.title, rule_version=rule_version, items=mapped_items))
    return papers


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_chat(
    *,
    client: OpenAI,
    config: RuntimeConfig,
    prompt: str,
    timeout_seconds: float,
    temperature: float,
) -> tuple[str, str, dict[str, Any]]:
    request_kwargs: dict[str, Any] = {
        "model": config.model_name,
        "messages": [{"role": "user", "content": prompt}],
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
    reasoning_parts: list[str] = []
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
            delta = choice.delta
            content_part = getattr(delta, "content", None)
            reasoning_part = getattr(delta, "reasoning_content", None)
            if content_part:
                content_parts.append(content_part)
            if reasoning_part:
                reasoning_parts.append(reasoning_part)
    else:
        choice = response.choices[0]
        content_parts.append(choice.message.content or "")
        reasoning = getattr(choice.message, "reasoning_content", None)
        if reasoning:
            reasoning_parts.append(reasoning)
        usage = response.usage.model_dump(mode="json") if response.usage is not None else None
        response_id = response.id
        model_returned = response.model
        finish_reason = choice.finish_reason

    content = "".join(content_parts)
    reasoning = "".join(reasoning_parts)
    metadata = {
        "status": "completed",
        "completed_at": utc_now(),
        "elapsed_seconds": round(time.time() - started, 3),
        "model_id": config.model_id,
        "model": config.model_name,
        "model_returned": model_returned,
        "thinking": config.thinking,
        "stream": config.stream,
        "response_id": response_id,
        "finish_reason": finish_reason,
        "usage": usage,
        "content_characters": len(content),
        "reasoning_characters": len(reasoning),
        "request_keys": sorted(request_kwargs),
    }
    return content, reasoning, metadata


def run_mapping(
    *,
    ie_input: Path,
    rule_path: Path,
    output_json: Path,
    model_id: str | None,
    thinking: str | None,
    temperature: float,
) -> dict[str, Any]:
    ie_papers = [read_ie_paper(ie_input)]
    cases = mapping_cases_from_ie_papers(ie_papers)
    rule_text = read_mapping_rule(rule_path)
    prompt = build_mapping_prompt(rule_text, cases)
    config = load_runtime_config(model_id=model_id, thinking=thinking, stream=True)
    client = create_openai_client(config, timeout_seconds=LLM_TIMEOUT_SECONDS)
    content, _reasoning, response_metadata = run_chat(
        client=client,
        config=config,
        prompt=prompt,
        timeout_seconds=LLM_TIMEOUT_SECONDS,
        temperature=temperature,
    )
    parsed, parse_error = parse_json_array(content)
    validation = evaluate_predictions(parsed, cases)
    if parsed is None:
        raise ValueError(f"Mapping Agent response was not valid JSON: {parse_error}")
    if not validation["schema_ok"]:
        raise ValueError(f"Mapping Agent response failed validation: {validation['problems']}")

    mapping_papers = mapping_papers_from_predictions(ie_papers, parsed, rule_version=rule_path.stem)
    write_mapping_paper(output_json, mapping_papers[0])
    return {
        "output_json": str(output_json),
        "ie_input": str(ie_input),
        "rule_path": str(rule_path),
        "item_count": len(cases),
        "model_id": config.model_id,
        "model": config.model_name,
        "thinking": config.thinking,
        "stream": config.stream,
        "response_metadata": response_metadata,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--rule", default=str(PROMPT_PATH))
    parser.add_argument("--model", default=None)
    parser.add_argument("--thinking", choices=["enabled", "disabled", "none"], default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    ie_dir = run_agent_dir(args.run_name, "ie_agent")
    rule_path = Path(args.rule).expanduser().resolve()
    if not ie_dir.is_dir():
        raise SystemExit(f"Missing IE Agent output directory: {ie_dir}")
    if not rule_path.exists():
        raise SystemExit(f"Missing Mapping Agent rule: {rule_path}")
    ie_files = sorted(ie_dir.glob("*.json"))
    if not ie_files:
        raise SystemExit(f"No IE Agent paper outputs found under: {ie_dir}")

    config = load_runtime_config(model_id=args.model, thinking=args.thinking, stream=True)
    statuses: list[dict[str, Any]] = []
    for ie_input in ie_files:
        output_json = paper_agent_path(args.run_name, "mapping_agent", ie_input.stem)
        statuses.append(
            run_mapping(
                ie_input=ie_input,
                rule_path=rule_path,
                output_json=output_json,
                model_id=args.model,
                thinking=args.thinking,
                temperature=args.temperature,
            )
        )
    update_run_metadata(
        args.run_name,
        "mapping_agent",
        {
            "input_dir": str(ie_dir),
            "output_dir": str(run_agent_dir(args.run_name, "mapping_agent")),
            "rule": str(rule_path),
            "rule_version": rule_path.stem,
            "model": runtime_config_metadata(config),
            "temperature": args.temperature,
            "llm_timeout_seconds": LLM_TIMEOUT_SECONDS,
            "paper_count": len(statuses),
            "item_count": sum(int(status["item_count"]) for status in statuses),
        },
    )
    print(
        json.dumps(
            {"mapping_dir": str(run_agent_dir(args.run_name, "mapping_agent")), "paper_count": len(statuses), "runs": statuses},
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
