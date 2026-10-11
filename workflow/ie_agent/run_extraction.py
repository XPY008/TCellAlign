from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any

from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflow.common.json_utils import parse_json_object
from workflow.common.llm_chat import apply_chat_options, create_openai_client
from workflow.common.llm_config import RuntimeConfig, load_runtime_config
from workflow.common.paths import paper_agent_path, run_agent_dir
from workflow.common.run_metadata import runtime_config_metadata, update_run_metadata
from workflow.common.agent_io import (
    SEARCH_QUALIFIED_PAPERS,
    ExtractedLabel,
    IEPaper,
    read_search_qualified_papers,
    write_ie_paper,
)

DEFAULT_PROMPT = Path(__file__).resolve().parent / "prompt.md"
LLM_TIMEOUT_SECONDS = 3600


@dataclass(frozen=True)
class ValidationSummary:
    schema_ok: bool
    item_count: int
    labels: list[str]
    problems: list[str]


@dataclass(frozen=True)
class ExtractionRunResult:
    parsed: dict[str, Any] | None
    validation: ValidationSummary


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_json_content(content: str) -> tuple[dict[str, Any] | None, str | None]:
    return parse_json_object(content)


def load_search_input(path: str | Path) -> tuple[str, dict[str, Any]]:
    input_path = Path(path).expanduser().resolve()
    text = input_path.read_text(encoding="utf-8")
    return text, {
        "source_path": str(input_path),
        "input_type": "search_markdown",
        "input_characters": len(text),
    }


def build_user_prompt(
    prompt_template: str,
    ie_input: str,
    *,
    input_label: str = "IE Agent text",
    input_instruction: str | None = None,
) -> str:
    instruction = input_instruction or "Use only the supplied text below."
    return (
        prompt_template.rstrip()
        + "\n\n"
        + f"{input_label} input:\n"
        + instruction.rstrip()
        + "\n\n"
        + "SUPPLIED TEXT START\n"
        + ie_input
        + "SUPPLIED TEXT END\n"
    )


def run_chat(
    *,
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float | None,
    thinking: str | None,
    provider_name: str | None,
    stream: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    request_kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return valid JSON only. Do not include markdown fences."},
            {"role": "user", "content": prompt},
        ],
    }
    apply_chat_options(
        request_kwargs,
        temperature=temperature,
        thinking=thinking,
        provider_name=provider_name,
        stream=stream,
        include_usage=True,
    )

    started = time.time()
    started_at = utc_now()

    try:
        response = client.chat.completions.create(**request_kwargs)
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        usage: dict[str, Any] | None = None
        finish_reason: str | None = None
        response_id: str | None = None
        model_returned: str | None = None

        if stream:
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
        reasoning_content = "".join(reasoning_parts)
        parsed, parse_error = parse_json_content(content)
        metadata = {
            "status": "completed",
            "started_at": started_at,
            "completed_at": utc_now(),
            "elapsed_seconds": round(time.time() - started, 3),
            "model": model,
            "model_returned": model_returned,
            "response_id": response_id,
            "finish_reason": finish_reason,
            "usage": usage,
            "content_characters": len(content),
            "reasoning_characters": len(reasoning_content),
            "parse_error": parse_error,
            "request_keys": sorted(request_kwargs),
        }
        return parsed, metadata
    except Exception as exc:
        error_payload = {
            "status": "failed",
            "started_at": started_at,
            "failed_at": utc_now(),
            "elapsed_seconds": round(time.time() - started, 3),
            "model": model,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "request_keys": sorted(request_kwargs),
        }
        return None, error_payload


def validate_output(parsed: dict[str, Any] | None) -> ValidationSummary:
    problems: list[str] = []
    labels: list[str] = []
    if not isinstance(parsed, dict):
        return ValidationSummary(schema_ok=False, item_count=0, labels=[], problems=["output is not a JSON object"])

    if not isinstance(parsed.get("title"), str):
        problems.append("title is missing or not a string")

    items = parsed.get("items")
    if not isinstance(items, list):
        problems.append("items is missing or not a list")
        items = []

    for index, item in enumerate(items):
        if not isinstance(item, dict):
            problems.append(f"items[{index}] is not an object")
            continue
        original_label = item.get("original_label")
        evidence = item.get("evidence")
        if not isinstance(original_label, str) or not original_label.strip():
            problems.append(f"items[{index}].original_label is missing or not a non-empty string")
        else:
            labels.append(original_label)
        if not isinstance(evidence, list) or not all(isinstance(value, str) for value in evidence):
            problems.append(f"items[{index}].evidence is missing or not a string list")

    return ValidationSummary(
        schema_ok=not problems,
        item_count=len(items),
        labels=labels,
        problems=problems,
    )


def ie_paper_from_parsed(title: str, parsed: dict[str, Any]) -> IEPaper:
    if not title.strip():
        raise ValueError("IE Agent output title must be non-empty.")
    items_payload = parsed.get("items")
    if not isinstance(items_payload, list):
        raise ValueError("IE Agent parsed output must contain items.")

    items: list[ExtractedLabel] = []
    for index, raw_item in enumerate(items_payload):
        if not isinstance(raw_item, dict):
            raise ValueError(f"items[{index}] must be an object.")
        original_label = raw_item.get("original_label")
        evidence = raw_item.get("evidence")
        if not isinstance(original_label, str) or not original_label.strip():
            raise ValueError(f"items[{index}].original_label must be a non-empty string.")
        if not isinstance(evidence, list):
            raise ValueError(f"items[{index}].evidence must be a non-empty string list.")
        clean_evidence = [value.strip() for value in evidence if isinstance(value, str) and value.strip()]
        if len(clean_evidence) != len(evidence) or not clean_evidence:
            raise ValueError(f"items[{index}].evidence must be a non-empty string list.")
        items.append(ExtractedLabel(original_label=original_label.strip(), evidence=clean_evidence))

    return IEPaper(title=title.strip(), items=items)


def run_extraction_once(
    *,
    search_input_path: Path,
    prompt_path: Path,
    config: RuntimeConfig,
    temperature: float,
) -> ExtractionRunResult:
    ie_input, input_metadata = load_search_input(search_input_path)
    user_prompt_input_label = "Search Agent paper markdown"
    user_prompt_instruction = "Use only the supplied full-paper markdown below."

    prompt_template = prompt_path.read_text(encoding="utf-8")
    user_prompt = build_user_prompt(
        prompt_template,
        ie_input,
        input_label=user_prompt_input_label,
        input_instruction=user_prompt_instruction,
    )

    client = create_openai_client(config, timeout_seconds=LLM_TIMEOUT_SECONDS)
    parsed, response_metadata = run_chat(
        client=client,
        model=config.model_name,
        prompt=user_prompt,
        temperature=temperature,
        thinking=config.thinking,
        provider_name=config.provider_name,
        stream=config.stream,
    )
    validation = validate_output(parsed)
    _ = input_metadata, response_metadata
    return ExtractionRunResult(parsed=parsed, validation=validation)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--prompt", default=str(DEFAULT_PROMPT))
    parser.add_argument("--model", default=None)
    parser.add_argument("--thinking", choices=["enabled", "disabled", "none"], default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    search_manifest_path = run_agent_dir(args.run_name, "search_agent") / SEARCH_QUALIFIED_PAPERS
    ie_dir = run_agent_dir(args.run_name, "ie_agent")
    prompt_path = Path(args.prompt).expanduser().resolve()
    if not search_manifest_path.exists():
        raise SystemExit(f"Missing Search Agent manifest: {search_manifest_path}")
    if not prompt_path.exists():
        raise SystemExit(f"Missing prompt file: {prompt_path}")

    manifest_papers = read_search_qualified_papers(search_manifest_path)
    if not manifest_papers:
        raise SystemExit(f"No Search Agent qualified papers found in: {search_manifest_path}")
    config = load_runtime_config(model_id=args.model, thinking=args.thinking, stream=True)
    written: list[str] = []
    for paper in manifest_papers:
        markdown_path = Path(paper.markdown_path).expanduser().resolve()
        if not markdown_path.exists():
            raise SystemExit(f"Missing Search Agent markdown for {paper.title}: {markdown_path}")
        result = run_extraction_once(
            search_input_path=markdown_path,
            prompt_path=prompt_path,
            config=config,
            temperature=args.temperature,
        )
        if result.parsed is None or not result.validation.schema_ok:
            raise SystemExit(f"IE Agent extraction failed validation for {paper.title}: {paper.paper_slug}")
        output_path = paper_agent_path(args.run_name, "ie_agent", paper.paper_slug)
        write_ie_paper(output_path, ie_paper_from_parsed(paper.title, result.parsed))
        written.append(str(output_path))

    update_run_metadata(
        args.run_name,
        "ie_agent",
        {
            "input_manifest": str(search_manifest_path),
            "output_dir": str(ie_dir),
            "prompt": str(prompt_path),
            "model": runtime_config_metadata(config),
            "temperature": args.temperature,
            "llm_timeout_seconds": LLM_TIMEOUT_SECONDS,
            "paper_count": len(written),
            "outputs": written,
        },
    )
    print(
        json.dumps(
            {"ie_dir": str(ie_dir), "paper_count": len(written), "outputs": written},
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
