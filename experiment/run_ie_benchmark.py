from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflow.common.json_utils import parse_json_object
from workflow.common.llm_chat import apply_chat_options, create_openai_client
from workflow.common.llm_config import RuntimeConfig, load_runtime_config
from workflow.ie_agent.run_extraction import validate_output


SOURCE_PROMPT = REPO_ROOT / "workflow/ie_agent/prompt.md"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "ie_evaluation"
TIMEOUT_SECONDS = 3600
MAX_OUTPUT_TOKENS = 65536


@dataclass(frozen=True)
class Condition:
    key: str
    model_id: str
    thinking: str
    reasoning_effort_override: str | None = None


@dataclass(frozen=True)
class Paper:
    stem: str
    pdf_filename: str


CONDITIONS = (
    Condition("deepseek_v4_flash_thinking", "deepseek_v4_flash", "enabled"),
    Condition("deepseek_v4_flash_non_thinking", "deepseek_v4_flash", "disabled"),
    Condition("deepseek_v4_pro_thinking", "deepseek_v4_pro", "enabled"),
    Condition("deepseek_v4_pro_non_thinking", "deepseek_v4_pro", "disabled"),
    Condition("mimo_v2_5_pro_openrouter_thinking", "mimo_v2_5_pro", "enabled"),
    Condition("gpt5_5_openrouter_medium", "gpt5.5", "enabled", "medium"),
)
GEMINI_CONDITION = Condition(
    "gemini_3_5_flash_openrouter_thinking_high",
    "gemini_3_5_flash",
    "enabled",
    "high",
)
AVAILABLE_CONDITIONS = CONDITIONS + (GEMINI_CONDITION,)

PAPERS = (
    Paper("BreastAtlas_NatGenet_2021", "BreastAtlas_NatGenet_2021.pdf"),
    Paper("CARTMultiomics_bioRxiv_2024", "CARTMultiomics_bioRxiv_2024.pdf"),
    Paper("Chu_NatMed_2023", "Chu_NatMed_2023.pdf"),
    Paper("FL_BloodCancerDiscov_2022", "FL_BloodCancerDiscov_2022.pdf"),
    Paper("OralGB_CancerSci_2023", "OralGB_CancerSci_2023.pdf"),
    Paper("Ovarian_Nature_2022", "Ovarian_Nature_2022.pdf"),
    Paper("TNBCStroma_EMBOJ_2020", "TNBCStroma_EMBOJ_2020.pdf"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() or None


def build_request_prompt(prompt_template: str, pdf_filename: str, markdown: str) -> str:
    return (
        prompt_template.rstrip("\n")
        + "\n\nProvided source PDF filename: "
        + pdf_filename
        + "\n\nPROVIDED SEARCH AGENT FULL-PAPER MARKDOWN START\n"
        + markdown
        + "\nPROVIDED SEARCH AGENT FULL-PAPER MARKDOWN END"
    )


@dataclass(frozen=True)
class TextResponse:
    content: str
    reasoning_content: str
    usage: dict[str, Any] | None
    finish_reason: str | None
    response_id: str | None
    model_returned: str | None
    request_keys: list[str]


def run_text_response(*, client, config: RuntimeConfig, messages: list[dict[str, str]],
                      timeout_seconds: float, max_output_tokens: int,
                      reasoning_effort: str | None) -> TextResponse:
    request = {"model": config.model_name, "messages": messages}
    apply_chat_options(request, temperature=None, thinking=config.thinking,
                       provider_name=config.provider_name, stream=True,
                       include_usage=True, max_tokens=max_output_tokens)
    if reasoning_effort is not None and config.thinking == "enabled":
        if config.provider_name == "openrouter":
            request["extra_body"]["reasoning"]["effort"] = reasoning_effort
        else:
            request["reasoning_effort"] = reasoning_effort
    response = client.chat.completions.create(**request, timeout=timeout_seconds)
    content_parts, reasoning_parts = [], []
    usage = finish_reason = response_id = model_returned = None
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
        if getattr(delta, "content", None):
            content_parts.append(delta.content)
        reasoning = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
        if reasoning:
            reasoning_parts.append(reasoning)
    return TextResponse("".join(content_parts), "".join(reasoning_parts), usage,
                        finish_reason, response_id, model_returned, sorted(request))


def prepare_inputs(output_root: Path, source_input_dir: Path) -> tuple[Path, Path, dict[str, Any]]:
    if output_root.resolve() == source_input_dir.resolve():
        raise ValueError("IE input and output directories must differ")
    for paper in PAPERS:
        source = source_input_dir / f"{paper.stem}.md"
        if not source.is_file():
            raise FileNotFoundError(f"Missing full-paper Markdown input: {source}")
    input_dir = output_root / "inputs" / "search_markdown"
    prompt_path = output_root / "inputs" / "prompt.md"
    input_dir.mkdir(parents=True, exist_ok=True)
    prompt_path.parent.mkdir(parents=True, exist_ok=True)

    if not SOURCE_PROMPT.exists():
        raise FileNotFoundError(SOURCE_PROMPT)
    if prompt_path.exists() and sha256_file(prompt_path) != sha256_file(SOURCE_PROMPT):
        raise ValueError(f"Existing prompt differs from source: {prompt_path}")
    if not prompt_path.exists():
        shutil.copy2(SOURCE_PROMPT, prompt_path)

    paper_metadata: dict[str, Any] = {}
    for paper in PAPERS:
        source = source_input_dir / f"{paper.stem}.md"
        destination = input_dir / source.name
        if not source.exists():
            raise FileNotFoundError(source)
        if destination.exists() and sha256_file(destination) != sha256_file(source):
            raise ValueError(f"Existing input differs from source: {destination}")
        if not destination.exists():
            shutil.copy2(source, destination)
        text = destination.read_text(encoding="utf-8")
        paper_metadata[paper.stem] = {
            "pdf_filename": paper.pdf_filename,
            "snapshot_path": str(destination),
            "characters": len(text),
            "sha256": sha256_text(text),
        }

    prompt_text = prompt_path.read_text(encoding="utf-8").rstrip("\n")
    return input_dir, prompt_path, {
        "prompt": {
            "path": str(prompt_path),
            "file_sha256": sha256_file(prompt_path),
            "normalized_content_sha256": sha256_text(prompt_text),
        },
        "papers": paper_metadata,
    }


def resolved_config(condition: Condition) -> RuntimeConfig:
    return load_runtime_config(model_id=condition.model_id, thinking=condition.thinking, stream=True)


def select_conditions(only_condition: str | None) -> tuple[Condition, ...]:
    if only_condition is None:
        return CONDITIONS
    for condition in AVAILABLE_CONDITIONS:
        if condition.key == only_condition:
            return (condition,)
    raise ValueError(f"Unknown condition: {only_condition}")


def config_metadata(config: RuntimeConfig, condition: Condition) -> dict[str, Any]:
    return {
        "model_id": config.model_id,
        "provider": config.provider_name,
        "base_url": config.base_url,
        "model_name": config.model_name,
        "api_mode": "chat_completions",
        "thinking": config.thinking,
        "configured_reasoning_effort": condition.reasoning_effort_override,
        "effective_reasoning_effort": condition.reasoning_effort_override if config.thinking == "enabled" else None,
        "stream": config.stream,
    }


def output_path(output_root: Path, condition: Condition, paper: Paper) -> Path:
    return (
        output_root
        / "ie_agent"
        / condition.key
        / f"{paper.stem}_prompt_V2_{condition.key}_direct.json"
    )


def completed_output(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    if not path.exists():
        return None, ["missing"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, [str(exc)]
    return payload, validate_output(payload).problems


def condition_run(
    condition: Condition,
    *,
    output_root: Path,
    input_dir: Path,
    prompt_path: Path,
) -> dict[str, Any]:
    config = resolved_config(condition)
    condition_dir = output_root / "ie_agent" / condition.key
    metadata_path = condition_dir / "run_metadata.json"
    condition_dir.mkdir(parents=True, exist_ok=True)
    prior: dict[str, Any] = {}
    if metadata_path.exists():
        prior_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if isinstance(prior_payload.get("papers"), dict):
            prior = prior_payload["papers"]
    metadata: dict[str, Any] = {
        "condition": condition.key,
        "created_at": utc_now(),
        "request": {
            **config_metadata(config, condition),
            "max_tokens": MAX_OUTPUT_TOKENS,
            "temperature": "provider default",
            "response_format": "prompt-only JSON instruction; no API-enforced schema",
            "messages": "one user message; no system message",
        },
        "papers": prior,
    }
    write_json(metadata_path, metadata)
    client = create_openai_client(config, timeout_seconds=TIMEOUT_SECONDS)
    prompt_template = prompt_path.read_text(encoding="utf-8")

    for paper in PAPERS:
        markdown_path = input_dir / f"{paper.stem}.md"
        markdown = markdown_path.read_text(encoding="utf-8")
        request_prompt = build_request_prompt(prompt_template, paper.pdf_filename, markdown)
        destination = output_path(output_root, condition, paper)
        existing, problems = completed_output(destination)
        prior_paper = metadata["papers"].get(paper.stem, {})
        if (
            existing is not None
            and not problems
            and prior_paper.get("status") == "completed"
            and prior_paper.get("request_prompt_sha256") == sha256_text(request_prompt)
        ):
            print(f"[{condition.key}] {paper.stem}: reuse completed output", flush=True)
            continue

        started = time.time()
        paper_metadata: dict[str, Any] = {
            "status": "running",
            "started_at": utc_now(),
            "input_path": str(markdown_path),
            "input_characters": len(markdown),
            "input_sha256": sha256_text(markdown),
            "request_prompt_characters": len(request_prompt),
            "request_prompt_sha256": sha256_text(request_prompt),
            "output_path": str(destination),
        }
        metadata["papers"][paper.stem] = paper_metadata
        write_json(metadata_path, metadata)
        print(f"[{condition.key}] {paper.stem}: request started", flush=True)

        try:
            response = run_text_response(
                client=client,
                config=config,
                messages=[{"role": "user", "content": request_prompt}],
                timeout_seconds=TIMEOUT_SECONDS,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                reasoning_effort=condition.reasoning_effort_override,
            )
            destination.with_suffix(".raw_response.txt").write_text(response.content, encoding="utf-8")
            destination.with_suffix(".reasoning.txt").write_text(response.reasoning_content, encoding="utf-8")
            if response.finish_reason != "stop":
                raise ValueError(f"IE response did not finish normally: {response.finish_reason}")
            parsed, parse_error = parse_json_object(response.content)
            validation_problems = validate_output(parsed).problems
            if parse_error:
                validation_problems.insert(0, parse_error)
            if validation_problems:
                raise ValueError("; ".join(validation_problems))
            assert parsed is not None
            write_json(destination, parsed)
            reasoning_tokens = None
            if isinstance(response.usage, dict):
                details = response.usage.get("completion_tokens_details")
                if isinstance(details, dict):
                    reasoning_tokens = details.get("reasoning_tokens")
            paper_metadata.update(
                {
                    "status": "completed",
                    "completed_at": utc_now(),
                    "elapsed_seconds": round(time.time() - started, 3),
                    "response_id": response.response_id,
                    "model_returned": response.model_returned,
                    "finish_reason": response.finish_reason,
                    "usage": response.usage,
                    "content_characters": len(response.content),
                    "content_sha256": sha256_text(response.content),
                    "reasoning_characters": len(response.reasoning_content),
                    "reasoning_sha256": sha256_text(response.reasoning_content)
                    if response.reasoning_content
                    else None,
                    "reasoning_tokens": reasoning_tokens,
                    "request_keys": response.request_keys,
                    "item_count": len(parsed["items"]),
                }
            )
            print(
                f"[{condition.key}] {paper.stem}: completed items={paper_metadata['item_count']}",
                flush=True,
            )
        except Exception as exc:
            paper_metadata.update(
                {
                    "status": "failed",
                    "failed_at": utc_now(),
                    "elapsed_seconds": round(time.time() - started, 3),
                    "error_type": type(exc).__name__,
                    "error": str(exc).replace(config.api_key, "[REDACTED]"),
                }
            )
            print(f"[{condition.key}] {paper.stem}: FAILED {type(exc).__name__}: {paper_metadata['error']}", flush=True)
        write_json(metadata_path, metadata)

    completed = sum(
        1 for paper in PAPERS if metadata["papers"].get(paper.stem, {}).get("status") == "completed"
    )
    metadata["completed_at"] = utc_now()
    metadata["coverage"] = f"{completed}/{len(PAPERS)}"
    write_json(metadata_path, metadata)
    return {"condition": condition.key, "coverage": metadata["coverage"], "metadata": str(metadata_path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing the seven named full-paper Markdown files")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--only-condition",
        choices=[condition.key for condition in AVAILABLE_CONDITIONS],
    )
    parser.add_argument("--max-workers", type=int)
    args = parser.parse_args()
    conditions = select_conditions(args.only_condition)
    max_workers = args.max_workers or len(conditions)
    output_root = args.output_root.expanduser().resolve()
    input_dir, prompt_path, inputs = prepare_inputs(output_root, args.input_dir.expanduser().resolve())

    initial_metadata = {
        "run_name": output_root.name,
        "created_at": utc_now(),
        "git_commit": git_commit(),
        "method": "current IE prompt with fixed full-paper Markdown snapshots",
        "execution": f"{len(conditions)} model condition(s) in parallel; papers serial within each condition",
        "input_source": str(args.input_dir.expanduser().resolve()),
        **inputs,
        "conditions": [
            {"key": condition.key, **config_metadata(resolved_config(condition), condition)}
            for condition in conditions
        ],
    }
    write_json(output_root / "run_metadata.json", initial_metadata)

    summaries: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                condition_run,
                condition,
                output_root=output_root,
                input_dir=input_dir,
                prompt_path=prompt_path,
            ): condition
            for condition in conditions
        }
        for future in as_completed(futures):
            summaries.append(future.result())

    summaries.sort(key=lambda item: item["condition"])
    initial_metadata["updated_at"] = utc_now()
    initial_metadata["condition_summaries"] = summaries
    write_json(output_root / "run_metadata.json", initial_metadata)
    print(json.dumps({"output_root": str(output_root), "conditions": summaries}, indent=2), flush=True)
    return 0 if all(item["coverage"] == "7/7" for item in summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
