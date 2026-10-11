"""Shared execution and frozen-file helpers for the final Mapping/Judge experiments."""
from __future__ import annotations

import json
from pathlib import Path

from workflow.common.json_utils import parse_json_array
from workflow.common.llm_chat import create_openai_client
from workflow.common.llm_config import load_runtime_config
from workflow.common.agent_io import read_ie_paper, write_mapping_paper
from workflow.mapping_agent.run_mapping import evaluate_predictions
from workflow.mapping_agent.mapping_task import build_mapping_prompt, read_mapping_rule, PROMPT_PATH
from workflow.mapping_agent.run_mapping import mapping_cases_from_ie_papers, run_chat as map_chat, mapping_papers_from_predictions


def save(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def run_paper(root: Path, source: Path, model: str, thinking: str, rule_path: Path | None = None) -> dict:
    stem = source.stem
    output = root / "calls" / stem
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"
    previous = json.loads(status_path.read_text())
    if previous["status"] == "completed":
        return previous
    if previous["status"] != "ie_completed":
        raise ValueError(f"Retained incomplete/failed call requires explicit inspection: {stem}")
    status = {"paper": stem, "status": "running", "agent": "Mapping"}
    save(status_path, status)
    config = None
    try:
        config = load_runtime_config(model_id=model, thinking=thinking, stream=True)
        paper = read_ie_paper(source)
        status.update(ie_items=len(paper.items))
        save(status_path, status)
        print(f"[Mapping start] {stem}: {len(paper.items)} populations", flush=True)
        cases = mapping_cases_from_ie_papers([paper])
        rule_path = rule_path or PROMPT_PATH
        prompt = build_mapping_prompt(read_mapping_rule(rule_path), cases)
        (output / "mapping_prompt.txt").write_text(prompt)
        with create_openai_client(config, timeout_seconds=3600).with_options(max_retries=0) as client:
            content, reasoning, metadata = map_chat(client=client, config=config, prompt=prompt, timeout_seconds=3600, temperature=0.0)
        (output / "mapping_raw_response.txt").write_text(content)
        (output / "mapping_reasoning.txt").write_text(reasoning)
        save(output / "mapping_response_metadata.json", metadata)
        parsed, parse_error = parse_json_array(content)
        validation = evaluate_predictions(parsed, cases)
        save(output / "mapping_validation.json", {**validation, "parse_error": parse_error})
        if metadata.get("finish_reason") != "stop" or not validation["schema_ok"]:
            raise ValueError("Mapping completion or schema failed; retained raw response.")
        mapped = mapping_papers_from_predictions([paper], parsed, rule_version=rule_path.stem)[0]
        write_mapping_paper(root / "mapping_agent" / f"{stem}.json", mapped)
        status.update(status="completed", agent="Mapping", mapped_items=len(mapped.items))
    except Exception as exc:
        status.update(status="failed", error=str(exc).replace(config.api_key, "[REDACTED]") if config else str(exc))
    save(status_path, status)
    print(json.dumps(status, ensure_ascii=False), flush=True)
    return status


def copy_frozen(source: Path, target: Path) -> None:
    content = source.read_bytes()
    if target.exists():
        if target.read_bytes() != content:
            raise ValueError(f"Retained artifact differs; refusing overwrite: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
