from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflow.common.json_utils import parse_json_array
from workflow.common.llm_chat import apply_chat_options, create_openai_client
from workflow.common.llm_config import ConfigError, load_runtime_config
from workflow.common.agent_io import read_ie_paper, write_mapping_paper
from workflow.mapping_agent.run_mapping import evaluate_predictions
from workflow.mapping_agent.mapping_task import MappingCase, build_mapping_prompt, read_mapping_rule, PROMPT_PATH
from workflow.mapping_agent.run_mapping import mapping_papers_from_predictions
from workflow.mapping_agent.slot_schema import SLOT_NAMES, build_mapped_label, normalize_mapped_slots

RULE = PROMPT_PATH
DEFAULT_ROOT = REPO_ROOT / "results/evidence_ablation"


@dataclass(frozen=True)
class ModelCondition:
    key: str
    label: str
    model_id: str
    thinking: str
    reasoning_effort: str | None = None


MODELS = (
    ModelCondition("mimo_thinking", "MiMo-V2.5-Pro (thinking)", "mimo_v2_5_pro", "enabled"),
    ModelCondition("mimo_non_thinking", "MiMo-V2.5-Pro", "mimo_v2_5_pro", "disabled"),
    ModelCondition("deepseek_flash_thinking", "DeepSeek-V4-Flash (thinking)", "deepseek_v4_flash", "enabled"),
    ModelCondition("deepseek_flash_non_thinking", "DeepSeek-V4-Flash", "deepseek_v4_flash", "disabled"),
    ModelCondition("deepseek_pro_thinking", "DeepSeek-V4-Pro (thinking)", "deepseek_v4_pro", "enabled"),
    ModelCondition("deepseek_pro_non_thinking", "DeepSeek-V4-Pro", "deepseek_v4_pro", "disabled"),
    ModelCondition("gpt55", "GPT-5.5", "gpt5.5", "enabled"),
    ModelCondition("claude_opus48", "Claude Opus 4.8", "claude_opus_4_8", "enabled"),
    ModelCondition("gemini35_flash", "Gemini 3.5 Flash", "gemini_3_5_flash", "enabled"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def normalize_name(value: str) -> str:
    return " ".join(value.split()).casefold()


def check_resume(directory: Path, model: ModelCondition, prompt: str) -> None:
    if (directory / "prompt.txt").read_text() != prompt:
        raise ValueError(f"Cannot resume with a changed prompt: {directory}")
    metadata = json.loads((directory / "response_metadata.json").read_text())
    config = load_runtime_config(model_id=model.model_id, thinking=model.thinking, stream=True)
    expected = {"model_id": model.model_id, "model_requested": config.model_name,
                "provider": config.provider_name, "thinking": model.thinking,
                "reasoning_effort": model.reasoning_effort, "temperature": 0.0}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Cannot resume with changed model settings: {directory}")


def ablation_prompt(rule_text: str, cases: list[MappingCase]) -> str:
    # Apply exactly the same clarification to both arms. Explicit label claims
    # are usable; properties merely associated with a familiar name are not.
    prompt = build_mapping_prompt(rule_text, cases)
    prompt = prompt.replace(
        "If the evidence does not support a slot, return:",
        "If neither the original label nor the supplied evidence supports a slot, return:",
    )
    prompt = prompt.replace(
        "Do not fill a slot only because it is familiar from the label name.\n",
        "Use biological properties explicitly stated in original_label or the supplied evidence.\n"
        "Do not infer additional properties merely associated with a familiar subset name.\n"
        "If the supplied evidence qualifies or contradicts a traditional label, follow the evidence.\n"
        "An empty evidence array means no additional evidence was supplied; it does not negate explicit label claims.\n",
    )
    return prompt


def request(model: ModelCondition, prompt: str, output_dir: Path, timeout: float) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    started = time.monotonic()
    metadata: dict[str, Any] = {"started_at": utc_now(), "condition": model.key, "status": "running"}
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    config = None
    write_json(output_dir / "response_metadata.json", metadata)
    try:
        config = load_runtime_config(model_id=model.model_id, thinking=model.thinking, stream=True)
        metadata.update({"model_id": config.model_id, "model_requested": config.model_name,
                         "provider": config.provider_name, "thinking": config.thinking,
                         "reasoning_effort": model.reasoning_effort, "temperature": 0.0})
        kwargs: dict[str, Any] = {"model": config.model_name, "messages": [{"role": "user", "content": prompt}]}
        apply_chat_options(kwargs, temperature=0.0, thinking=config.thinking,
                           provider_name=config.provider_name, stream=True, include_usage=True)
        if model.reasoning_effort is not None:
            if config.provider_name != "openrouter":
                raise ValueError("This explicit reasoning-effort setting requires the configured OpenRouter provider.")
            kwargs["extra_body"] = {"reasoning": {"effort": model.reasoning_effort}}
        metadata["request_options"] = {k: v for k, v in kwargs.items() if k != "messages"}
        with create_openai_client(config, timeout_seconds=timeout).with_options(max_retries=0) as client:
            response = client.chat.completions.create(**kwargs)
            for chunk in response:
                if chunk.id:
                    metadata["response_id"] = chunk.id
                if chunk.model:
                    metadata["model_returned"] = chunk.model
                if chunk.usage is not None:
                    metadata["usage"] = chunk.usage.model_dump(mode="json")
                for choice in chunk.choices:
                    if choice.finish_reason:
                        metadata["finish_reason"] = choice.finish_reason
                    if choice.delta.content:
                        content_parts.append(choice.delta.content)
                    reasoning = getattr(choice.delta, "reasoning_content", None) or getattr(choice.delta, "reasoning", None)
                    if reasoning:
                        reasoning_parts.append(reasoning)
            metadata["status"] = "completed" if content_parts else "empty_response"
    except Exception as exc:
        message = str(exc).replace(config.api_key, "[REDACTED]") if config else str(exc)
        metadata.update(status="unavailable" if isinstance(exc, ConfigError) else "failed", error_type=type(exc).__name__, error=message[:2000])
    metadata.update(completed_at=utc_now(), elapsed_seconds=round(time.monotonic() - started, 3))
    (output_dir / "raw_response.txt").write_text("".join(content_parts), encoding="utf-8")
    (output_dir / "reasoning.txt").write_text("".join(reasoning_parts), encoding="utf-8")
    write_json(output_dir / "response_metadata.json", metadata)
    return metadata


def prepare(root: Path, cases_path: Path, gold_path: Path) -> None:
    for source in (cases_path, gold_path, RULE):
        if not source.is_file():
            raise FileNotFoundError(f"Missing evidence-ablation input: {source}")
    if cases_path.resolve() == gold_path.resolve():
        raise ValueError("Population inputs and reference labels must be separate files")
    for source, name in ((cases_path, "cases.json"), (gold_path, "gold.json")):
        if source.resolve() == (root / "inputs" / name).resolve():
            raise ValueError(f"Input must be separate from its frozen experiment output: {source}")
    paper = read_ie_paper(cases_path)
    gold = json.loads(gold_path.read_text())
    if len(paper.items) != 16 or len(gold.get("items", [])) != 16:
        raise ValueError("Evidence ablation requires the 16 selected cases and 16 reference records")
    if [item.get("case_id") for item in gold["items"]] != list(range(16)):
        raise ValueError("Reference case IDs must match the input order from 0 through 15")
    for item in gold["items"]:
        if not isinstance(item.get("project_aligned_label"), str) or not item["project_aligned_label"].strip():
            raise ValueError("Each reference needs a project_aligned_label")
        modules = item.get("expected_modules")
        if not isinstance(modules, dict) or set(modules) != set(SLOT_NAMES) or any(value is not None and not isinstance(value, str) for value in modules.values()):
            raise ValueError("Each reference needs the seven expected_modules with string or null values")
    inputs = root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    for source, name in ((cases_path, "cases.json"),
                         (gold_path, "gold.json"),
                         (RULE, "prompt.md")):
        destination = inputs / name
        if destination.exists():
            if destination.read_bytes() != source.read_bytes():
                raise ValueError(f"Frozen input differs from source: {destination}")
        else:
            shutil.copy2(source, destination)
    metadata_path = root / "run_metadata.json"
    if not metadata_path.exists():
        write_json(metadata_path, {
            "run_name": root.name, "created_at": utc_now(), "experiment": "Exp1: label-only vs evidence-assisted mapping",
            "source_cases": str(cases_path), "source_gold": str(gold_path), "source_rule": str(RULE),
            "case_count": 16, "arms": ["label_only", "label_plus_evidence"], "repetitions": 1,
            "input_batching": "all 16 cases in one request per arm and configuration",
            "judge_used": False, "gold_visible_during_inference": False,
            "primary_metric": "case-insensitive whitespace-normalized exact match against project_aligned_label",
            "paired_contrast": "only evidence arrays differ; identical original labels, rule and prompt instructions",
            "prompt_clarification": "Both arms may use explicit label claims; unsupported associated properties remain prohibited.",
            "scope": "Guideline Table 7 examples; not independent real-paper validation. Label-only lacks some information in the full-evidence gold.",
            "conditions": [vars(model) for model in MODELS],
        })


def score_predictions(parsed: list[dict[str, Any]], gold: dict[str, Any]) -> list[dict[str, Any]]:
    predictions = {item["case_id"]: item for item in parsed}
    scored = []
    for target in gold["items"]:
        prediction = predictions[target["case_id"]]
        slots, problems = normalize_mapped_slots(prediction["mapped_slots"])
        if problems:
            raise ValueError(f"Cannot score invalid slots: {problems}")
        label = build_mapped_label(slots)
        differences = []
        for slot in SLOT_NAMES:
            actual = slots[slot]["module"]
            expected = target["expected_modules"][slot]
            if normalize_name(actual or "") != normalize_name(expected or ""):
                differences.append({"slot": slot, "expected": expected, "predicted": actual,
                                    "type": "omitted" if actual is None else "extra" if expected is None else "different"})
        scored.append({"case_id": target["case_id"], "predicted_label": label,
                       "gold_label": target["project_aligned_label"],
                       "exact_match": normalize_name(label) == normalize_name(target["project_aligned_label"]),
                       "slot_differences": differences})
    return scored


def run_arm(model: ModelCondition, arm: str, root: Path) -> dict[str, Any]:
    directory = root / model.key / arm
    status_path = directory / "status.json"
    source = json.loads((root / "inputs/cases.json").read_text())
    if arm == "label_only":
        for item in source["items"]:
            item["evidence"] = []
    ie_path = directory / "ie_agent/table7_nomenclature.json"
    write_json(ie_path, source)
    paper = read_ie_paper(ie_path)
    cases = [MappingCase(i, item.original_label, item.evidence) for i, item in enumerate(paper.items)]
    prompt = ablation_prompt(read_mapping_rule(root / "inputs/prompt.md"), cases)
    if status_path.exists():
        check_resume(directory, model, prompt)
        return json.loads(status_path.read_text())
    metadata = request(model, prompt, directory, 900)
    parsed, parse_error = parse_json_array((directory / "raw_response.txt").read_text())
    validation = evaluate_predictions(parsed, cases)
    result: dict[str, Any] = {"condition": model.key, "arm": arm, "status": metadata["status"],
                              "parse_error": parse_error, "validation": validation, "response_metadata": metadata}
    if metadata["status"] == "completed" and validation["schema_ok"] and metadata.get("finish_reason") != "length":
        output = mapping_papers_from_predictions([paper], parsed, rule_version="prompt")[0]
        output_path = directory / "mapping_agent/table7_nomenclature.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_mapping_paper(output_path, output)
        gold = json.loads((root / "inputs/gold.json").read_text())
        scored = score_predictions(parsed, gold)
        write_json(directory / "evaluation.json", scored)
        result.update(status="valid", correct=sum(row["exact_match"] for row in scored), n=len(scored))
        result["accuracy"] = result["correct"] / result["n"]
    elif metadata["status"] == "completed":
        result["status"] = "invalid_output"
    write_json(status_path, result)
    print(json.dumps({"condition": model.key, "arm": arm, "status": result["status"],
                      "correct": result.get("correct"), "elapsed_seconds": metadata["elapsed_seconds"]}), flush=True)
    return result


def summarize(root: Path) -> None:
    rows = []
    paired = []
    for model in MODELS:
        row: dict[str, Any] = {"condition": model.key, "model": model.label}
        for arm in ("label_only", "label_plus_evidence"):
            status_path = root / model.key / arm / "status.json"
            status = json.loads(status_path.read_text()) if status_path.exists() else {"status": "not_run"}
            row[arm + "_status"] = status["status"]
            row[arm + "_correct"] = status.get("correct")
            row[arm + "_accuracy"] = status.get("accuracy")
            response_metadata = status.get("response_metadata", {})
            row[arm + "_requested_model"] = response_metadata.get("model_requested")
            row[arm + "_returned_model"] = response_metadata.get("model_returned")
            if model.key.startswith("deepseek_flash") and response_metadata.get("model_returned") == "deepseek-flash":
                row["model"] = "DeepSeek-V4.1-Flash" + (" (thinking)" if model.thinking == "enabled" else "")
                row["version_note"] = "Provider routes retired V4-Flash alias to V4.1-Flash; not original paper model."
            usage = response_metadata.get("usage") or {}
            row[arm + "_tokens"] = usage.get("total_tokens")
            row[arm + "_cost"] = usage.get("cost")
        if all(row[arm + "_status"] == "valid" for arm in ("label_only", "label_plus_evidence")):
            before = json.loads((root / model.key / "label_only/evaluation.json").read_text())
            after = json.loads((root / model.key / "label_plus_evidence/evaluation.json").read_text())
            row["delta_percentage_points"] = 100 * (row["label_plus_evidence_accuracy"] - row["label_only_accuracy"])
            row["corrected"] = sum(not x["exact_match"] and y["exact_match"] for x, y in zip(before, after, strict=True))
            row["regressed"] = sum(x["exact_match"] and not y["exact_match"] for x, y in zip(before, after, strict=True))
            paired.extend({"condition": model.key, "case_id": x["case_id"], "gold_label": x["gold_label"],
                           "label_only": x["predicted_label"], "label_plus_evidence": y["predicted_label"],
                           "before_correct": x["exact_match"], "after_correct": y["exact_match"]}
                          for x, y in zip(before, after, strict=True))
        rows.append(row)
    write_json(root / "summary.json", rows)
    write_json(root / "paired_cases.json", paired)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (root / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["run", "summarize"])
    parser.add_argument("--cases", type=Path, help="Selected 16-case IE JSON input; required for run")
    parser.add_argument("--gold", type=Path, help="Reference JSON with case_id/project_aligned_label/expected_modules; required for run")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--models", nargs="*", choices=[model.key for model in MODELS])
    args = parser.parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    models = [model for model in MODELS if not args.models or model.key in args.models]
    if args.action == "run":
        if args.cases is None or args.gold is None:
            parser.error("run requires --cases and --gold")
        prepare(root, args.cases.expanduser().resolve(), args.gold.expanduser().resolve())
        jobs = [(model, arm) for model in models for arm in ("label_only", "label_plus_evidence")]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_arm, model, arm, root) for model, arm in jobs]
            for future in as_completed(futures):
                future.result()
        summarize(root)
    else:
        summarize(root)


if __name__ == "__main__":
    main()
