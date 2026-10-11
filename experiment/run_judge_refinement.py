"""One frozen Flash review and optional Mapping revision per paper."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workflow.common.json_utils import parse_json_array
from workflow.common.prompts import read_prompt_section
from workflow.common.llm_chat import create_openai_client
from workflow.common.llm_config import load_runtime_config
from workflow.common.agent_io import ExtractedLabel, IEPaper, MappingPaper, read_ie_paper, read_mapping_paper, write_mapping_paper
from workflow.mapping_agent.run_mapping import evaluate_predictions
from workflow.mapping_agent.mapping_task import PROMPT_PATH as MAPPING_PROMPT_PATH, build_mapping_prompt, read_mapping_rule
from experiment.experiment_utils import copy_frozen, save
from experiment.run_mapping_benchmark import lineage_comparison, validate_completed as validate_mapping
from workflow.mapping_agent.run_mapping import mapping_cases_from_ie_papers, run_chat, mapping_papers_from_predictions
from workflow.judge_agent.judge_task import DEFAULT_USER_TEMPLATE_PATH, build_user_prompt, mapping_records, validate_feedback


def prepare(root: Path, mapping_arm: str, name: str, judge_prompt: Path = DEFAULT_USER_TEMPLATE_PATH, refinement_prompt: Path = MAPPING_PROMPT_PATH) -> Path:
    review = root / "reviews" / name
    source_metadata = json.loads((root / "run_metadata.json").read_text())
    assert source_metadata["model_id"] == "deepseek_v4_flash"
    papers = source_metadata["papers"]
    path = review / "run_metadata.json"
    if path.exists():
        retained = json.loads(path.read_text())["configuration"]
        if [item["paper"] for item in retained["inputs"]] != papers:
            raise ValueError("Selected papers differ from the retained review")
    rule_source = Path(source_metadata["rules"][mapping_arm]["frozen"])
    rule = review / "inputs" / rule_source.name
    sources = {"rule": rule_source, "judge_prompt.txt": judge_prompt, "refinement_prompt.txt": refinement_prompt}
    for key, source in sources.items():
        copy_frozen(source, rule if key == "rule" else review / "inputs" / key)
    inputs = []
    for paper in papers:
        source = root / mapping_arm
        assert json.loads((source / "calls" / paper / "status.json").read_text())["status"] == "completed"
        validate_mapping(source, paper, rule_source)
        for agent in ("ie_agent", "mapping_agent"):
            copy_frozen(source / agent / f"{paper}.json", review / "inputs" / agent / f"{paper}.json")
        inputs.append({"paper": paper, "items": len(read_ie_paper(source / "ie_agent" / f"{paper}.json").items)})
        status = review / "calls" / paper / "status.json"
        if not status.exists():
            save(status, {"paper": paper, "status": "prepared"})
    configuration = {"mapping_root": str(root), "mapping_arm": mapping_arm, "sources": {key: str(path) for key, path in sources.items()}, "frozen_rule": str(rule), "inputs": inputs,
                     "model_id": "deepseek_v4_flash", "thinking": "enabled", "temperature": 0.0, "stream": True, "concurrency": 2, "max_review_rounds": 1, "gold_visible_to_models": False}
    if path.exists():
        assert json.loads(path.read_text())["configuration"] == configuration, "Retained review configuration changed"
    else:
        save(path, {"created_at": datetime.now(timezone.utc).isoformat(), "status": "prepared", "configuration": configuration})
    return review


def refinement_inputs(paper: MappingPaper, feedback: list[dict]) -> tuple[list[dict], list[int], IEPaper]:
    records = mapping_records(paper)
    flagged = [item["case_id"] for item in feedback if item["decision"] == "revise"]
    by_id = {item["case_id"]: item for item in feedback}
    reviewed = [{**records[index], "case_id": local, "original_case_id": index, "feedback": by_id[index]["feedback"]} for local, index in enumerate(flagged)]
    subset = IEPaper(paper.title, [ExtractedLabel(paper.items[index].original_label, paper.items[index].evidence) for index in flagged])
    return reviewed, flagged, subset


def merge_revision(paper: MappingPaper, flagged: list[int], subset: IEPaper, predictions: list) -> MappingPaper:
    revised = mapping_papers_from_predictions([subset], predictions, rule_version=paper.rule_version)[0]
    items = list(paper.items)
    for index, item in zip(flagged, revised.items):
        items[index] = item
    return MappingPaper(paper.title, paper.rule_version, items)


def model_call(call: Path, agent: str, prompt: str) -> str:
    (call / f"{agent}_prompt.txt").write_text(prompt)
    save(call / f"{agent}_status.json", {"status": "running"})
    config = load_runtime_config(model_id="deepseek_v4_flash", thinking="enabled", stream=True)
    try:
        with create_openai_client(config, timeout_seconds=3600).with_options(max_retries=0) as client:
            content, reasoning, metadata = run_chat(client=client, config=config, prompt=prompt, timeout_seconds=3600, temperature=0.0)
    except Exception as exc:
        raise RuntimeError(str(exc).replace(config.api_key, "[REDACTED]")) from None
    (call / f"{agent}_raw_response.txt").write_text(content)
    (call / f"{agent}_reasoning.txt").write_text(reasoning)
    save(call / f"{agent}_response_metadata.json", metadata)
    if metadata.get("finish_reason") != "stop":
        raise ValueError(f"{agent} did not finish normally")
    return content


def validate_completed(review: Path, paper: str) -> None:
    call = review / "calls" / paper
    original = read_mapping_paper(review / "inputs/mapping_agent" / f"{paper}.json")
    feedback = validate_feedback(json.loads((call / "judge_parsed_response.json").read_text()), len(original.items))
    _, flagged, subset = refinement_inputs(original, feedback)
    predictions = json.loads((call / "refinement_parsed_response.json").read_text()) if flagged else []
    for agent in ("judge", "refinement") if flagged else ("judge",):
        assert json.loads((call / f"{agent}_response_metadata.json").read_text())["finish_reason"] == "stop"
        assert json.loads((call / f"{agent}_validation.json").read_text())["schema_ok"]
    if flagged:
        assert evaluate_predictions(predictions, mapping_cases_from_ie_papers([subset]))["schema_ok"]
    relation = [{"case_id": local, "original_case_id": index} for local, index in enumerate(flagged)]
    assert json.loads((call / "refinement_case_map.json").read_text()) == relation
    expected = merge_revision(original, flagged, subset, predictions)
    assert read_mapping_paper(review / "mapping_agent" / f"{paper}.json") == expected
    ie = read_ie_paper(review / "inputs/ie_agent" / f"{paper}.json")
    assert [(item.original_label, item.evidence) for item in expected.items] == [(item.original_label, item.evidence) for item in ie.items]


def run_one(review: Path, paper: str, rule: Path) -> dict:
    call = review / "calls" / paper
    status_path = call / "status.json"
    status = json.loads(status_path.read_text())
    if status["status"] == "completed":
        validate_completed(review, paper)
        return status
    if status["status"] != "prepared":
        raise ValueError(f"Retained unfinished/failed review needs inspection: {paper}")
    agent = "judge"
    status.update(status="running", agent=agent)
    save(status_path, status)
    try:
        original = read_mapping_paper(review / "inputs/mapping_agent" / f"{paper}.json")
        rule_text = read_mapping_rule(rule)
        prompt = build_user_prompt(rule_text=rule_text, records=mapping_records(original), prompt_path=review / "inputs/judge_prompt.txt")
        parsed, error = parse_json_array(model_call(call, agent, prompt))
        save(call / "judge_parsed_response.json", parsed)
        try:
            feedback = validate_feedback(parsed, len(original.items))
        except ValueError as exc:
            save(call / "judge_validation.json", {"schema_ok": False, "parse_error": error, "error": str(exc)})
            raise
        save(call / "judge_validation.json", {"schema_ok": True, "parse_error": error})
        save(call / "judge_status.json", {"status": "completed"})
        reviewed, flagged, subset = refinement_inputs(original, feedback)
        save(call / "refinement_case_map.json", [{"case_id": local, "original_case_id": index} for local, index in enumerate(flagged)])
        predictions = []
        if flagged:
            agent = "refinement"
            status.update(agent=agent)
            save(status_path, status)
            cases = mapping_cases_from_ie_papers([subset])
            prompt = read_prompt_section(review / "inputs/refinement_prompt.txt", "Refinement").replace("{{RULE_TEXT}}", rule_text).replace("{{REVIEW_RECORDS_JSON}}", json.dumps(reviewed, ensure_ascii=False, indent=2)).replace("{{MAPPING_PROMPT}}", build_mapping_prompt(rule_text, cases))
            predictions, error = parse_json_array(model_call(call, agent, prompt))
            save(call / "refinement_parsed_response.json", predictions)
            validation = evaluate_predictions(predictions, cases)
            save(call / "refinement_validation.json", {**validation, "parse_error": error})
            if not validation["schema_ok"]:
                raise ValueError("Refinement response failed Mapping validation")
            save(call / "refinement_status.json", {"status": "completed"})
        else:
            save(call / "refinement_status.json", {"status": "skipped", "reason": "No records flagged for revision"})
        write_mapping_paper(review / "mapping_agent" / f"{paper}.json", merge_revision(original, flagged, subset, predictions))
        validate_completed(review, paper)
        status.update(status="completed", mapped_items=len(original.items), revised_candidates=len(flagged), uncertain=sum(item["decision"] == "uncertain" for item in feedback))
    except Exception as exc:
        status.update(status="failed", agent=agent, error=str(exc))
        save(call / f"{agent}_status.json", {"status": "failed", "error": str(exc)})
    save(status_path, status)
    print(json.dumps(status, ensure_ascii=False), flush=True)
    return status


def summarize(review: Path) -> dict:
    path = review / "run_metadata.json"
    metadata = json.loads(path.read_text())
    mapping_root = Path(metadata["configuration"]["mapping_root"])
    statuses, cases, models = [], [], set()
    completed_calls = 0
    for paper in (item["paper"] for item in metadata["configuration"]["inputs"]):
        call = review / "calls" / paper
        status = json.loads((call / "status.json").read_text())
        statuses.append(status)
        for agent in ("judge", "refinement"):
            response_path = call / f"{agent}_response_metadata.json"
            if response_path.exists():
                models.add(json.loads(response_path.read_text())["model_returned"])
            stage_status = call / f"{agent}_status.json"
            completed_calls += stage_status.exists() and json.loads(stage_status.read_text())["status"] == "completed"
        before = read_mapping_paper(review / "inputs/mapping_agent" / f"{paper}.json")
        after, feedback = None, {}
        if status["status"] == "completed":
            validate_completed(review, paper)
            after = read_mapping_paper(review / "mapping_agent" / f"{paper}.json")
            feedback = {item["case_id"]: item for item in json.loads((call / "judge_parsed_response.json").read_text())}
        gold = json.loads((mapping_root / "inputs/label_evidence" / f"{paper}.json").read_text())["items"]
        assert len(gold) == len(before.items)
        for index, item in enumerate(gold):
            assert (item["original_label"].strip(), [value.strip() for value in item["evidence"]]) == (before.items[index].original_label, before.items[index].evidence)
            old, new = before.items[index].mapped_label, after.items[index].mapped_label if after else None
            raw = lambda value: re.sub(r"\s+", "", value) == re.sub(r"\s+", "", item["golden_map_label"]) if value is not None else None
            old_normalized = lineage_comparison(item, old)
            new_normalized = lineage_comparison(item, new)
            cases.append({"paper": paper, "index": index, "original_label": item["original_label"], "golden_map_label": item["golden_map_label"],
                          "before": old, "after": new, "decision": feedback.get(index, {}).get("decision"), "feedback": feedback.get(index, {}).get("feedback"),
                          "raw_before": raw(old), "raw_after": raw(new), "lineage_normalized_reference": old_normalized["lineage_normalized_reference"], "lineage_source": old_normalized["lineage_source"],
                          "lineage_normalized_before": old_normalized["lineage_normalized_exact"], "lineage_normalized_after": new_normalized["lineage_normalized_exact"]})
    complete = all(status["status"] == "completed" for status in statuses)
    scores = {}
    for metric in ("raw", "lineage_normalized"):
        pairs = [(case[f"{metric}_before"], case[f"{metric}_after"]) for case in cases]
        scores[metric] = {"before": sum(old is True for old, _ in pairs), "after": sum(new is True for _, new in pairs) if complete else None,
                          "corrected": sum(old is False and new is True for old, new in pairs) if complete else None,
                          "harmed": sum(old is True and new is False for old, new in pairs) if complete else None,
                          "unchanged_match_status": sum(old == new and old is not None for old, new in pairs) if complete else None,
                          "unresolved": sum(old is None or new is None for old, new in pairs), "total": len(cases)}
    execution = {"updated_at": datetime.now(timezone.utc).isoformat(), "completed_papers": sum(status["status"] == "completed" for status in statuses), "completed_model_calls": completed_calls, "actual_models_returned": sorted(models)}
    metadata.update(status="completed" if complete else "incomplete", execution=execution)
    save(review / "execution_status.json", statuses)
    save(review / "comparison.json", {"metric": "Provided-reference full-name concordance, not biological accuracy", "complete": complete, "scores": scores,
                                      "changed_labels": sum(case["before"] != case["after"] for case in cases) if complete else None, "cases": cases})
    save(path, metadata)
    return {"status": metadata["status"], **execution, "scores": scores}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--mapping-arm", default="v5")
    parser.add_argument("--review-name")
    parser.add_argument("--judge-prompt", type=Path, default=DEFAULT_USER_TEMPLATE_PATH)
    parser.add_argument("--refinement-prompt", type=Path, default=MAPPING_PROMPT_PATH)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not all((args.root, args.mapping_arm, args.review_name)):
        parser.error("--root and --review-name are required")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", value) for value in (args.mapping_arm, args.review_name)):
        parser.error("Arm and review names must use letters, digits, underscores or hyphens")
    review = prepare(args.root.resolve(), args.mapping_arm, args.review_name, args.judge_prompt.resolve(), args.refinement_prompt.resolve())
    if not args.run:
        print("Review inputs and prompts frozen; no model called")
        return
    metadata = json.loads((review / "run_metadata.json").read_text())
    metadata.update(status="running", started_at=datetime.now(timezone.utc).isoformat())
    save(review / "run_metadata.json", metadata)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending = [pool.submit(run_one, review, item["paper"], Path(metadata["configuration"]["frozen_rule"])) for item in metadata["configuration"]["inputs"]]
            for future in as_completed(pending):
                future.result()
    finally:
        result = summarize(review)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
