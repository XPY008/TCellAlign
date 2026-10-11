"""Mapping-only Flash comparison on frozen label/evidence inputs, with gold kept out of prompts."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workflow.common.agent_io import read_ie_paper, read_mapping_paper
from workflow.mapping_agent.mapping_task import PROMPT_PATH
from experiment.experiment_utils import copy_frozen, run_paper, save
from workflow.mapping_agent.slot_schema import build_mapped_label

PAPERS = ("01_ZC3H12C_T_cell_persistence", "02_Glioblastoma_bystander_TRM_like", "03_Age_CMV_immune_checkpoint_blockade",
          "04_Schizophrenia_T_cell_adaptations", "05_CNS_viral_infections_TRM", "06_Multiple_myeloma_CD3xBCMA")


def prepare(root: Path, rules: dict[str, Path], input_dir: Path, papers: list[str] | tuple[str, ...] = PAPERS) -> None:
    if root.resolve() == input_dir.resolve():
        raise ValueError("Mapping input and output directories must differ")
    for paper in papers:
        source = input_dir / f"{paper}.json"
        if not source.is_file():
            raise FileNotFoundError(f"Missing label/evidence input: {source}")
    for rule in rules.values():
        if not rule.is_file():
            raise FileNotFoundError(f"Missing Mapping prompt: {rule}")
    metadata_path = root / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else None
    if metadata is not None and metadata["papers"] != list(papers):
        raise ValueError("Selected papers differ from the retained Mapping run")
    inputs = []
    for paper in papers:
        source = input_dir / f"{paper}.json"
        frozen = root / "inputs/label_evidence" / source.name
        copy_frozen(source, frozen)
        payload = json.loads(frozen.read_text())
        assert all(isinstance(item.get("golden_map_label"), str) and item["golden_map_label"].strip() for item in payload["items"])
        sanitized = asdict(read_ie_paper(frozen))
        inputs.append({"paper": paper, "source": str(source), "frozen": str(frozen), "items": len(sanitized["items"])})
        for arm in rules:
            target = root / arm / "ie_agent" / source.name
            if target.exists():
                assert json.loads(target.read_text()) == sanitized, f"Retained model input changed: {target}"
            else:
                save(target, sanitized)
            status = root / arm / "calls" / paper / "status.json"
            if not status.exists():
                save(status, {"paper": paper, "status": "ie_completed", "ie_items": len(sanitized["items"]), "ie_model_called": False})
    metadata = metadata if metadata is not None else {
        "created_at": datetime.now(timezone.utc).isoformat(), "status": "prepared",
        "model_id": "deepseek_v4_flash", "thinking": "enabled", "temperature": 0.0, "stream": True, "concurrency": 2,
        "agents": ["Mapping"], "ie_model_called": False, "judge_run": False,
        "papers": list(papers), "inputs": inputs, "rules": {}, "log_path": str(ROOT / "logs" / f"{root.name}.log"),
    }
    assert metadata["inputs"] == inputs and metadata["model_id"] == "deepseek_v4_flash"
    for arm, source in rules.items():
        frozen = root / arm / "inputs" / source.name
        copy_frozen(source, frozen)
        record = {"source": str(source), "frozen": str(frozen), "rule_version": source.stem}
        assert arm not in metadata["rules"] or metadata["rules"][arm] == record, f"Retained rule source changed: {arm}"
        metadata["rules"][arm] = record
    metadata["expected_mapping_calls"] = len(metadata["rules"]) * len(papers)
    metadata["expected_mapping_items"] = len(metadata["rules"]) * sum(item["items"] for item in inputs)
    save(metadata_path, metadata)


def validate_completed(root: Path, paper: str, rule: Path) -> dict:
    original = read_ie_paper(root / "ie_agent" / f"{paper}.json")
    mapped = read_mapping_paper(root / "mapping_agent" / f"{paper}.json")
    assert [(x.original_label, x.evidence) for x in original.items] == [(x.original_label, x.evidence) for x in mapped.items]
    assert mapped.rule_version == rule.stem
    assert all(item.mapped_label == build_mapped_label(item.mapped_slots) for item in mapped.items)
    call = root / "calls" / paper
    response = json.loads((call / "mapping_response_metadata.json").read_text())
    validation = json.loads((call / "mapping_validation.json").read_text())
    assert response["finish_reason"] == "stop" and validation["schema_ok"]
    return response


def run_one(root: Path, paper: str, rule: Path) -> dict:
    status = json.loads((root / "calls" / paper / "status.json").read_text())
    if status["status"] == "completed":
        validate_completed(root, paper, rule)
        return status
    if status["status"] != "ie_completed":
        raise ValueError(f"Retained unfinished/failed call needs inspection: {root.name}/{paper}")
    return run_paper(root, root / "ie_agent" / f"{paper}.json", "deepseek_v4_flash", "enabled", rule_path=rule)


def lineage_comparison(item: dict, prediction: str | None) -> dict:
    def canonical(text: str) -> str:
        return re.sub(r"(?<![A-Za-z0-9])(CD[48])(?:\s*[+⁺])?(?![A-Za-z0-9+⁺]|\s*[-−⁻])", r"\1+", text)

    reference = canonical(item["golden_map_label"])
    explicit = re.search(r"(?<![A-Za-z0-9])CD[48](?:[+⁺\-−⁻]|\s|dim\b|$)", reference)
    if explicit:
        source = "golden_map_label"
    else:
        for source, text in (("original_label", item["original_label"]), ("evidence", "\n".join(item["evidence"]))):
            lineages = set(re.findall(r"(?<![A-Za-z0-9])(CD[48])\+", canonical(text)))
            if lineages:
                break
        if len(lineages) == 1:
            reference = next(iter(lineages)) + "+ " + reference
        elif not lineages:
            source = "not_specified_compare_original"
        else:
            reference = None
            source = "unresolved_" + source + "_ambiguous"
    exact = re.sub(r"\s+", "", canonical(prediction)) == re.sub(r"\s+", "", reference) if prediction is not None and reference is not None else None
    return {"lineage_normalized_reference": reference, "lineage_source": source, "lineage_normalized_exact": exact}


def summarize(root: Path) -> dict:
    metadata_path = root / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    statuses, models = [], set()
    comparison = {"metric": "Provided-reference full-name concordance; not biological accuracy", "normalization": "Raw score ignores whitespace only; lineage-normalized score follows the recorded policy", "arms": {}}
    comparison["lineage_normalized_policy"] = {
        "authorization": "User approved lineage notation normalization for the main comparison, retaining raw full-name concordance.",
        "reference": "Preserve explicit CD4/CD8 lineage anywhere in gold, including marker-prefixed or dim/negative qualifiers; otherwise use unique positive lineage in original_label, then unique evidence lineage. Normalize bare CD4/CD8 and ASCII/superscript positive notation in both comparison copies; preserve dim, negative and all other markers.",
        "matching": "Require the complete prediction to equal the comparison reference ignoring whitespace. Preserve markers and all other modules; never strip predicted lineage.",
        "unresolved": "User approved original full-name comparison when lineage is not specified in gold, label or evidence. Ambiguous inferred lineage remains unresolved, retained in the total and not counted as a match. Raw gold and raw exact remain unchanged.",
    }
    for arm, record in metadata["rules"].items():
        cases = []
        for paper in metadata["papers"]:
            status = json.loads((root / arm / "calls" / paper / "status.json").read_text())
            statuses.append({"arm": arm, **status})
            mapped = None
            if status["status"] == "completed":
                response = validate_completed(root / arm, paper, Path(record["frozen"]))
                models.add(response["model_returned"])
                mapped = read_mapping_paper(root / arm / "mapping_agent" / f"{paper}.json")
            gold = json.loads((root / "inputs/label_evidence" / f"{paper}.json").read_text())["items"]
            for index, item in enumerate(gold):
                prediction = mapped.items[index].mapped_label if mapped else None
                exact = re.sub(r"\s+", "", prediction) == re.sub(r"\s+", "", item["golden_map_label"]) if prediction is not None else None
                cases.append({"paper": paper, "index": index, "original_label": item["original_label"], "golden_map_label": item["golden_map_label"], "prediction": prediction, "exact": exact, **lineage_comparison(item, prediction)})
        complete = all(case["prediction"] is not None for case in cases)
        comparison["arms"][arm] = {"complete": complete, "exact": sum(case["exact"] is True for case in cases) if complete else None, "total": len(cases),
                                   "lineage_normalized_exact": sum(case["lineage_normalized_exact"] is True for case in cases) if complete else None,
                                   "lineage_normalized_total": len(cases), "lineage_unresolved": sum(case["lineage_normalized_reference"] is None for case in cases), "cases": cases}
    completed = sum(status["status"] == "completed" for status in statuses)
    execution = {"updated_at": datetime.now(timezone.utc).isoformat(), "completed_calls": completed,
                 "mapped_items": sum(status.get("mapped_items", 0) for status in statuses if status["status"] == "completed"), "actual_models_returned": sorted(models)}
    metadata.update(status="completed" if completed == metadata["expected_mapping_calls"] else "incomplete", execution=execution)
    save(root / "execution_status.json", statuses)
    save(root / "comparison.json", comparison)
    save(metadata_path, metadata)
    return {"status": metadata["status"], **execution}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing the six named label/evidence JSON files")
    parser.add_argument("--root", type=Path, required=True, help="Mapping experiment output directory")
    parser.add_argument("--rule", nargs=2, action="append", metavar=("ARM", "PATH"))
    parser.add_argument("--papers", nargs="+", choices=PAPERS, default=list(PAPERS))
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.rule is None:
        args.rule = [("v5", str(PROMPT_PATH))]
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", arm) for arm, _ in args.rule) or len({arm for arm, _ in args.rule}) != len(args.rule):
        parser.error("Rule arm names must be unique and use letters, digits, underscores or hyphens")
    if len(args.papers) != len(set(args.papers)):
        parser.error("--papers must not contain duplicates")
    root = args.root.resolve()
    rules = {arm: Path(path).resolve() for arm, path in args.rule}
    prepare(root, rules, args.input_dir.expanduser().resolve(), args.papers)
    if not args.run:
        print("Frozen inputs and rules ready; no model called")
        return
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending = [pool.submit(run_one, root / arm, paper, root / arm / "inputs" / rule.name) for arm, rule in rules.items() for paper in args.papers]
            for future in as_completed(pending):
                future.result()
    finally:
        result = summarize(root)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
