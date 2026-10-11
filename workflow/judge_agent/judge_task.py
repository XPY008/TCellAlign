from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from workflow.common.agent_io import MappingPaper


DEFAULT_USER_TEMPLATE_PATH = Path(__file__).resolve().parent / "prompt.md"


def build_mapping_record_json(
    *,
    paper_title: str,
    original_label: str,
    evidence: Sequence[str],
    mapped_label: str,
) -> str:
    return json.dumps({"title": paper_title, "original_label": original_label,
                       "evidence": list(evidence), "mapped_label": mapped_label},
                      ensure_ascii=False, indent=2)


def mapping_records(paper: MappingPaper) -> list[dict]:
    return [{"case_id": index, **json.loads(build_mapping_record_json(paper_title=paper.title, original_label=item.original_label, evidence=item.evidence, mapped_label=item.mapped_label)), "mapped_slots": item.mapped_slots}
            for index, item in enumerate(paper.items)]


def validate_feedback(payload: list | None, count: int) -> list[dict]:
    if payload is None or len(payload) != count:
        raise ValueError("Judge must return one decision per input record")
    seen = set()
    for item in payload:
        if not isinstance(item, dict) or set(item) != {"case_id", "decision", "feedback"}:
            raise ValueError("Judge record fields must be case_id, decision and feedback")
        index = item["case_id"]
        if type(index) is not int or index not in range(count) or index in seen:
            raise ValueError("Judge case IDs must match the input exactly once")
        seen.add(index)
        if item["decision"] not in {"keep", "revise", "uncertain"} or not isinstance(item["feedback"], str):
            raise ValueError("Invalid Judge decision or feedback")
        if item["decision"] == "revise" and not item["feedback"].strip():
            raise ValueError("A revision requires actionable feedback")
    return sorted(payload, key=lambda item: item["case_id"])


def build_user_prompt(*, rule_text: str, records: Sequence[dict], prompt_path: Path = DEFAULT_USER_TEMPLATE_PATH) -> str:
    return (prompt_path.read_text(encoding="utf-8")
            .replace("{{RULE_TEXT}}", rule_text.strip())
            .replace("{{MAPPING_RECORDS_JSON}}", json.dumps(list(records), ensure_ascii=False, indent=2))
            .strip())
