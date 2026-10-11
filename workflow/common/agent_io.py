from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

from workflow.mapping_agent.slot_schema import normalize_mapped_slots


SEARCH_QUALIFIED_PAPERS = "search_qualified_papers.json"


@dataclass(frozen=True)
class SearchPaper:
    paper_slug: str
    title: str
    markdown_path: str


@dataclass(frozen=True)
class ExtractedLabel:
    original_label: str
    evidence: list[str]


@dataclass(frozen=True)
class IEPaper:
    title: str
    items: list[ExtractedLabel]


@dataclass(frozen=True)
class MappedLabel:
    original_label: str
    evidence: list[str]
    mapped_label: str
    mapped_slots: dict[str, dict[str, str | None]]


@dataclass(frozen=True)
class MappingPaper:
    title: str
    rule_version: str
    items: list[MappedLabel]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _required_string(value: Any, *, path: Path, field_name: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ValueError(f"{path} is missing required field {field_name!r}.")


def _required_evidence(value: Any, *, path: Path, field_name: str) -> list[str]:
    if isinstance(value, list) and all(isinstance(item, str) and item.strip() for item in value):
        return [item.strip() for item in value]
    raise ValueError(f"{path} field {field_name!r} must be a non-empty string array.")


def _required_items(payload: dict[str, Any], *, path: Path) -> list[Any]:
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError(f"{path} field 'items' must be a JSON array.")
    return items


def write_search_qualified_papers(path: Path, papers: Sequence[SearchPaper]) -> None:
    _write_json(
        path,
        {
            "papers": [
                {
                    "paper_slug": paper.paper_slug,
                    "title": paper.title,
                    "markdown_path": paper.markdown_path,
                }
                for paper in papers
            ],
        },
    )


def read_search_qualified_papers(path: Path) -> list[SearchPaper]:
    payload = _read_json_object(path)
    raw_papers = payload.get("papers")
    if not isinstance(raw_papers, list):
        raise ValueError(f"{path} field 'papers' must be a JSON array.")

    papers: list[SearchPaper] = []
    for index, item in enumerate(raw_papers):
        if not isinstance(item, dict):
            raise ValueError(f"{path} papers[{index}] must be an object.")
        papers.append(
            SearchPaper(
                paper_slug=_required_string(
                    item.get("paper_slug"),
                    path=path,
                    field_name=f"papers[{index}].paper_slug",
                ),
                title=_required_string(item.get("title"), path=path, field_name=f"papers[{index}].title"),
                markdown_path=_required_string(
                    item.get("markdown_path"),
                    path=path,
                    field_name=f"papers[{index}].markdown_path",
                ),
            )
        )
    return papers


def write_ie_paper(path: Path, paper: IEPaper) -> None:
    _write_json(
        path,
        {
            "title": paper.title,
            "items": [
                {"original_label": item.original_label, "evidence": item.evidence}
                for item in paper.items
            ],
        },
    )


def read_ie_paper(path: Path) -> IEPaper:
    payload = _read_json_object(path)
    items: list[ExtractedLabel] = []
    for item_index, raw_item in enumerate(_required_items(payload, path=path)):
        if not isinstance(raw_item, dict):
            raise ValueError(f"{path} items[{item_index}] must be an object.")
        items.append(
            ExtractedLabel(
                original_label=_required_string(
                    raw_item.get("original_label"),
                    path=path,
                    field_name=f"items[{item_index}].original_label",
                ),
                evidence=_required_evidence(
                    raw_item.get("evidence"),
                    path=path,
                    field_name=f"items[{item_index}].evidence",
                ),
            )
        )
    return IEPaper(
        title=_required_string(payload.get("title"), path=path, field_name="title"),
        items=items,
    )


def write_mapping_paper(path: Path, paper: MappingPaper) -> None:
    _write_json(
        path,
        {
            "title": paper.title,
            "rule_version": paper.rule_version,
            "items": [
                {
                    "original_label": item.original_label,
                    "evidence": item.evidence,
                    "mapped_label": item.mapped_label,
                    "mapped_slots": item.mapped_slots,
                }
                for item in paper.items
            ],
        },
    )


def read_mapping_paper(path: Path) -> MappingPaper:
    payload = _read_json_object(path)
    items: list[MappedLabel] = []
    for item_index, raw_item in enumerate(_required_items(payload, path=path)):
        if not isinstance(raw_item, dict):
            raise ValueError(f"{path} items[{item_index}] must be an object.")
        mapped_slots, slot_problems = normalize_mapped_slots(raw_item.get("mapped_slots"))
        if slot_problems:
            raise ValueError(f"{path} items[{item_index}].mapped_slots is invalid: {slot_problems}")
        items.append(
            MappedLabel(
                original_label=_required_string(
                    raw_item.get("original_label"),
                    path=path,
                    field_name=f"items[{item_index}].original_label",
                ),
                evidence=_required_evidence(
                    raw_item.get("evidence"),
                    path=path,
                    field_name=f"items[{item_index}].evidence",
                ),
                mapped_label=_required_string(
                    raw_item.get("mapped_label"),
                    path=path,
                    field_name=f"items[{item_index}].mapped_label",
                ),
                mapped_slots=mapped_slots,
            )
        )
    return MappingPaper(
        title=_required_string(payload.get("title"), path=path, field_name="title"),
        rule_version=_required_string(payload.get("rule_version"), path=path, field_name="rule_version"),
        items=items,
    )
