from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

from workflow.common.prompts import read_prompt_section


PROMPT_PATH = Path(__file__).resolve().parent / "prompt.md"


def read_mapping_rule(path: Path = PROMPT_PATH) -> str:
    return read_prompt_section(path, "Nomenclature rule")


@dataclass(frozen=True)
class MappingCase:
    case_id: int
    original_label: str
    evidence: list[str]


def prompt_cases(cases: Sequence[MappingCase]) -> list[dict[str, object]]:
    return [
        {
            "case_id": case.case_id,
            "original_label": case.original_label,
            "evidence": case.evidence,
        }
        for case in cases
    ]


def build_mapping_prompt(reference_text: str, cases: Sequence[MappingCase]) -> str:
    template = read_prompt_section(PROMPT_PATH, "Mapping task")
    return template.replace("{{RULE_TEXT}}", reference_text.rstrip()).replace(
        "{{CASES_JSON}}", json.dumps(prompt_cases(cases), ensure_ascii=False, indent=2)
    ) + "\n"
