from __future__ import annotations

from typing import Any


SLOT_NAMES = (
    "lineage",
    "function",
    "migration",
    "migration_subscript",
    "differentiation",
    "differentiation_subscript",
    "antigen",
)

SELF_T_LINEAGES = {"gamma-delta T", "NKT", "iNKT", "MAIT", "DN T", "DP T"}


def normalize_mapped_slots(value: Any) -> tuple[dict[str, dict[str, str | None]], list[str]]:
    problems: list[str] = []
    normalized: dict[str, dict[str, str | None]] = {}

    if not isinstance(value, dict):
        return normalized, ["mapped_slots is missing or not an object."]

    extra_slots = sorted(set(value) - set(SLOT_NAMES))
    if extra_slots:
        problems.append(f"mapped_slots has unexpected slot(s): {extra_slots}.")

    for slot_name in SLOT_NAMES:
        raw_slot = value.get(slot_name)
        if not isinstance(raw_slot, dict):
            problems.append(f"mapped_slots.{slot_name} is missing or not an object.")
            continue

        extra_fields = sorted(set(raw_slot) - {"module", "semantic_value"})
        if extra_fields:
            problems.append(f"mapped_slots.{slot_name} has unexpected field(s): {extra_fields}.")

        module = raw_slot.get("module")
        semantic_value = raw_slot.get("semantic_value")
        if module is None and semantic_value is None:
            normalized[slot_name] = {"module": None, "semantic_value": None}
            continue

        if isinstance(module, str) and module.strip() and isinstance(semantic_value, str) and semantic_value.strip():
            normalized[slot_name] = {
                "module": module.strip(),
                "semantic_value": semantic_value.strip(),
            }
            continue

        problems.append(
            f"mapped_slots.{slot_name} must have both module and semantic_value as non-empty strings, "
            "or both as null."
        )

    return normalized, problems


def build_mapped_label(mapped_slots: Any) -> str:
    normalized, problems = normalize_mapped_slots(mapped_slots)
    if problems:
        raise ValueError(f"Invalid mapped_slots: {problems}")

    modules = {slot_name: normalized[slot_name]["module"] for slot_name in SLOT_NAMES}
    lineage = modules["lineage"]
    function = modules["function"]
    suffix = "".join(
        module or ""
        for module in (
            modules["migration"],
            modules["migration_subscript"],
            modules["differentiation"],
            modules["differentiation_subscript"],
            modules["antigen"],
        )
    )

    body = f"{function}{suffix}" if function else f"T{suffix}"
    if lineage is None:
        return body
    if lineage in SELF_T_LINEAGES and body == "T":
        return lineage
    return f"{lineage} {body}"
