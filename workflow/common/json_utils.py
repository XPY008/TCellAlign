from __future__ import annotations

import json
import re
from typing import Any


def strip_json_fence(content: str) -> str:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_json_object(content: str) -> tuple[dict[str, Any] | None, str | None]:
    text = strip_json_fence(content)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as first_error:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None, f"No JSON object found: {first_error}"
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as second_error:
            return None, str(second_error)
    if not isinstance(parsed, dict):
        return None, "Top-level JSON is not an object."
    return parsed, None


def parse_json_array(content: str) -> tuple[list[Any] | None, str | None]:
    text = strip_json_fence(content)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as first_error:
        match = re.search(r"\[.*\]", text, flags=re.DOTALL)
        if not match:
            return None, f"No JSON array found: {first_error}"
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as second_error:
            return None, str(second_error)
    if not isinstance(parsed, list):
        return None, "Top-level JSON is not a list."
    return parsed, None
