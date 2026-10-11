from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from workflow.common.llm_config import RuntimeConfig
from workflow.common.paths import RESULTS_ROOT, run_dir


RUN_METADATA_FILE = "run_metadata.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def runtime_config_metadata(config: RuntimeConfig) -> dict[str, Any]:
    return {
        "model_id": config.model_id,
        "provider": config.provider_name,
        "model_name": config.model_name,
        "thinking": config.thinking,
        "stream": config.stream,
        "supports_thinking": config.supports_thinking,
    }


def update_run_metadata(
    run_name: str,
    agent: str,
    agent_metadata: dict[str, Any],
    *,
    results_root: Path | str = RESULTS_ROOT,
) -> Path:
    root = run_dir(run_name, results_root=results_root)
    metadata_path = root / RUN_METADATA_FILE
    if metadata_path.exists():
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{metadata_path} must contain a JSON object.")
    else:
        payload = {
            "run_name": run_name,
            "created_at": utc_now(),
            "agents": {},
        }

    agents = payload.setdefault("agents", {})
    if not isinstance(agents, dict):
        raise ValueError(f"{metadata_path} field 'agents' must be a JSON object.")
    agents[agent] = {
        **agent_metadata,
        "updated_at": utc_now(),
    }

    root.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata_path
