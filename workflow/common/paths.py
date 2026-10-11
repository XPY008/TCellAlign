from __future__ import annotations

from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results"


def slugify(value: str, *, default: str = "run") -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("_")
    return slug or default


def run_dir(run_name: str, *, results_root: Path | str = RESULTS_ROOT) -> Path:
    return Path(results_root).expanduser().resolve() / slugify(run_name)


def run_agent_dir(run_name: str, agent: str, *, results_root: Path | str = RESULTS_ROOT) -> Path:
    return run_dir(run_name, results_root=results_root) / agent


def paper_agent_path(
    run_name: str,
    agent: str,
    paper_slug: str,
    *,
    results_root: Path | str = RESULTS_ROOT,
) -> Path:
    return run_agent_dir(run_name, agent, results_root=results_root) / f"{slugify(paper_slug, default='paper')}.json"
