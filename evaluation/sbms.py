from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

import numpy as np
from openai import OpenAI
from scipy.optimize import linear_sum_assignment

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflow.common.llm_config import ConfigError, EmbeddingConfig, load_embedding_config

DEFAULT_EMBEDDING_MODEL_ID = "qwen_embedding"
DEFAULT_QWEN_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
DEFAULT_EMBEDDING_BATCH_SIZE = 10
DEFAULT_TERM_FIELDS = ("original_label", "cell_subtype", "cell_type", "paper_label", "label", "name", "term")
EmbeddingConfigError = ConfigError


@dataclass(frozen=True)
class SBMSResult:
    score: float
    precision: float
    recall: float
    row_ind: np.ndarray
    col_ind: np.ndarray
    sim_matrix: np.ndarray


def sbms_from_similarity(sim_matrix: np.ndarray) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    sim_matrix: row indicates gold terms, column indicates predicted terms.
    Each entry is the similarity between the corresponding gold and predicted term.
    """
    cost_matrix = 1 - sim_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    sim_score_sum = sim_matrix[row_ind, col_ind].sum()
    precision = sim_score_sum / sim_matrix.shape[1]
    recall = sim_score_sum / sim_matrix.shape[0]
    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall, row_ind, col_ind


def default_output_dir(
    *,
    repo_root: Path = REPO_ROOT,
    run_id: str | None = None,
) -> Path:
    resolved_run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    return repo_root / "results" / "evaluation" / resolved_run_id


def resolve_embedding_config(
    *,
    config_path: str | Path | None = None,
    embedding_model_id: str = DEFAULT_EMBEDDING_MODEL_ID,
    embedding_model_name: str | None = None,
    environ: dict[str, str] | None = None,
) -> EmbeddingConfig:
    return load_embedding_config(
        environ=environ,
        config_path=config_path or (REPO_ROOT / "llm.json"),
        embedding_model_id=embedding_model_id,
        embedding_model_name=embedding_model_name,
    )


class QwenEmbeddingClient:
    """OpenAI-compatible embedding client for the project's Qwen embedding endpoint."""

    def __init__(self, config: EmbeddingConfig, client: Any | None = None) -> None:
        self.config = config
        self._client = client or OpenAI(api_key=config.api_key, base_url=config.base_url)

    def embed_terms(self, terms: Sequence[str], *, batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE) -> np.ndarray:
        embeddings: list[list[float]] = []
        for batch in _batched(list(terms), batch_size):
            response = self._client.embeddings.create(model=self.config.model_name, input=batch)
            ordered = sorted(response.data, key=lambda item: item.index)
            embeddings.extend([list(item.embedding) for item in ordered])
        return np.array(embeddings, dtype=float)


def _batched(values: Sequence[str], batch_size: int) -> Iterable[list[str]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    for start in range(0, len(values), batch_size):
        yield list(values[start : start + batch_size])


def cosine_similarity_matrix(gold_embs: np.ndarray, pred_embs: np.ndarray) -> np.ndarray:
    gold_norm = gold_embs / np.linalg.norm(gold_embs, axis=1, keepdims=True)
    pred_norm = pred_embs / np.linalg.norm(pred_embs, axis=1, keepdims=True)
    return gold_norm @ pred_norm.T


def compute_sbms(
    gold: Sequence[str],
    pred: Sequence[str],
    embedding_client: QwenEmbeddingClient,
    *,
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
) -> SBMSResult:
    if not gold:
        raise ValueError("gold terms must not be empty.")
    if not pred:
        raise ValueError("predicted terms must not be empty.")
    gold_embs = embedding_client.embed_terms(gold, batch_size=batch_size)
    pred_embs = embedding_client.embed_terms(pred, batch_size=batch_size)
    sim_matrix = cosine_similarity_matrix(gold_embs, pred_embs)
    score, precision, recall, row_ind, col_ind = sbms_from_similarity(sim_matrix)
    return SBMSResult(
        score=score,
        precision=precision,
        recall=recall,
        row_ind=row_ind,
        col_ind=col_ind,
        sim_matrix=sim_matrix,
    )


def load_terms(path: str | Path, *, field: str | None = None) -> list[str]:
    term_path = Path(path)
    suffix = term_path.suffix.lower()
    if suffix == ".json":
        return _load_terms_from_json(json.loads(term_path.read_text(encoding="utf-8")), field=field)
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        return _load_terms_from_delimited(term_path, delimiter=delimiter, field=field)
    return [line.strip() for line in term_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_terms_from_json(payload: Any, *, field: str | None) -> list[str]:
    if isinstance(payload, dict):
        if isinstance(payload.get("items"), list):
            return _terms_from_records(payload["items"], field=field)
        if isinstance(payload.get("final_items"), list):
            return _terms_from_records(payload["final_items"], field=field)
        if isinstance(payload.get("terms"), list):
            return _terms_from_records(payload["terms"], field=field)
        if field is not None and isinstance(payload.get(field), list):
            return _terms_from_records(payload[field], field=None)
    if isinstance(payload, list):
        return _terms_from_records(payload, field=field)
    raise ValueError("JSON input must be a list, or an object with items/final_items/terms.")


def _load_terms_from_delimited(path: Path, *, delimiter: str, field: str | None) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            return []
        resolved_field = _resolve_record_field(reader.fieldnames, field=field)
        return _compact_terms(row.get(resolved_field, "") for row in reader)


def _terms_from_records(records: Sequence[Any], *, field: str | None) -> list[str]:
    if all(isinstance(record, str) for record in records):
        return _compact_terms(records)
    if not all(isinstance(record, dict) for record in records):
        raise ValueError("Term records must be strings or objects.")
    field_names: list[str] = []
    for record in records:
        for key in record:
            if key not in field_names:
                field_names.append(key)
    resolved_field = _resolve_record_field(field_names, field=field)
    return _compact_terms(record.get(resolved_field, "") for record in records)


def _resolve_record_field(field_names: Sequence[str], *, field: str | None) -> str:
    if field is not None:
        if field not in field_names:
            raise ValueError(f"Requested term field {field!r} is not present. Available fields: {list(field_names)}")
        return field
    for candidate in DEFAULT_TERM_FIELDS:
        if candidate in field_names:
            return candidate
    if field_names:
        return field_names[0]
    raise ValueError("No fields are available for term extraction.")


def _compact_terms(values: Iterable[Any]) -> list[str]:
    terms: list[str] = []
    for value in values:
        if isinstance(value, str) and value.strip():
            terms.append(value.strip())
    return terms


def write_sbms_outputs(
    output_dir: str | Path,
    *,
    gold_terms: Sequence[str],
    pred_terms: Sequence[str],
    result: SBMSResult,
    embedding_config: EmbeddingConfig,
) -> dict[str, str]:
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = resolved_output_dir / "similarity_matrix.csv"
    assignments_path = resolved_output_dir / "assignments.csv"
    summary_path = resolved_output_dir / "summary.json"

    with matrix_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gold_term", *pred_terms])
        for gold_term, scores in zip(gold_terms, result.sim_matrix):
            writer.writerow([gold_term, *[f"{score:.10f}" for score in scores]])

    with assignments_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["gold_index", "pred_index", "gold_term", "pred_term", "similarity"],
        )
        writer.writeheader()
        for row_index, col_index in zip(result.row_ind, result.col_ind):
            writer.writerow(
                {
                    "gold_index": int(row_index),
                    "pred_index": int(col_index),
                    "gold_term": gold_terms[int(row_index)],
                    "pred_term": pred_terms[int(col_index)],
                    "similarity": f"{result.sim_matrix[int(row_index), int(col_index)]:.10f}",
                }
            )

    summary = {
        "SBMS_F1": float(result.score),
        "score_name": "SBMS-F1",
        "precision": float(result.precision),
        "recall": float(result.recall),
        "gold_count": len(gold_terms),
        "pred_count": len(pred_terms),
        "matched_count": int(len(result.row_ind)),
        "embedding_provider": embedding_config.provider_name,
        "embedding_base_url": embedding_config.base_url,
        "embedding_model": embedding_config.model_name,
        "metric": (
            "Soft Bipartite Matching F1 Score: Hungarian one-to-one assignment over cosine similarity, "
            "precision=sum(match_sim)/len(pred), recall=sum(match_sim)/len(gold), "
            "score=F1(precision, recall)."
        ),
        "artifact_paths": {
            "similarity_matrix": str(matrix_path),
            "assignments": str(assignments_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary["artifact_paths"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare two IE Agent term sets with Qwen embeddings and Hungarian matching."
    )
    parser.add_argument("--gold", required=True, help="Gold term file: JSON, CSV, TSV, or one-term-per-line text.")
    parser.add_argument("--pred", required=True, help="Predicted term file: JSON, CSV, TSV, or text.")
    parser.add_argument("--gold-field", help="Field/column to read from the gold file.")
    parser.add_argument("--pred-field", help="Field/column to read from the predicted file.")
    parser.add_argument("--llm-config", default=str(REPO_ROOT / "llm.json"), help="Path to llm.json.")
    parser.add_argument(
        "--embedding-model-id",
        default=DEFAULT_EMBEDDING_MODEL_ID,
        help="Configured embedding model id in llm.json.",
    )
    parser.add_argument(
        "--embedding-model-name",
        help=f"Qwen embedding model name. Defaults to {DEFAULT_QWEN_EMBEDDING_MODEL}.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_EMBEDDING_BATCH_SIZE,
        help="Embedding request batch size. Defaults to 10 for conservative endpoint compatibility.",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory. Defaults to results/evaluation/<timestamp>.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    gold_terms = load_terms(args.gold, field=args.gold_field)
    pred_terms = load_terms(args.pred, field=args.pred_field)
    if not gold_terms:
        raise ValueError(f"No gold terms were loaded from {args.gold}.")
    if not pred_terms:
        raise ValueError(f"No predicted terms were loaded from {args.pred}.")

    embedding_config = resolve_embedding_config(
        config_path=args.llm_config,
        embedding_model_id=args.embedding_model_id,
        embedding_model_name=args.embedding_model_name,
    )
    result = compute_sbms(
        gold_terms,
        pred_terms,
        QwenEmbeddingClient(embedding_config),
        batch_size=args.batch_size,
    )
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir()
    artifact_paths = write_sbms_outputs(
        output_dir,
        gold_terms=gold_terms,
        pred_terms=pred_terms,
        result=result,
        embedding_config=embedding_config,
    )
    print(
        json.dumps(
            {
                "SBMS_F1": result.score,
                "score_name": "SBMS-F1",
                "precision": result.precision,
                "recall": result.recall,
                "artifact_paths": artifact_paths,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
