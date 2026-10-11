from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
import json
from numbers import Number
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

import fitz
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflow.common.llm_chat import apply_chat_options, create_openai_client
from workflow.common.llm_config import load_runtime_config, load_semantic_scholar_api_key
from workflow.common.paths import run_agent_dir
from workflow.common.prompts import read_prompt_section
from workflow.common.run_metadata import runtime_config_metadata, update_run_metadata
from workflow.common.agent_io import SEARCH_QUALIFIED_PAPERS, SearchPaper, write_search_qualified_papers

SEARCH_DIR = Path(__file__).resolve().parent
PROMPT_PATH = SEARCH_DIR / "prompt.md"
SEMANTIC_SCHOLAR_BASE_URL = "https://api.semanticscholar.org/graph/v1"
SEMANTIC_SCHOLAR_FIELDS = "paperId,url,externalIds,title,year,citationCount"
SEMANTIC_SCHOLAR_SEARCH_FIELDS = "paperId,title,openAccessPdf"
SEMANTIC_SCHOLAR_MIN_INTERVAL_SECONDS = 5.0
SEMANTIC_SCHOLAR_MAX_ATTEMPTS = 6
SEMANTIC_SCHOLAR_429_RETRY_SECONDS = 30.0
PAPER_TITLE_MAX_BYTES = 200
CELLXGENE_TITLE_SIMILARITY_THRESHOLD = 0.95
CELLXGENE_FETCH_MAX_ATTEMPTS = 4
CELLXGENE_FETCH_RETRY_SECONDS = 10.0
LLM_TIMEOUT_SECONDS = 3600
API_TIMEOUT_SECONDS = 30
_last_semantic_scholar_request_at: float | None = None


@dataclass(frozen=True)
class QualifiedPaper:
    source_pdf: Path
    markdown: str
    title: str
    step3_output: dict[str, Any]
    dataset_downloads: tuple[DatasetDownload, ...]
    verified_reference: str
    semantic_scholar_paper: dict[str, Any]


@dataclass(frozen=True)
class DatasetDownload:
    source: str
    identifier: str
    filename: str
    reference: str
    url: str | None = None


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def http_json(url: str, *, timeout_seconds: float, headers: dict[str, str] | None = None) -> Any:
    request = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.load(response)


def wait_for_semantic_scholar_rate_limit(
    *,
    monotonic: Any = time.monotonic,
    sleep: Any = time.sleep,
) -> None:
    global _last_semantic_scholar_request_at
    now = monotonic()
    if _last_semantic_scholar_request_at is not None:
        elapsed = now - _last_semantic_scholar_request_at
        if elapsed < SEMANTIC_SCHOLAR_MIN_INTERVAL_SECONDS:
            sleep(SEMANTIC_SCHOLAR_MIN_INTERVAL_SECONDS - elapsed)
            now = monotonic()
    _last_semantic_scholar_request_at = now


def clean_page_text(text: str) -> str:
    return text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n").strip()


def render_full_paper_markdown(pdf_path: Path) -> str:
    with fitz.open(pdf_path) as document:
        if document.page_count < 2:
            raise ValueError(f"PDF has fewer than two pages: {pdf_path}")
        page_texts = [clean_page_text(page.get_text("text")) for page in document]
    body = "\n\n".join(text for text in page_texts if text)
    return f"# {pdf_path.stem}\n\n{body}\n" if body else f"# {pdf_path.stem}\n"


def parse_json_content(content: str) -> dict[str, Any]:
    text = content.strip()
    fenced_match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced_match:
        text = fenced_match.group(1).strip()
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("LLM output is not a JSON object.")
    return parsed


def build_llm_prompt(prompt_template: str, markdown: str) -> str:
    return (
        prompt_template.rstrip()
        + "\n\nFull-paper Markdown:\n\n"
        + "<paper_markdown>\n"
        + markdown
        + "\n</paper_markdown>\n"
    )


def run_llm_json(
    *,
    client: OpenAI,
    model: str,
    prompt_template: str,
    markdown: str,
    temperature: float | None,
    thinking: str | None,
    provider_name: str | None,
    stream: bool,
) -> dict[str, Any]:
    request_kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return valid JSON only."},
            {"role": "user", "content": build_llm_prompt(prompt_template, markdown)},
        ],
    }
    apply_chat_options(
        request_kwargs,
        temperature=temperature,
        thinking=thinking,
        provider_name=provider_name,
        stream=stream,
    )

    response = client.chat.completions.create(**request_kwargs)
    if not stream:
        return parse_json_content(response.choices[0].message.content or "")

    content_parts: list[str] = []
    for chunk in response:
        if not chunk.choices:
            continue
        content = getattr(chunk.choices[0].delta, "content", None)
        if content:
            content_parts.append(content)
    return parse_json_content("".join(content_parts))


def iter_input_pdfs(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".pdf")


def paper_passes_step2(payload: dict[str, Any]) -> bool:
    value = payload.get("paper_passes_screening")
    if not isinstance(value, bool):
        raise ValueError("Step 2 output must contain boolean paper_passes_screening.")
    return value


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value.casefold()).strip()


def normalize_doi(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip()
    text = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^doi:\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def normalize_pmid(value: Any) -> str:
    if isinstance(value, int):
        return str(value)
    if not isinstance(value, str):
        return ""
    return re.sub(r"^pmid:\s*", "", value.strip(), flags=re.IGNORECASE)


def normalize_year(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if not isinstance(value, str):
        return None
    match = re.search(r"\b(19|20)\d{2}\b", value)
    return int(match.group(0)) if match else None


def title_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, normalize_text(left), normalize_text(right)).ratio()


def slugify_title(title: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", title.strip()).strip("_").lower()
    return slug or "paper"


def extract_uuids(text: str) -> list[str]:
    return re.findall(
        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
        text,
    )


def cellxgene_dataset_reference(dataset: dict[str, Any]) -> str:
    dataset_id = dataset.get("dataset_id")
    collection_id = dataset.get("collection_id")
    if collection_id and dataset_id:
        return f"CELLxGENE Census dataset {dataset_id} in https://cellxgene.cziscience.com/collections/{collection_id}"
    if collection_id:
        return f"https://cellxgene.cziscience.com/collections/{collection_id}"
    return f"CELLxGENE Census dataset {dataset_id}"


def dataset_filename(title: Any, identifier: str, suffix: str) -> str:
    label = title if isinstance(title, str) and title.strip() else identifier
    stem = re.sub(r"[^A-Za-z0-9]+", "_", label.strip()).strip("_")
    if not stem:
        stem = identifier
    return f"{stem}_{identifier}{suffix}"


def cellxgene_dataset_is_usable(dataset: dict[str, Any]) -> bool:
    cell_count = dataset.get("dataset_total_cell_count")
    return bool(dataset.get("dataset_id") and dataset.get("dataset_h5ad_path") and isinstance(cell_count, Number) and cell_count > 0)


def clue_is_scrnaseq(clue: dict[str, Any]) -> bool:
    assay_hint = clue.get("assay_hint")
    return isinstance(assay_hint, str) and normalize_text(assay_hint) == "scrna-seq"


def cellxgene_dataset_matches(clue: dict[str, Any], dataset: dict[str, Any]) -> bool:
    clue_text = " ".join(str(value) for value in clue.values() if value)
    clue_uuids = set(extract_uuids(clue_text))
    metadata_uuids = {
        value
        for value in (dataset.get("collection_id"), dataset.get("dataset_id"), dataset.get("dataset_version_id"))
        if isinstance(value, str)
    }
    if clue_uuids.intersection(metadata_uuids):
        return True

    clue_doi = normalize_doi(clue.get("doi"))
    collection_doi = normalize_doi(dataset.get("collection_doi"))
    if clue_doi and collection_doi and clue_doi == collection_doi:
        return True

    citation = normalize_text(dataset.get("citation"))
    if clue_doi and citation and clue_doi in citation:
        return True

    collection_name = normalize_text(dataset.get("collection_name"))
    clue_collection_name = normalize_text(clue.get("collection_name"))
    if clue_collection_name and collection_name and clue_collection_name in collection_name:
        return True

    clue_dataset_name = normalize_text(clue.get("dataset_name"))
    dataset_title = normalize_text(dataset.get("dataset_title"))
    if clue_dataset_name and dataset_title and clue_dataset_name in dataset_title:
        return True

    return False


def paper_has_scrnaseq_evidence(step3_output: dict[str, Any]) -> bool:
    evidence_items = step3_output.get("scrnaseq_evidence")
    if isinstance(evidence_items, list):
        for item in evidence_items:
            if isinstance(item, dict) and clue_is_scrnaseq(item):
                return True

    for field_name in ("geo_accessions", "cellxgene_clues"):
        items = step3_output.get(field_name)
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict) and clue_is_scrnaseq(item):
                return True
    return False


def cellxgene_dataset_matches_paper_identity(paper_identity: dict[str, Any], dataset: dict[str, Any]) -> bool:
    doi = normalize_doi(paper_identity.get("doi"))
    if doi and cellxgene_dataset_matches({"doi": doi}, dataset):
        return True

    title = paper_identity.get("title")
    if not isinstance(title, str) or not title.strip():
        return False

    for field_name in ("collection_name", "dataset_title"):
        value = dataset.get(field_name)
        if (
            isinstance(value, str)
            and title_similarity(title, value) >= CELLXGENE_TITLE_SIMILARITY_THRESHOLD
        ):
            return True
    return False


@lru_cache(maxsize=1)
def fetch_cellxgene_datasets() -> list[dict[str, Any]]:
    import cellxgene_census

    for attempt in range(1, CELLXGENE_FETCH_MAX_ATTEMPTS + 1):
        try:
            with cellxgene_census.open_soma(census_version="stable") as census:
                datasets = census["census_info"]["datasets"].read().concat().to_pandas()
            return datasets.to_dict("records")
        except Exception as exc:
            if attempt == CELLXGENE_FETCH_MAX_ATTEMPTS:
                raise RuntimeError("CELLxGENE Census datasets lookup failed after retrying.") from exc
            time.sleep(CELLXGENE_FETCH_RETRY_SECONDS)
    raise RuntimeError("CELLxGENE Census datasets lookup exhausted retry attempts.")


def verify_cellxgene_reference(step3_output: dict[str, Any], *, timeout_seconds: float) -> str | None:
    downloads = collect_cellxgene_dataset_downloads(step3_output)
    return downloads[0].reference if downloads else None


def collect_cellxgene_dataset_downloads(step3_output: dict[str, Any]) -> list[DatasetDownload]:
    clues = step3_output.get("cellxgene_clues")
    scrnaseq_clues = (
        [clue for clue in clues if isinstance(clue, dict) and clue_is_scrnaseq(clue)]
        if isinstance(clues, list)
        else []
    )

    identity = step3_output.get("paper_identity")
    should_check_identity = isinstance(identity, dict) and paper_has_scrnaseq_evidence(step3_output)
    if not should_check_identity and not scrnaseq_clues:
        return []

    datasets = fetch_cellxgene_datasets()
    downloads: list[DatasetDownload] = []
    seen_dataset_ids: set[str] = set()

    def add_match(dataset: dict[str, Any]) -> None:
        dataset_id = dataset.get("dataset_id")
        if not isinstance(dataset_id, str) or not dataset_id.strip() or dataset_id in seen_dataset_ids:
            return
        seen_dataset_ids.add(dataset_id)
        downloads.append(
            DatasetDownload(
                source="CELLxGENE",
                identifier=dataset_id,
                filename=dataset_filename(dataset.get("dataset_title"), dataset_id, ".h5ad"),
                reference=cellxgene_dataset_reference(dataset),
                url=dataset.get("dataset_h5ad_path") if isinstance(dataset.get("dataset_h5ad_path"), str) else None,
            )
        )

    if should_check_identity:
        for dataset in datasets:
            if (
                cellxgene_dataset_is_usable(dataset)
                and cellxgene_dataset_matches_paper_identity(identity, dataset)
            ):
                add_match(dataset)

    for clue in scrnaseq_clues:
        for dataset in datasets:
            if cellxgene_dataset_is_usable(dataset) and cellxgene_dataset_matches(clue, dataset):
                add_match(dataset)
    return downloads


def verify_expression_reference(step3_output: dict[str, Any], *, timeout_seconds: float) -> str | None:
    downloads = collect_dataset_downloads(step3_output, timeout_seconds=timeout_seconds)
    return downloads[0].reference if downloads else None


def collect_dataset_downloads(
    step3_output: dict[str, Any],
    *,
    timeout_seconds: float,
) -> list[DatasetDownload]:
    return collect_cellxgene_dataset_downloads(step3_output)


def semantic_scholar_headers(api_key: str | None) -> dict[str, str]:
    if api_key and api_key.strip():
        return {"x-api-key": api_key.strip()}
    return {}


def semantic_scholar_json(url: str, *, api_key: str | None, timeout_seconds: float) -> Any:
    for attempt in range(1, SEMANTIC_SCHOLAR_MAX_ATTEMPTS + 1):
        try:
            wait_for_semantic_scholar_rate_limit()
            return http_json(url, timeout_seconds=timeout_seconds, headers=semantic_scholar_headers(api_key))
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == SEMANTIC_SCHOLAR_MAX_ATTEMPTS:
                if exc.code == 429:
                    raise RuntimeError(
                        "Semantic Scholar API returned HTTP 429 after retrying. "
                        "Wait before rerunning Search Agent."
                    ) from exc
                raise
            time.sleep(SEMANTIC_SCHOLAR_429_RETRY_SECONDS)
        except urllib.error.URLError:
            if attempt == SEMANTIC_SCHOLAR_MAX_ATTEMPTS:
                raise
            time.sleep(SEMANTIC_SCHOLAR_429_RETRY_SECONDS)
    raise RuntimeError("Semantic Scholar lookup exhausted retry attempts.")


def search_semantic_scholar_papers(
    query: str,
    *,
    max_results: int,
    api_key: str | None,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    params = urllib.parse.urlencode(
        {
            "query": query,
            "fields": SEMANTIC_SCHOLAR_SEARCH_FIELDS,
            "limit": str(max_results),
        }
    )
    url = f"{SEMANTIC_SCHOLAR_BASE_URL}/paper/search?{params}"
    payload = semantic_scholar_json(
        url,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
    )
    results = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(results, list):
        raise ValueError("Semantic Scholar paper search did not return a data list.")
    return [paper for paper in results if isinstance(paper, dict)]


def sanitize_paper_title(title: str) -> str:
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", title)
    sanitized = sanitized.strip().rstrip(".")
    if len(sanitized.encode("utf-8")) > PAPER_TITLE_MAX_BYTES:
        sanitized = (
            sanitized.encode("utf-8")[:PAPER_TITLE_MAX_BYTES]
            .decode("utf-8", errors="ignore")
            .rstrip(" .")
        )
    return sanitized


def download_open_pdf(
    url: str,
    destination: Path,
    *,
    timeout_seconds: float,
) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(f"{destination.suffix}.part")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "tc-norm-search/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            signature = response.read(5)
            if signature != b"%PDF-":
                return False
            with temporary_path.open("wb") as handle:
                handle.write(signature)
                shutil.copyfileobj(response, handle)
        temporary_path.replace(destination)
        return True
    finally:
        temporary_path.unlink(missing_ok=True)


def download_cellxgene_h5ad(
    dataset_id: str,
    destination: Path,
) -> None:
    import cellxgene_census

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(f"{destination.suffix}.part")
    try:
        cellxgene_census.download_source_h5ad(
            dataset_id,
            to_path=str(temporary_path),
            progress_bar=False,
        )
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def download_dataset_asset(
    download: DatasetDownload,
    destination: Path,
    *,
    timeout_seconds: float,
) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        print(f"[skip dataset exists] {destination.name}", flush=True)
        return
    if download.source == "CELLxGENE":
        download_cellxgene_h5ad(download.identifier, destination)
        return
    raise ValueError(f"Unsupported Search Agent dataset source for h5ad download: {download.source}")


def acquire_semantic_scholar_pdfs(
    query: str,
    *,
    max_results: int,
    output_dir: Path,
    api_key: str | None,
    timeout_seconds: float,
) -> list[Path]:
    print(f"[search] Semantic Scholar: {query}", flush=True)
    papers = search_semantic_scholar_papers(
        query,
        max_results=max_results,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
    )
    downloaded: list[Path] = []
    open_pdf_count = 0
    for paper in papers:
        title = paper.get("title")
        open_access_pdf = paper.get("openAccessPdf")
        if (
            not isinstance(title, str)
            or not isinstance(open_access_pdf, dict)
        ):
            continue
        sanitized_title = sanitize_paper_title(title)
        if not sanitized_title:
            continue
        pdf_url = open_access_pdf.get("url")
        if not isinstance(pdf_url, str) or not pdf_url.strip():
            continue
        open_pdf_count += 1
        destination = output_dir / f"{sanitized_title}.pdf"
        if destination.exists():
            print(f"[skip duplicate title] {sanitized_title}", flush=True)
            continue
        try:
            if download_open_pdf(
                pdf_url.strip(),
                destination,
                timeout_seconds=timeout_seconds,
            ):
                downloaded.append(destination)
                print(f"[download] {destination.name}", flush=True)
            else:
                print(f"[skip non-pdf] {title}", flush=True)
        except (OSError, urllib.error.URLError) as exc:
            print(f"[skip download error] {title}: {exc}", flush=True)
    print(
        f"[search] results={len(papers)} open_pdf={open_pdf_count} downloaded={len(downloaded)}",
        flush=True,
    )
    return downloaded


def get_semantic_scholar_paper_by_id(
    external_id: str,
    *,
    api_key: str | None,
    timeout_seconds: float,
) -> dict[str, Any]:
    encoded_id = urllib.parse.quote(external_id, safe=":")
    query = urllib.parse.urlencode({"fields": SEMANTIC_SCHOLAR_FIELDS})
    url = f"{SEMANTIC_SCHOLAR_BASE_URL}/paper/{encoded_id}?{query}"
    payload = semantic_scholar_json(url, api_key=api_key, timeout_seconds=timeout_seconds)
    if not isinstance(payload, dict) or "citationCount" not in payload:
        raise ValueError(f"Semantic Scholar paper lookup did not return citationCount for {external_id}.")
    return payload


def select_semantic_scholar_title_match(
    candidates: list[dict[str, Any]],
    *,
    title: str,
    year: int | None,
) -> dict[str, Any]:
    if not candidates:
        raise ValueError(f"Semantic Scholar title search returned no candidates for {title!r}.")
    same_year = [
        candidate
        for candidate in candidates
        if year is not None and candidate.get("year") == year
    ]
    pool = same_year or candidates
    return max(pool, key=lambda item: title_similarity(title, item.get("title") or ""))


def search_semantic_scholar_by_title(
    title: str,
    *,
    year: int | None,
    api_key: str | None,
    timeout_seconds: float,
) -> dict[str, Any]:
    query = urllib.parse.urlencode(
        {"query": title, "fields": SEMANTIC_SCHOLAR_FIELDS, "limit": "10"}
    )
    url = f"{SEMANTIC_SCHOLAR_BASE_URL}/paper/search?{query}"
    payload = semantic_scholar_json(url, api_key=api_key, timeout_seconds=timeout_seconds)
    results = payload.get("data", []) if isinstance(payload, dict) else []
    return select_semantic_scholar_title_match(results, title=title, year=year)


def lookup_semantic_scholar_paper(
    paper_identity: dict[str, Any],
    *,
    api_key: str | None = None,
    timeout_seconds: float,
) -> dict[str, Any]:
    not_found_error: urllib.error.HTTPError | None = None
    doi = normalize_doi(paper_identity.get("doi"))
    if doi:
        try:
            return get_semantic_scholar_paper_by_id(
                f"DOI:{doi}",
                api_key=api_key,
                timeout_seconds=timeout_seconds,
            )
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
            not_found_error = exc

    normalized_pmid = normalize_pmid(paper_identity.get("pmid"))
    if normalized_pmid:
        try:
            return get_semantic_scholar_paper_by_id(
                f"PMID:{normalized_pmid}",
                api_key=api_key,
                timeout_seconds=timeout_seconds,
            )
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
            not_found_error = exc

    title = paper_identity.get("title")
    if not isinstance(title, str) or not title.strip():
        if not_found_error is not None:
            raise not_found_error
        raise ValueError("Step 3 paper_identity.title is required for title search and output naming.")
    return search_semantic_scholar_by_title(
        title.strip(),
        year=normalize_year(paper_identity.get("year")),
        api_key=api_key,
        timeout_seconds=timeout_seconds,
    )


def output_title(step3_output: dict[str, Any]) -> str:
    identity = step3_output.get("paper_identity")
    if not isinstance(identity, dict):
        raise ValueError("Step 3 output is missing paper_identity.")
    title = identity.get("title")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("Step 3 paper_identity.title is required.")
    return title.strip()


def write_ranked_outputs(
    output_dir: Path,
    papers: list[QualifiedPaper],
    *,
    dataset_timeout_seconds: float,
) -> None:
    qualified_paper_dir = output_dir / "qualified_papers"
    qualified_markdown_dir = output_dir / "qualified_markdown"
    qualified_paper_dir.mkdir(parents=True, exist_ok=True)
    qualified_markdown_dir.mkdir(parents=True, exist_ok=True)

    ranked = sorted(papers, key=lambda paper: paper.semantic_scholar_paper.get("citationCount") or 0, reverse=True)
    rows: list[dict[str, Any]] = []
    search_papers: list[SearchPaper] = []
    for rank, paper in enumerate(ranked, start=1):
        slug = f"{rank}_{slugify_title(paper.title)}"
        paper_asset_dir = qualified_paper_dir / slug
        dataset_dir = paper_asset_dir / "datasets"
        pdf_path = paper_asset_dir / "paper.pdf"
        markdown_path = qualified_markdown_dir / f"{slug}.md"
        paper_asset_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(paper.source_pdf, pdf_path)
        markdown_path.write_text(paper.markdown, encoding="utf-8")
        dataset_paths: list[Path] = []
        for download in paper.dataset_downloads:
            dataset_path = dataset_dir / download.filename
            print(f"[download dataset] {slug}: {download.filename}", flush=True)
            download_dataset_asset(
                download,
                dataset_path,
                timeout_seconds=dataset_timeout_seconds,
            )
            dataset_paths.append(dataset_path)
        search_papers.append(SearchPaper(paper_slug=slug, title=paper.title, markdown_path=str(markdown_path)))
        rows.append(
            {
                "rank": rank,
                "paper_slug": slug,
                "title": paper.title,
                "citation_count": paper.semantic_scholar_paper.get("citationCount"),
                "citation_source": "Semantic Scholar",
                "semantic_scholar_paper_id": paper.semantic_scholar_paper.get("paperId"),
                "semantic_scholar_url": paper.semantic_scholar_paper.get("url"),
                "semantic_scholar_title": paper.semantic_scholar_paper.get("title"),
                "semantic_scholar_publication_year": paper.semantic_scholar_paper.get("year"),
                "paper_asset_dir": str(paper_asset_dir),
                "pdf_path": str(pdf_path),
                "markdown_path": str(markdown_path),
                "dataset_paths": ";".join(str(path) for path in dataset_paths),
                "h5ad_reference": paper.verified_reference,
            }
        )

    fieldnames = [
        "rank",
        "paper_slug",
        "title",
        "citation_count",
        "citation_source",
        "semantic_scholar_paper_id",
        "semantic_scholar_url",
        "semantic_scholar_title",
        "semantic_scholar_publication_year",
        "paper_asset_dir",
        "pdf_path",
        "markdown_path",
        "dataset_paths",
        "h5ad_reference",
    ]
    with (output_dir / "ranked_title_list.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_search_qualified_papers(output_dir / SEARCH_QUALIFIED_PAPERS, search_papers)


def process_pdf(
    pdf_path: Path,
    *,
    client: OpenAI,
    model: str,
    step2_prompt: str,
    step3_prompt: str,
    temperature: float | None,
    thinking: str | None,
    provider_name: str | None,
    stream: bool,
    api_timeout_seconds: float,
    semantic_scholar_api_key: str | None,
) -> QualifiedPaper | None:
    print(f"[parse] {pdf_path.name}", flush=True)
    markdown = render_full_paper_markdown(pdf_path)

    print(f"[screen] {pdf_path.name}", flush=True)
    step2_output = run_llm_json(
        client=client,
        model=model,
        prompt_template=step2_prompt,
        markdown=markdown,
        temperature=temperature,
        thinking=thinking,
        provider_name=provider_name,
        stream=stream,
    )
    if not paper_passes_step2(step2_output):
        print(f"[reject step2] {pdf_path.name}", flush=True)
        return None

    print(f"[extract clues] {pdf_path.name}", flush=True)
    step3_output = run_llm_json(
        client=client,
        model=model,
        prompt_template=step3_prompt,
        markdown=markdown,
        temperature=temperature,
        thinking=thinking,
        provider_name=provider_name,
        stream=stream,
    )
    title = output_title(step3_output)

    print(f"[verify data] {title}", flush=True)
    dataset_downloads = tuple(collect_dataset_downloads(step3_output, timeout_seconds=api_timeout_seconds))
    if not dataset_downloads:
        print(f"[reject step3] {title}", flush=True)
        return None
    verified_reference = "; ".join(download.reference for download in dataset_downloads)

    print(f"[citation] {title}", flush=True)
    identity = step3_output.get("paper_identity")
    if not isinstance(identity, dict):
        raise ValueError("Step 3 output is missing paper_identity.")
    semantic_scholar_paper = lookup_semantic_scholar_paper(
        identity,
        api_key=semantic_scholar_api_key,
        timeout_seconds=api_timeout_seconds,
    )
    return QualifiedPaper(
        source_pdf=pdf_path,
        markdown=markdown,
        title=title,
        step3_output=step3_output,
        dataset_downloads=dataset_downloads,
        verified_reference=verified_reference,
        semantic_scholar_paper=semantic_scholar_paper,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Search Agent raw PDF screening and citation ranking."
    )
    parser.add_argument("--run-name", required=True)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-dir")
    input_group.add_argument("--query")
    parser.add_argument("--max-results", type=int, default=50)
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--thinking", choices=["enabled", "disabled", "none"], default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    output_dir = run_agent_dir(args.run_name, "search_agent")
    semantic_scholar_api_key = load_semantic_scholar_api_key()
    if args.input_dir is not None:
        input_mode = "local"
        input_dir = Path(args.input_dir).expanduser().resolve()
        if not input_dir.is_dir():
            raise SystemExit(f"Missing input directory: {input_dir}")
        pdfs = iter_input_pdfs(input_dir)
        if not pdfs:
            raise SystemExit(f"No PDF files found directly under: {input_dir}")
    else:
        input_mode = "semantic_scholar_query"
        input_dir = (REPO_ROOT / "data" / "papers" / args.run_name).resolve()
        pdfs = acquire_semantic_scholar_pdfs(
            args.query,
            max_results=args.max_results,
            output_dir=input_dir,
            api_key=semantic_scholar_api_key,
            timeout_seconds=API_TIMEOUT_SECONDS,
        )
        if not pdfs:
            raise SystemExit(
                f"No valid open-access PDFs downloaded from Semantic Scholar for query: {args.query}"
            )

    step2_prompt = read_prompt_section(PROMPT_PATH, "Screening")
    step3_prompt = read_prompt_section(PROMPT_PATH, "Data availability")
    config = load_runtime_config(model_id=args.model, thinking=args.thinking, stream=True)
    client = create_openai_client(config, timeout_seconds=LLM_TIMEOUT_SECONDS)

    qualified: list[QualifiedPaper] = []
    for pdf_path in pdfs:
        paper = process_pdf(
            pdf_path,
            client=client,
            model=config.model_name,
            step2_prompt=step2_prompt,
            step3_prompt=step3_prompt,
            temperature=args.temperature,
            thinking=config.thinking,
            provider_name=config.provider_name,
            stream=config.stream,
            api_timeout_seconds=API_TIMEOUT_SECONDS,
            semantic_scholar_api_key=semantic_scholar_api_key,
        )
        if paper is not None:
            qualified.append(paper)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_ranked_outputs(output_dir, qualified, dataset_timeout_seconds=API_TIMEOUT_SECONDS)
    update_run_metadata(
        args.run_name,
        "search_agent",
        {
            "input_mode": input_mode,
            "input_dir": str(input_dir),
            "query": args.query,
            "max_results": args.max_results if args.query is not None else None,
            "downloaded_pdf_count": len(pdfs) if args.query is not None else None,
            "output_dir": str(output_dir),
            "model": runtime_config_metadata(config),
            "temperature": args.temperature,
            "prompts": {
                "screening": str(PROMPT_PATH) + "#screening",
                "data_availability": str(PROMPT_PATH) + "#data-availability",
            },
            "llm_timeout_seconds": LLM_TIMEOUT_SECONDS,
            "api_timeout_seconds": API_TIMEOUT_SECONDS,
            "qualified_paper_count": len(qualified),
        },
    )
    print(f"[done] qualified papers: {len(qualified)}", flush=True)
    print(f"[done] output: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
