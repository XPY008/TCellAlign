#!/usr/bin/env python3
"""Compute Merge Consistency (MC) from author-annotation expression centroids.

Each mapped label merging at least two dataset annotations receives the mean
pairwise cosine similarity of its annotation centroids. MC is the unweighted mean
of these per-merged-label scores. Labels making no merge are excluded; a method
making no merges has MC=NaN. Dataset subsetting and expression preprocessing are
shared across methods within each paper.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
np.random.seed(0)
METHOD_SUBDIRS = ["mapping_agent", "bioportal_6onto_results", "zooma_4onto_results", "PubMedBert_results", "SapBERT_results", "BM25_results", "Levenshtein_results"]   # <-- EDIT to choose method folders

@dataclass(frozen=True)
class DatasetSpec:
    paper_key: str
    paper_name: str
    h5ad_path: Path
    group_column: str


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-dir", required=True, help="Directory containing method JSON folders and correspondence")
    p.add_argument("--dataset-summary-csv", required=True,
                   help="CSV with dataset_path, column_name, and optional paper_name; relative paths use the CSV directory")
    p.add_argument("--method-subdirs", nargs="+", default=METHOD_SUBDIRS)
    p.add_argument("--correspondence-subdir", default="data_correspondence")
    p.add_argument("--output-dir", required=True, help="Destination for combined CSVs and per-method results")
    p.add_argument("--output-name", default="merge_consistency")
    p.add_argument("--embedding-key", default=None,
                   help="explicit adata.obsm key (case-insensitive); default auto-picks integrated/PCA")
    p.add_argument("--embedding-mode", choices=["auto", "obsm", "pca"], default="auto",
                   help="auto: integrated latent (harmony/scVI/pca) else compute PCA; never UMAP/tSNE")
    p.add_argument("--n-pcs", type=int, default=50)
    p.add_argument("--n-hvg", type=int, default=3000)
    p.add_argument("--cell-type-col", default="cell_type",
                   help="obs column with CellxGene Cell-Ontology harmonisation, added as a baseline")
    p.add_argument("--cell-type-name", default="cellxgene_cl")
    p.add_argument("--no-cell-type-baseline", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def paper_key(name):
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def parse_dataset_specs_csv(summary_csv):
    df = pd.read_csv(summary_csv)
    required = {"dataset_path", "column_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{summary_csv} missing required columns: {sorted(missing)}")

    specs = {}
    for row in df.itertuples(index=False):
        if pd.isna(row.dataset_path) or pd.isna(row.column_name) or not str(row.column_name).strip():
            raise ValueError(f"{summary_csv} contains an empty dataset_path or column_name")
        dataset_path = Path(str(row.dataset_path)).expanduser()
        if not dataset_path.is_absolute():
            dataset_path = summary_csv.parent / dataset_path
        dataset_path = dataset_path.resolve()
        if not dataset_path.is_file():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        name = getattr(row, "paper_name", None)
        paper_name = str(name).strip() if pd.notna(name) and str(name).strip() else dataset_path.stem
        key = paper_key(paper_name)
        if not key or key in specs:
            raise ValueError(f"Empty or duplicate normalized paper_name: {paper_name}")
        specs[key] = DatasetSpec(key, paper_name, dataset_path, str(row.column_name).strip())
    if not specs:
        raise ValueError(f"{summary_csv} contains no dataset rows")
    return specs


def _load_items(path):
    payload = json.load(path.open())
    return (payload.get("title") or path.stem), payload.get("items", [])

NO_MATCH_TOKENS = {"", "no_match", "nomatch", "no match", "none", "null", "na", "n/a", "nan", "-"}


def _is_no_match(value) -> bool:
    """True when the correspondence declined to match this original label."""
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return str(value).strip().lower() in NO_MATCH_TOKENS


def build_mapping_table(method_json, correspondence_dir):
    """Return (df[original_label, mapped_label, matched_data_label], info).

    Rows whose correspondence is NO_MATCH (null / "NO_MATCH" / match_status=="no_match",
    or simply absent from the correspondence file) are DROPPED: the original label has
    no counterpart in the h5ad, so there is no annotation, no cells and no CL term to
    score it against. They are counted in `info` and reported, never silently hidden.
    """
    _, items = _load_items(method_json)
    df = pd.DataFrame(items)
    if not {"original_label", "mapped_label"}.issubset(df.columns):
        raise ValueError(f"{method_json} needs original_label, mapped_label")
    if df["original_label"].duplicated().any():
        raise ValueError(f"Duplicate original_label in {method_json}")

    if "matched_data_label" not in df.columns:
        corr = correspondence_dir / method_json.name
        if not corr.exists():
            raise FileNotFoundError(f"{method_json.name} lacks matched_data_label and no {corr}")
        _, c_items = _load_items(corr)
        c = pd.DataFrame(c_items)
        if "original_label" not in c.columns or "matched_data_label" not in c.columns:
            raise ValueError(f"{corr} needs original_label, matched_data_label")
        if c["original_label"].duplicated().any():
            raise ValueError(f"Duplicate original_label in {corr}")
        keep = ["original_label", "matched_data_label"] + \
               (["match_status"] if "match_status" in c.columns else [])
        # LEFT join: an original missing from the correspondence is treated as NO_MATCH
        df = df[["original_label", "mapped_label"]].merge(c[keep], on="original_label",
                                                          how="left", validate="one_to_one")

    no_match = df["matched_data_label"].map(_is_no_match)
    if "match_status" in df.columns:
        no_match = no_match | df["match_status"].astype(str).str.strip().str.lower().eq("no_match")

    info = {"n_originals_total": int(len(df)),
            "n_no_match": int(no_match.sum()),
            "no_match_labels": df.loc[no_match, "original_label"].astype(str).tolist()}

    kept = df.loc[~no_match, ["original_label", "mapped_label", "matched_data_label"]].copy()
    kept["matched_data_label"] = kept["matched_data_label"].astype(str)
    kept["mapped_label"] = kept["mapped_label"].astype(str)
    return kept, info

def _looks_like_counts(X):
    d = X.data[:10000] if sparse.issparse(X) else np.asarray(X).ravel()[:10000]; d = d[d != 0]
    return bool(d.size) and np.allclose(d, np.round(d)) and d.max() > 30


INTEGRATED_PRIORITY = ["X_scVI", "X_scanvi", "X_scANVI", "X_harmony", "harmony", "X_pca", "pca", "X_pca_harmony"]


def _find_rep(adata, preferred):
    keys = {k.lower(): k for k in adata.obsm.keys()}
    if preferred:
        return keys.get(preferred.lower())
    for cand in INTEGRATED_PRIORITY:
        actual = keys.get(cand.lower())
        if actual and "umap" not in actual.lower() and "tsne" not in actual.lower():
            return actual
    return None


def get_embedding(adata, mode, key, n_pcs, n_hvg):
    if mode in ("auto", "obsm"):
        k = _find_rep(adata, key)
        if k is not None:
            return np.asarray(adata.obsm[k], float), f"obsm[{k}]"
        if mode == "obsm":
            raise ValueError(f"no usable (non-UMAP/tSNE) obsm rep in {list(adata.obsm)}; use auto or pca")
    X = adata.X
    if _looks_like_counts(X):
        adata = adata.copy(); sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata); X = adata.X
    Xd = X.toarray() if sparse.issparse(X) else np.asarray(X, float)
    v = Xd.var(0); hvg = np.argsort(v)[::-1][:min(n_hvg, Xd.shape[1])]
    Xh = Xd[:, hvg]; mu, sd = Xh.mean(0), Xh.std(0); sd[sd == 0] = 1
    Xh = np.clip((Xh - mu) / sd, -10, 10)
    d = min(n_pcs, Xh.shape[1] - 1, Xh.shape[0] - 1)
    return PCA(n_components=d, random_state=0).fit_transform(Xh), f"PCA({d}) on {len(hvg)} HVGs (computed)"


def merged_label_consistency(item_df, Z, obs_annot):
    """Return the mean MC over merged labels and the corresponding per-label audit."""
    obs_annot = np.asarray(obs_annot).astype(str)
    ann = sorted(set(obs_annot))
    if len(ann) < 2:
        return {}, []
    cen = {a: Z[obs_annot == a].mean(0) for a in ann}
    C = np.vstack([cen[a] for a in ann])
    similarities = 1.0 - cdist(C, C, metric="cosine")
    a_index = {a: i for i, a in enumerate(ann)}
    records = []
    for m, grp in item_df.groupby(item_df["mapped_label"].astype(str)):
        alist = sorted({str(x) for x in grp["matched_data_label"] if str(x) in a_index})
        if len(alist) < 2:
            records.append({"mapped_label": m, "n_annotations": len(alist),
                            "merged": False, "avg_similarity": np.nan,
                            "matched_data_labels": " | ".join(alist)})
            continue
        ii = [a_index[a] for a in alist]
        sub = similarities[np.ix_(ii, ii)]
        tri = sub[np.triu_indices(len(ii), 1)]
        records.append({"mapped_label": m, "n_annotations": len(alist), "merged": True,
                        "avg_similarity": float(np.mean(tri)),
                        "matched_data_labels": " | ".join(alist)})
    per_label = [r for r in records if r["merged"]]
    summary = {"n_merged_labels": len(per_label), "MC": np.nan}
    if per_label:
        avgs = np.array([r["avg_similarity"] for r in per_label], float)
        summary["MC"] = float(avgs.mean())
    return summary, records


def project(obs_annot, item_df):
    l2m = {d: sorted(set(v.astype(str))) for d, v in item_df.groupby("matched_data_label")["mapped_label"]}
    unamb = {d: ms[0] for d, ms in l2m.items() if len(ms) == 1}
    conflicts = [d for d, ms in l2m.items() if len(ms) > 1]
    return np.array([unamb.get(a) for a in obs_annot], dtype=object), conflicts


def evaluate_method(item_df, obs_annot, Z):
    mapped, conflicts = project(obs_annot, item_df)
    valid = np.array([m is not None for m in mapped])
    out = {"n_original_labels": len(item_df),
           "n_unique_mapped_labels": int(item_df["mapped_label"].astype(str).nunique()),
           "n_conflicts": len(conflicts), "n_cells_valid": int(valid.sum())}
    mc, records = merged_label_consistency(item_df, Z, obs_annot)
    out.update(mc)
    out["_merge_audit"] = records
    return out


def main():
    a = parse_args()
    base = Path(a.base_dir).expanduser().resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"Input directory not found: {base}")
    summary_csv = Path(a.dataset_summary_csv).expanduser().resolve()
    if not summary_csv.is_file():
        raise FileNotFoundError(f"Dataset summary CSV not found: {summary_csv}")
    specs = parse_dataset_specs_csv(summary_csv)
    output = Path(a.output_dir).expanduser().resolve()
    for name in [*a.method_subdirs, a.correspondence_subdir, a.output_name]:
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError(f"Expected a directory or file name, got: {name}")
    corr_dir = base / a.correspondence_subdir

    paper_methods = defaultdict(dict)
    for method in a.method_subdirs:
        mdir = base / method
        if not mdir.is_dir():
            print(f"[skip method] {mdir} missing"); continue
        for jp in sorted(mdir.glob("*.json")):
            if "summary" in jp.stem.lower() or "metrics" in jp.stem.lower():
                continue
            paper_methods[paper_key(jp.stem)][method] = jp

    if not paper_methods:
        raise ValueError(f"No method JSON files found in {base}")

    combined, per_method = [], defaultdict(list)
    merge_audits = defaultdict(list)
    for pk, mjsons in paper_methods.items():
        spec = specs.get(pk)
        if spec is None:
            print(f"[skip paper] no dataset spec for {pk}"); continue
        tables, union, nm_info = {}, set(), {}
        for method, jp in mjsons.items():
            try:
                t, info = build_mapping_table(jp, corr_dir)
            except Exception as exc:
                print(f"[skip] {spec.paper_name}/{method}: {type(exc).__name__}: {exc}"); continue
            tables[method] = t; nm_info[method] = info
            union |= set(t["matched_data_label"].astype(str))
        if not tables:
            continue
        try:
            adata = ad.read_h5ad(spec.h5ad_path)
            if spec.group_column not in adata.obs.columns:
                raise ValueError(f"column '{spec.group_column}' not in obs")
            annot = adata.obs[spec.group_column].astype(str)
            present = sorted(union & set(annot.unique())); missing = sorted(union - set(annot.unique()))
            mask = annot.isin(present).to_numpy()
            if mask.sum() < 10 or len(present) < 2:
                raise ValueError(f"subset too small: {int(mask.sum())} cells, {len(present)} annotations")
            adata = adata[mask].copy()
            if adata.shape[0] > 20000:
                idx = np.random.default_rng(a.seed).choice(adata.shape[0], 20000, replace=False)
                adata = adata[idx].copy()
            Z, emb_desc = get_embedding(adata, a.embedding_mode, a.embedding_key, a.n_pcs, a.n_hvg)
            obs_annot = adata.obs[spec.group_column].astype(str).to_numpy()
        except Exception as exc:
            print(f"[skip paper] {spec.paper_name}: {type(exc).__name__}: {exc}"); continue
        if missing:
            print(f"  [note] {spec.paper_name}: {len(missing)} mapped annotation(s) absent: "
                  f"{missing[:5]}{'...' if len(missing) > 5 else ''}")
        for method, t in tables.items():
            try:
                row = evaluate_method(t, obs_annot, Z)
            except Exception as exc:
                print(f"[skip] {spec.paper_name}/{method}: {type(exc).__name__}: {exc}"); continue
            audit = row.pop("_merge_audit", [])
            for r in audit:
                merge_audits[method].append({"paper_name": spec.paper_name, "method": method, **r})
            inf = nm_info.get(method, {"n_originals_total": len(t), "n_no_match": 0})
            row = {"method": method, "paper_name": spec.paper_name, "embedding": emb_desc,
                   "n_originals_total": inf["n_originals_total"], "n_no_match": inf["n_no_match"],
                   "n_annotations_in_subset": len(present), "n_cells_subset": int(mask.sum()),
                   **row}
            per_method[method].append(row); combined.append(row)
        # ---- CellxGene cell_type baseline: mapped_label = each annotation's CL term ----
        if not a.no_cell_type_baseline and a.cell_type_col in adata.obs.columns:
            try:
                grp = adata.obs[spec.group_column].astype(str).to_numpy()
                ct = adata.obs[a.cell_type_col].astype(str).to_numpy()
                dfx = pd.DataFrame({"ann": grp, "ct": ct})
                recs, nonuniq = [], []
                for a_val, sub in dfx.groupby("ann"):
                    vc = sub["ct"].value_counts()
                    if vc.size > 1:
                        nonuniq.append(a_val)
                    recs.append({"original_label": a_val, "mapped_label": str(vc.index[0]),
                                 "matched_data_label": a_val})
                if nonuniq:
                    print(f"  [note] {spec.paper_name}: {len(nonuniq)} annotation(s) not 1:1 to "
                          f"'{a.cell_type_col}'; used majority: {nonuniq[:5]}{'...' if len(nonuniq) > 5 else ''}")
                row = evaluate_method(pd.DataFrame(recs), obs_annot, Z)
                audit = row.pop("_merge_audit", [])
                for r in audit:
                    merge_audits[a.cell_type_name].append(
                        {"paper_name": spec.paper_name, "method": a.cell_type_name, **r})
                row = {"method": a.cell_type_name, "paper_name": spec.paper_name, "embedding": emb_desc,
                       "n_annotations_in_subset": len(present), "n_cells_subset": int(mask.sum()),
                       **row}
                per_method[a.cell_type_name].append(row); combined.append(row)
            except Exception as exc:  # noqa: BLE001
                print(f"[skip] {spec.paper_name}/{a.cell_type_name}: {type(exc).__name__}: {exc}")
        elif not a.no_cell_type_baseline:
            print(f"  [note] {spec.paper_name}: no '{a.cell_type_col}' column; cell_type baseline skipped")

        print(f"[done] {spec.paper_name}  (subset {int(mask.sum())} cells, {len(present)} annotations, {emb_desc})")

    if not any(per_method.get(method) for method in a.method_subdirs):
        raise ValueError("No method score rows produced; check dataset names, JSON correspondence, and H5AD columns")

    cols = ["method", "paper_name", "embedding", "n_originals_total", "n_no_match",
            "n_annotations_in_subset", "n_cells_subset", "n_original_labels",
            "n_unique_mapped_labels", "n_conflicts", "n_cells_valid", "n_merged_labels", "MC"]
    for method, rows in per_method.items():
        out_dir = output / method; out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows); df = df[[c for c in cols if c in df.columns]].sort_values("paper_name")
        df.to_csv(out_dir / f"{a.output_name}.per_paper.csv", index=False)
        if merge_audits.get(method):
            pd.DataFrame(merge_audits[method]).sort_values(
                ["paper_name", "avg_similarity"]).to_csv(
                out_dir / f"{a.output_name}.merge_audit.csv", index=False)
        print(f"[{method}] {len(rows)} papers -> {out_dir}")
    if combined:
        cdf = pd.DataFrame(combined); cdf = cdf[[c for c in cols if c in cdf.columns]]
        output.mkdir(parents=True, exist_ok=True)
        cdf.to_csv(output / "merge_consistency_per_paper.csv", index=False)
        print(f"[combined] {output / 'merge_consistency_per_paper.csv'}")

if __name__ == "__main__":
    main()
