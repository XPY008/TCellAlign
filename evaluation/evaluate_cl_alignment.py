#!/usr/bin/env python3
"""Evaluate annotation grouping against Cell Ontology using AMI and unweighted Merge F1.

Scores use matched author-annotation categories, not individual cells. Conflicting
annotations are excluded and counted. NO_MATCH records are excluded and reported.
CL is a grouping reference rather than independent biological ground truth.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score
# ============================================================================ #
METHOD_SUBDIRS = ["mapping_agent", "bioportal_6onto_results", "zooma_4onto_results", "PubMedBert_results", "SapBERT_results", "BM25_results", "Levenshtein_results"]   # <-- EDIT to choose method folders

# ============================================================================ #

@dataclass(frozen=True)
class DatasetSpec:
    paper_key: str
    paper_name: str
    h5ad_path: Path
    group_column: str


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-dir", required=True, help="Directory containing method JSON folders and correspondence")
    p.add_argument("--dataset-summary-csv", required=True,
                   help="CSV with dataset_path, column_name, and optional paper_name; relative paths use the CSV directory")
    p.add_argument("--method-subdirs", nargs="+", default=METHOD_SUBDIRS)
    p.add_argument("--correspondence-subdir", default="data_correspondence")
    p.add_argument("--output-dir", required=True, help="Destination for combined CSVs and per-method results")
    p.add_argument("--output-name", default="cl_alignment")
    p.add_argument("--cell-type-col", default="cell_type", help="Cell-Ontology column in .obs")
    p.add_argument("--include-cl-baseline", action="store_true",
                   help="also score a 'cellxgene_cl' method (trivially perfect; off by default)")
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
 

# --------------------------------------------------------------- metrics -----
def annotation_to_mapped(item_df):
    """A -> M, dropping annotations claimed by >1 mapped label (conflicts)."""
    a2m = defaultdict(set)
    for _, r in item_df.iterrows():
        a2m[str(r["matched_data_label"])].add(str(r["mapped_label"]))
    clean = {a: next(iter(ms)) for a, ms in a2m.items() if len(ms) == 1}
    conflicts = sorted(a for a, ms in a2m.items() if len(ms) > 1)
    return clean, conflicts


def pair_counting(anns, m_of, cl_of):
    """Classic clustering precision/recall/F1 over annotation pairs.
      TP: merged by M and same CL      FP: merged by M but different CL (wrong merge)
      FN: same CL but not merged by M  (M is finer than the ontology)"""
    tp = fp = fn = 0.0
    for a, b in combinations(anns, 2):
        same_m = m_of[a] == m_of[b]
        same_cl = cl_of[a] == cl_of[b]
        if same_m and same_cl:
            tp += 1.0
        elif same_m and not same_cl:
            fp += 1.0
        elif same_cl and not same_m:
            fn += 1.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    rec = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    f1 = (2 * prec * rec / (prec + rec)) if (np.isfinite(prec) and np.isfinite(rec)
                                             and (prec + rec) > 0) else np.nan
    return prec, rec, f1, tp, fp, fn


def evaluate_method_vs_cl(item_df, cl_of, n_cells_of):
    m_of, conflicts = annotation_to_mapped(item_df)
    anns = sorted(a for a in m_of if a in cl_of)
    out = {"n_annotations_scored": len(anns), "n_conflicts": len(conflicts),
           "n_mapped_labels": len({m_of[a] for a in anns}),
           "n_cl_terms": len({cl_of[a] for a in anns})}
    if len(anns) < 2:
        return out, []

    M = np.array([m_of[a] for a in anns]); CL = np.array([cl_of[a] for a in anns])
    p, r, f1, tp, fp, fn = pair_counting(anns, m_of, cl_of)
    out.update({"merge_precision": p, "merge_recall": r, "Merge_F1": f1,
                "n_pairs_merged": int(tp + fp), "n_pairs_wrong_merge": int(fp),
                "n_pairs_same_cl_not_merged": int(fn)})
    out["AMI"] = float(adjusted_mutual_info_score(CL, M))

    # ---- per-merged-label audit: is each merge ontology-consistent? ----
    audit = []
    by_m = defaultdict(list)
    for a in anns:
        by_m[m_of[a]].append(a)
    for m, alist in sorted(by_m.items()):
        cls = sorted({cl_of[a] for a in alist})
        audit.append({"mapped_label": m, "n_annotations": len(alist),
                      "merged": len(alist) > 1, "n_distinct_cl": len(cls),
                      "cl_consistent": len(cls) == 1,
                      "cl_terms": " | ".join(cls),
                      "matched_data_labels": " | ".join(sorted(alist)),
                      "n_cells": int(sum(n_cells_of.get(a, 0) for a in alist))})
    out["n_merged_labels"] = sum(x["merged"] for x in audit)
    return out, audit


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

    combined, per_method, audits = [], defaultdict(list), defaultdict(list)
    for pk, mjsons in paper_methods.items():
        spec = specs.get(pk)
        if spec is None:
            print(f"[skip paper] no dataset spec for {pk}"); continue
        tables, union = {}, set()
        infos = {}
        for method, jp in mjsons.items():
            try:
                t, info = build_mapping_table(jp, corr_dir)
            except Exception as exc:
                print(f"[skip] {spec.paper_name}/{method}: {type(exc).__name__}: {exc}"); continue
            tables[method] = t; infos[method] = info
            union |= set(t["matched_data_label"].astype(str))
        if infos:
            nm = {m: i["n_no_match"] for m, i in infos.items()}
            if any(v for v in nm.values()):
                ex = next(i["no_match_labels"] for i in infos.values() if i["no_match_labels"])
                print(f"  [note] {mjsons and list(mjsons)[0] and ''}"
                      f"NO_MATCH dropped per method: {nm}  e.g. {ex[:3]}")
            kept_sets = {m: frozenset(t["original_label"]) for m, t in tables.items()}
            if len(set(kept_sets.values())) > 1:
                print("  [WARN] methods disagree on which originals are NO_MATCH -> the scored "
                      "subset differs BETWEEN METHODS, so the comparison is not like-for-like. "
                      "Put matched_data_label in the shared correspondence dir instead.")
        if not tables:
            continue
        try:
            adata = ad.read_h5ad(spec.h5ad_path)
            if spec.group_column not in adata.obs.columns:
                raise ValueError(f"column '{spec.group_column}' not in obs")
            if a.cell_type_col not in adata.obs.columns:
                raise ValueError(f"no Cell-Ontology column '{a.cell_type_col}' in obs")
            annot = adata.obs[spec.group_column].astype(str)
            present = sorted(union & set(annot.unique()))
            if len(present) < 2:
                raise ValueError(f"only {len(present)} mapped annotation(s) present")
            sub = adata.obs.loc[annot.isin(present), [spec.group_column, a.cell_type_col]].astype(str)
            sub.columns = ["ann", "cl"]
            cl_of, n_cells_of, nonuniq = {}, {}, []
            for av, g in sub.groupby("ann"):
                vc = g["cl"].value_counts()
                if vc.size > 1:
                    nonuniq.append(av)
                cl_of[av] = str(vc.index[0]); n_cells_of[av] = int(len(g))
        except Exception as exc:
            print(f"[skip paper] {spec.paper_name}: {type(exc).__name__}: {exc}"); continue
        if nonuniq:
            print(f"  [note] {spec.paper_name}: {len(nonuniq)} annotation(s) not 1:1 to "
                  f"'{a.cell_type_col}'; used majority: {nonuniq[:5]}{'...' if len(nonuniq) > 5 else ''}")

        work = dict(tables)
        if a.include_cl_baseline:   # trivially perfect -- reference vs itself
            work["cellxgene_cl"] = pd.DataFrame(
                [{"original_label": x, "mapped_label": cl_of[x], "matched_data_label": x}
                 for x in cl_of])
        for method, t in work.items():
            try:
                row, audit = evaluate_method_vs_cl(t, cl_of, n_cells_of)
                if "AMI" not in row:
                    raise ValueError("fewer than two unambiguous annotations to score")
            except Exception as exc:
                print(f"[skip] {spec.paper_name}/{method}: {type(exc).__name__}: {exc}"); continue
            for r in audit:
                audits[method].append({"paper_name": spec.paper_name, "method": method, **r})
            info = infos.get(method, {"n_originals_total": len(t), "n_no_match": 0})
            row = {"method": method, "paper_name": spec.paper_name,
                   "n_originals_total": info["n_originals_total"], "n_no_match": info["n_no_match"],
                   "n_annotations_present": len(present), **row}
            per_method[method].append(row); combined.append(row)
        print(f"[done] {spec.paper_name}  ({len(present)} annotations, "
              f"{len(set(cl_of.values()))} CL terms)")

    if not any(per_method.get(method) for method in a.method_subdirs):
        raise ValueError("No method score rows produced; check dataset names, JSON correspondence, and H5AD columns")

    cols = ["method", "paper_name", "n_originals_total", "n_no_match",
            "n_annotations_present", "n_annotations_scored", "n_conflicts",
            "n_mapped_labels", "n_cl_terms", "n_merged_labels",
            "AMI", "Merge_F1", "merge_precision", "merge_recall",
            "n_pairs_merged", "n_pairs_wrong_merge", "n_pairs_same_cl_not_merged"]
    for method, rows in per_method.items():
        od = output / method; od.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows); df = df[[c for c in cols if c in df.columns]].sort_values("paper_name")
        df.to_csv(od / f"{a.output_name}.per_paper.csv", index=False)
        if audits.get(method):
            pd.DataFrame(audits[method]).sort_values(
                ["paper_name", "cl_consistent", "n_distinct_cl"],
                ascending=[True, True, False]).to_csv(od / f"{a.output_name}.merge_audit.csv", index=False)
        print(f"[{method}] {len(rows)} papers -> {od}")
    if combined:
        cdf = pd.DataFrame(combined); cdf = cdf[[c for c in cols if c in cdf.columns]]
        output.mkdir(parents=True, exist_ok=True)
        cdf.to_csv(output / "cl_alignment_per_paper.csv", index=False)
        print(f"[combined] {output / 'cl_alignment_per_paper.csv'}")


if __name__ == "__main__":
    main()