#!/usr/bin/env python3
"""Plot boxplots and write mean/sd summary tables comparing Mapping Agent against other methods."""

from __future__ import annotations

import argparse
from pathlib import Path

import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _dedupe(seq):
    """Preserve order, drop repeats (the original list repeated several metrics)."""
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x); out.append(x)
    return out


DISPLAY_NAMES = {
    "AMI": "AMI",
    "Merge_F1": "Merge F1",
    "MC": "Merge Consistency",
    "SBMS_F1": "SBMS-F1",
    "merge_precision": "Merge Precision",
    "merge_recall": "Merge Recall",
    "n_merged_labels": "Merged label count",
}

METRIC_DIRECTION = {
    "AMI": "higher", "Merge_F1": "higher", "MC": "higher", "SBMS_F1": "higher",
    "merge_precision": "higher", "merge_recall": "descriptive (low = finer than CL)",
    "n_merged_labels": "descriptive",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        required=True,
        help="Per-paper combined evaluation CSV",
    )
    parser.add_argument(
        "--category-summary-csv",
        help="Summary CSV used to map each paper_name to a journal/category",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for plots and summary CSVs")
    parser.add_argument("--metrics", nargs="+", required=True, choices=list(DISPLAY_NAMES), help="Paper metrics or merge counts to plot")
    parser.add_argument("--highlight-method", default="mapping_agent", help="Method to highlight in color")
    parser.add_argument("--decimals", type=int, default=3, help="Decimals in the formatted table")
    return parser.parse_args()


def build_method_order(df: pd.DataFrame, highlight_method: str) -> list[str]:
    methods = df["method"].astype(str).unique().tolist()
    others = sorted(method for method in methods if method != highlight_method)
    return ([highlight_method] if highlight_method in methods else []) + others


def build_palette(method_order: list[str], highlight_method: str) -> dict[str, str]:
    palette = {method: "#b8c4d6" for method in method_order}
    if highlight_method in palette:
        palette[highlight_method] = "#d04a35"
    return palette


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def add_category_column(df: pd.DataFrame, summary_csv: Path) -> pd.DataFrame:
    summary_df = pd.read_csv(summary_csv)
    required = {"paper_name", "journal"}
    missing = required - set(summary_df.columns)
    if missing:
        raise ValueError(f"{summary_csv} missing required columns: {sorted(missing)}")
    mapping_df = summary_df[["paper_name", "journal"]].drop_duplicates()
    if mapping_df["paper_name"].duplicated().any():
        dup = mapping_df.loc[mapping_df["paper_name"].duplicated(), "paper_name"].tolist()
        raise ValueError(f"Duplicate paper_name to journal mappings in {summary_csv}: {dup[:5]}")
    out = df.merge(mapping_df, on="paper_name", how="left")
    out["journal"] = out["journal"].fillna("Uncategorized")
    return out


# --------------------------------------------------------------- tables ------
def summarize_metrics(plot_df: pd.DataFrame, method_order: list[str]) -> pd.DataFrame:
    """Tidy per-category summary: one row per (category, method, metric).

    n_papers is the count of NON-NULL values, so a metric that is NaN for a method
    (e.g. MC when nothing was merged) reports the papers it actually
    covers rather than silently averaging over fewer points.
    """
    frames = []
    overall = plot_df.copy()
    overall["journal"] = "All Papers"
    for source in (overall, plot_df):
        g = (source.groupby(["journal", "method", "metric"])["value"]
             .agg(n_papers="count", mean="mean", sd="std", median="median",
                  q25=lambda s_: s_.quantile(0.25), q75=lambda s_: s_.quantile(0.75),
                  min="min", max="max")
             .reset_index())
        g["iqr"] = g["q75"] - g["q25"]
        frames.append(g)
    tidy = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["journal", "method", "metric"], keep="first")
    tidy["metric_display"] = tidy["metric"].map(lambda m: DISPLAY_NAMES.get(m, m))
    tidy["better"] = tidy["metric"].map(lambda m: METRIC_DIRECTION.get(m, ""))
    tidy["method"] = pd.Categorical(tidy["method"], categories=method_order, ordered=True)
    cat_order = ["All Papers"] + sorted(x for x in tidy["journal"].unique() if x != "All Papers")
    tidy["journal"] = pd.Categorical(tidy["journal"], categories=cat_order, ordered=True)
    return tidy.sort_values(["journal", "metric", "method"]).reset_index(drop=True)


def format_wide(tidy: pd.DataFrame, stat: str = "mean_sd", decimals: int = 3) -> pd.DataFrame:
    """Wide table: rows = category x method, cols = metrics.
    stat='mean_sd'     -> 'mean ± sd (n=N)'
    stat='median_iqr'  -> 'median [q25-q75] (n=N)'   (robust to the outlier papers)
    stat='both'        -> 'mean ± sd | med M (n=N)'
    """
    def _fmt(row):
        if pd.isna(row["mean"]) and pd.isna(row["median"]):
            return ""
        n = int(row["n_papers"])
        if stat == "median_iqr":
            med = f"{row['median']:.{decimals}f}"
            if pd.isna(row["q25"]) or pd.isna(row["q75"]):
                return f"{med} (n={n})"
            return f"{med} [{row['q25']:.{decimals}f}-{row['q75']:.{decimals}f}] (n={n})"
        m = f"{row['mean']:.{decimals}f}"
        sd = "—" if pd.isna(row["sd"]) else f"{row['sd']:.{decimals}f}"
        if stat == "both":
            return f"{m} ± {sd} | med {row['median']:.{decimals}f} (n={n})"
        return f"{m} ± {sd} (n={n})"
    t = tidy.copy()
    t["cell"] = t.apply(_fmt, axis=1)
    wide = t.pivot_table(index=["journal", "method"], columns="metric_display",
                         values="cell", aggfunc="first", observed=True)
    order = [DISPLAY_NAMES.get(m, m) for m in tidy["metric"].drop_duplicates()]
    wide = wide[[c for c in order if c in wide.columns]]
    return wide.reset_index()


def plot_metric_grid(
    plot_df: pd.DataFrame,
    available_metrics: list[str],
    method_order: list[str],
    palette: dict[str, str],
    output_path: Path,
    title: str,
) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    n_metrics = len(available_metrics)
    ncols = 3
    nrows = (n_metrics + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 4.6 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for index, metric in enumerate(available_metrics):
        print(metric)
        ax = axes_flat[index]
        subset = plot_df.loc[plot_df["metric"] == metric].copy()
        sns.boxplot(data=subset, x="method", hue="method", y="value", order=method_order,
                    palette=palette, dodge=False, width=0.65, fliersize=0, ax=ax)
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        sns.stripplot(data=subset, x="method", y="value", order=method_order, color="#2f2f2f",
                      alpha=0.65, size=4, jitter=0.18, ax=ax)

        mean_series = subset.groupby("method")["value"].mean().reindex(method_order)
        x_positions, y_values = [], []
        for x_pos, method in enumerate(method_order):
            mean_value = mean_series.get(method)
            if pd.notna(mean_value):
                x_positions.append(x_pos); y_values.append(float(mean_value))
        if x_positions:
            ax.scatter(x_positions, y_values, marker="D", s=48, color="#111111",
                       edgecolor="#ffffff", linewidth=0.8, zorder=5)
            y_min, y_max = ax.get_ylim()
            y_span = max(y_max - y_min, 1e-9)
            text_offset = 0.015 * y_span
            for x_pos, mean_value in zip(x_positions, y_values):
                ax.text(x_pos, mean_value + text_offset, f"{mean_value:.3f}", ha="center",
                        va="bottom", fontsize=9, color="#111111")

        ax.set_title(DISPLAY_NAMES.get(metric, metric))
        ax.set_xlabel(""); ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=35)

    for index in range(n_metrics, len(axes_flat)):
        axes_flat[index].axis("off")

    fig.suptitle(title, y=0.995, fontsize=18)
    fig.supxlabel("Method"); fig.supylabel("Score")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "method_comparison_boxplots.png"

    df = pd.read_csv(input_csv)
    if args.category_summary_csv:
        summary_csv = Path(args.category_summary_csv).expanduser().resolve()
        df = add_category_column(df, summary_csv)
    else:
        df["journal"] = "All Papers"

    metrics = _dedupe(args.metrics)
    available_metrics = [m for m in metrics if m in df.columns]
    if not available_metrics:
        raise ValueError(f"None of the requested metrics were found in {input_csv}")

    plot_df = df[["method", "paper_name", "journal", *available_metrics]].melt(
        id_vars=["method", "paper_name", "journal"],
        value_vars=available_metrics, var_name="metric", value_name="value",
    )
    plot_df = plot_df.dropna(subset=["value"]).copy()
    if plot_df.empty:
        raise ValueError("No non-null values available for the requested metrics")

    method_order = build_method_order(plot_df, args.highlight_method)
    palette = build_palette(method_order, args.highlight_method)
    plot_df["metric_display"] = plot_df["metric"].map(lambda value: DISPLAY_NAMES.get(value, value))

    plot_metric_grid(plot_df, available_metrics, method_order, palette, output_path,
                     "Mapping Agent vs Other Methods Across Evaluation Metrics")
    print(output_path)

    # ---------------- summary tables (mean / sd per category, method, metric) ------------
    tidy = summarize_metrics(plot_df, method_order)
    tidy_path = output_dir / "metric_summary_by_category.csv"
    tidy.to_csv(tidy_path, index=False)
    print(tidy_path)

    for stat, suffix in (("mean_sd", "wide_mean_sd"), ("median_iqr", "wide_median_iqr"),
                         ("both", "wide")):
        wp = output_dir / f"metric_summary_by_category_{suffix}.csv"
        format_wide(tidy, stat=stat, decimals=args.decimals).to_csv(wp, index=False)
        print(wp)

    # per-category files, named to match the per-category plots
    for journal in [j for j in tidy["journal"].cat.categories if j != "All Papers"]:
        sub = tidy.loc[tidy["journal"] == journal]
        if sub.empty:
            continue
        sub_path = output_dir / f"metric_summary_{slugify(journal)}.csv"
        format_wide(sub, stat="both", decimals=args.decimals).to_csv(sub_path, index=False)
        print(sub_path)

    if "journal" in plot_df.columns and plot_df["journal"].notna().any():
        for journal in sorted(plot_df["journal"].dropna().unique()):
            category_df = plot_df.loc[plot_df["journal"] == journal].copy()
            if category_df.empty:
                continue
            category_output = output_dir / f"method_comparison_boxplots_{slugify(journal)}.png"
            plot_metric_grid(category_df, available_metrics, method_order, palette, category_output,
                             f"{journal}: Mapping Agent vs Other Methods Across Evaluation Metrics")
            print(category_output)


if __name__ == "__main__":
    main()
