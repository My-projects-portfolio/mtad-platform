#!/usr/bin/env python3
"""
baseline_analysis.py — first-pass analysis utilities for the MTAD platform.

Operates entirely on results/runs/<Model>_summary.csv files written by
run_baseline.py. No re-runs required.

What it produces:
  - Long-form DataFrame of all (model, dataset, seed) cells
  - mean ± std per (model, dataset) for any metric
  - Paired Wilcoxon test between any two models across datasets
  - Per-dataset bar chart (one panel per dataset, error bars = seed std)
  - F1-inflation scatter (Standard-F1 vs PA-F1, all cells, by model)
  - Pareto plot (accuracy vs cost on any cost axis)

CLI usage:
  python extensions/analysis/baseline_analysis.py \\
      --results_dir results \\
      --out_dir results/figures \\
      --metric AUC-PR

Library usage:
  from extensions.analysis.baseline_analysis import (
      load_results, mean_std_table, paired_wilcoxon,
      per_dataset_bar_chart, f1_inflation_scatter, pareto_plot,
  )
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DETECTION_METRICS = [
    "AUC-PR", "AUC-ROC", "VUS-PR", "VUS-ROC",
    "Standard-F1", "PA-F1", "Event-based-F1", "R-based-F1", "Affiliation-F",
]
COST_METRICS = [
    "wall_time_seconds", "fit_time_seconds", "score_time_seconds",
    "infer_per_window_ms", "peak_gpu_memory_bytes", "param_count",
]

# Consistent palette across all figures — matches the demo deck's Ocean Gradient
PALETTE = {
    "IForest":            "#065A82",  # deep blue
    "OCSVM":              "#3A86A8",
    "LOF":                "#5DA5C1",
    "AutoEncoder":        "#1C7293",  # teal (reconstruction)
    "OmniAnomaly":        "#3E9CBA",
    "USAD":               "#5DB4C9",
    "LSTMAD":             "#00A896",  # accent (forecasting)
    "AnomalyTransformer": "#B85042",  # warm (transformer)
    "TranAD":             "#D87662",
    "TimesNet":           "#6D2E46",  # recent SOTA
    "UserMethod":         "#FFB000",  # gold (proposed)
}


def _color(model):
    return PALETTE.get(model, "#94A3B8")


def _short_dataset_name(filename):
    """002_MSL_id_1_Sensor_tr_500_1st_900.csv -> 002_MSL"""
    stem = filename.split(".csv")[0]
    parts = stem.split("_id_")
    return parts[0] if parts else stem


# ---------------------------------------------------------------------------
# Loading and aggregation
# ---------------------------------------------------------------------------

def load_results(results_dir, models=None):
    """Load all <Model>_summary.csv files into one long-form DataFrame.

    Parameters
    ----------
    results_dir : str or Path
        Directory containing runs/ with the *_summary.csv files.
    models : list of str or None
        If given, only load these models. Otherwise all summary files.

    Returns
    -------
    pd.DataFrame
        One row per (model, file, seed). Columns: model, file, dataset, seed,
        all detection metrics, all cost metrics.
    """
    runs_dir = Path(results_dir) / "runs"
    csv_paths = sorted(runs_dir.glob("*_summary.csv"))
    if models is not None:
        wanted = {f"{m}_summary.csv" for m in models}
        csv_paths = [p for p in csv_paths if p.name in wanted]
    if not csv_paths:
        raise FileNotFoundError(f"No *_summary.csv files found in {runs_dir}")

    dfs = []
    for csv_path in csv_paths:
        model_name = csv_path.stem.replace("_summary", "")
        df = pd.read_csv(csv_path)
        df["model"] = model_name
        df["dataset"] = df["file"].map(_short_dataset_name)
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def mean_std_table(df, metric, decimals=3):
    """Return a (dataset × model) table with cells = 'mean ± std' strings."""
    grouped = df.groupby(["dataset", "model"])[metric].agg(["mean", "std", "count"])
    grouped["formatted"] = grouped.apply(
        lambda r: (
            f"{r['mean']:.{decimals}f} ± {r['std']:.{decimals}f}"
            if pd.notna(r["std"]) and r["count"] > 1
            else f"{r['mean']:.{decimals}f}"
        ),
        axis=1,
    )
    return grouped["formatted"].unstack(level="model")


def paired_wilcoxon(df, model_a, model_b, metric):
    """Paired Wilcoxon signed-rank between two models across datasets.

    Uses each model's mean across seeds per dataset, then pairs by dataset.
    Returns dict with model means, win counts, statistic, and p-value.
    """
    means = (
        df.groupby(["model", "dataset"])[metric].mean().unstack(level="model")
    )
    if model_a not in means.columns or model_b not in means.columns:
        return {"error": f"Missing model. Available: {sorted(means.columns)}"}

    paired = means[[model_a, model_b]].dropna()
    n = len(paired)
    if n < 3:
        return {"error": f"Only {n} datasets have both models — Wilcoxon needs ≥3"}

    a_vals = paired[model_a].values
    b_vals = paired[model_b].values

    if np.allclose(a_vals, b_vals):
        return {
            "n_datasets": n,
            "model_a_mean": float(a_vals.mean()),
            "model_b_mean": float(b_vals.mean()),
            "model_a_wins": 0,
            "model_b_wins": 0,
            "ties": n,
            "statistic": None,
            "p_value": 1.0,
            "note": "identical values across all datasets",
        }

    stat, p_value = wilcoxon(a_vals, b_vals)
    return {
        "n_datasets": n,
        "model_a_mean": float(a_vals.mean()),
        "model_b_mean": float(b_vals.mean()),
        "model_a_wins": int((a_vals > b_vals).sum()),
        "model_b_wins": int((b_vals > a_vals).sum()),
        "ties": int((a_vals == b_vals).sum()),
        "statistic": float(stat),
        "p_value": float(p_value),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _save_fig(fig, save_path):
    """Save a figure as both PNG (for the website) and PDF (for paper inclusion)."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    pdf_path = save_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"  saved {save_path}")
    print(f"        {pdf_path}")


def per_dataset_bar_chart(df, metric="AUC-PR", save_path=None, models=None):
    """One subplot per dataset; bars = models; error bars = seed std."""
    datasets = sorted(df["dataset"].unique())
    if models is None:
        models = sorted(df["model"].unique())

    stats = df.groupby(["dataset", "model"])[metric].agg(["mean", "std", "count"])

    n_cols = min(5, len(datasets))
    n_rows = (len(datasets) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.8 * n_cols, 2.7 * n_rows))
    axes = np.atleast_2d(axes).flatten()

    max_y = 0.0
    for ax, ds in zip(axes, datasets):
        means, stds, colors, labels = [], [], [], []
        for m in models:
            try:
                row = stats.loc[(ds, m)]
                means.append(row["mean"])
                stds.append(row["std"] if pd.notna(row["std"]) else 0.0)
                colors.append(_color(m))
                labels.append(m)
            except KeyError:
                continue

        if not means:
            ax.set_visible(False)
            continue

        x = np.arange(len(labels))
        ax.bar(
            x, means, yerr=stds, color=colors, capsize=3,
            edgecolor="white", linewidth=0.5,
        )
        ax.set_title(ds, fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        max_y = max(max_y, max(np.array(means) + np.array(stds)))

    for ax in axes:
        if ax.get_visible():
            ax.set_ylim(0, min(1.05, max_y * 1.15))

    for ax in axes[len(datasets):]:
        ax.set_visible(False)

    fig.suptitle(
        f"{metric} per dataset (error bars = ±1 SD across seeds)",
        fontsize=12, fontweight="bold", y=1.00,
    )
    fig.tight_layout()
    if save_path:
        _save_fig(fig, save_path)
    return fig


def f1_inflation_scatter(df, save_path=None, models=None):
    """Standard-F1 (x) vs PA-F1 (y), each point a (model, dataset, seed) cell.

    The y=x diagonal is the 'no inflation' baseline. Points well above the
    diagonal show how much point-adjustment inflates F1 over the honest
    threshold-swept number.
    """
    if models is None:
        models = sorted(df["model"].unique())

    fig, ax = plt.subplots(figsize=(7, 7))

    for m in models:
        sub = df[df["model"] == m]
        if sub.empty:
            continue
        ax.scatter(
            sub["Standard-F1"], sub["PA-F1"],
            color=_color(m), label=m, alpha=0.7, s=45,
            edgecolor="white", linewidth=0.5,
        )

    ax.plot([0, 1], [0, 1], color="#94A3B8", linestyle="--", linewidth=1, zorder=0)
    ax.text(
        0.55, 0.50, "y = x (no inflation)", color="#64748B",
        fontsize=9, rotation=45, ha="center", va="center",
    )

    ax.set_xlabel("Standard-F1 (oracle, threshold-swept)", fontsize=11)
    ax.set_ylabel("PA-F1 (point-adjusted, oracle)", fontsize=11)
    ax.set_title(
        "F1 inflation: PA-F1 vs Standard-F1 across all (model, dataset, seed) cells",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.legend(loc="lower right", fontsize=10, frameon=False)

    fig.tight_layout()
    if save_path:
        _save_fig(fig, save_path)
    return fig


def pareto_plot(df, x="param_count", y="AUC-PR", save_path=None,
                models=None, log_x=True, mean_only=True):
    """Accuracy (y) vs cost (x). Default cost axis: param_count.

    With mean_only=True (default), each point is a (model, dataset) pair
    averaged across seeds — gives the typical performance and cost.
    """
    if models is None:
        models = sorted(df["model"].unique())

    if mean_only:
        agg = df.groupby(["model", "dataset"])[[x, y]].mean().reset_index()
    else:
        agg = df.copy()

    fig, ax = plt.subplots(figsize=(9, 6))
    for m in models:
        sub = agg[agg["model"] == m]
        if sub.empty:
            continue
        ax.scatter(
            sub[x], sub[y], color=_color(m), label=m, alpha=0.8, s=75,
            edgecolor="white", linewidth=0.5,
        )

    if log_x:
        ax.set_xscale("symlog", linthresh=1)

    ax.set_xlabel(x, fontsize=11)
    ax.set_ylabel(y, fontsize=11)
    ax.set_title(
        f"Pareto: {y} vs {x} (mean across seeds per (model, dataset))",
        fontsize=12, fontweight="bold",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.legend(loc="best", fontsize=10, frameon=False)

    fig.tight_layout()
    if save_path:
        _save_fig(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# CLI driver
# ---------------------------------------------------------------------------

def _print_pairwise_wilcoxon(df, models, metric):
    """Pretty-print pairwise paired-Wilcoxon results across model pairs."""
    print(f"\nPaired Wilcoxon on {metric} (each model's per-dataset mean across seeds, paired by dataset):")
    print("-" * 90)
    print(f"  {'A vs B':<40} {'n':>3}  {'mean A':>7}  {'mean B':>7}  "
          f"{'wins A/B':>9}  {'p':>7}")
    print("-" * 90)
    for i, m_a in enumerate(models):
        for m_b in models[i + 1:]:
            r = paired_wilcoxon(df, m_a, m_b, metric)
            if "error" in r:
                print(f"  {m_a} vs {m_b}: {r['error']}")
                continue
            note = f" ({r['note']})" if "note" in r else ""
            print(
                f"  {m_a:<18} vs {m_b:<18} {r['n_datasets']:>3}  "
                f"{r['model_a_mean']:>7.3f}  {r['model_b_mean']:>7.3f}  "
                f"{r['model_a_wins']:>3} / {r['model_b_wins']:<3}  "
                f"{r['p_value']:>7.4f}{note}"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", default="results",
                        help="Directory containing runs/*_summary.csv")
    parser.add_argument("--out_dir", default="results/figures",
                        help="Where to save figures")
    parser.add_argument("--metric", default="AUC-PR",
                        help="Lead detection metric for tables and bar chart")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    df = load_results(args.results_dir)
    models = sorted(df["model"].unique())
    datasets = sorted(df["dataset"].unique())
    print(f"Loaded {len(df)} cells across {len(models)} models, {len(datasets)} datasets")
    print(f"Models:   {models}")
    print(f"Datasets: {datasets}")

    # Mean ± std table
    print(f"\n{args.metric} (mean ± std across seeds):")
    print("-" * 90)
    table = mean_std_table(df, args.metric)
    print(table.to_string())

    # Pairwise Wilcoxon
    _print_pairwise_wilcoxon(df, models, args.metric)

    # Figures
    print(f"\nGenerating figures to {out_dir}/")
    per_dataset_bar_chart(df, metric=args.metric,
                          save_path=out_dir / f"per_dataset_bars__{args.metric}.png")
    f1_inflation_scatter(df, save_path=out_dir / "f1_inflation_scatter.png")
    pareto_plot(df, x="param_count", y=args.metric,
                save_path=out_dir / f"pareto_params__{args.metric}.png")
    pareto_plot(df, x="infer_per_window_ms", y=args.metric,
                save_path=out_dir / f"pareto_infer_per_window__{args.metric}.png",
                log_x=True)

    print("\nDone.")


if __name__ == "__main__":
    main()