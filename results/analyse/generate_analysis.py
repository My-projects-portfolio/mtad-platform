#!/usr/bin/env python3
"""
generate_analysis.py — Aggregate MTAD platform results into tables and figures.
(PATCHED VERSION — robust metric extraction across multiple sidecar schemas.)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
except ImportError:
    print("ERROR: matplotlib is required.  pip install matplotlib", file=sys.stderr)
    sys.exit(1)

METRIC_KEYS = [
    "AUC-PR", "AUC-ROC", "VUS-PR", "VUS-ROC",
    "Standard-F1", "PA-F1", "Event-based-F1", "R-based-F1", "Affiliation-F",
]

# Alternate spellings — tried in order
METRIC_ALIASES = {
    "AUC-PR":         ["AUC-PR", "AUC_PR", "aucpr", "auc_pr", "auc-pr", "AUCPR", "PR-AUC", "AP", "ap"],
    "AUC-ROC":        ["AUC-ROC", "AUC_ROC", "aucroc", "auc_roc", "auc-roc", "AUCROC", "ROC-AUC", "AUROC", "auroc"],
    "VUS-PR":         ["VUS-PR", "VUS_PR", "vuspr", "vus_pr", "vus-pr", "VUSPR"],
    "VUS-ROC":        ["VUS-ROC", "VUS_ROC", "vusroc", "vus_roc", "vus-roc", "VUSROC"],
    "Standard-F1":    ["Standard-F1", "Standard_F1", "standard_f1", "F1", "f1", "Std-F1", "Std_F1"],
    "PA-F1":          ["PA-F1", "PA_F1", "pa_f1", "PA F1", "Point-Adjusted-F1", "PointAdjustedF1"],
    "Event-based-F1": ["Event-based-F1", "Event_based_F1", "event_f1", "EventF1", "event-f1", "Event-F1"],
    "R-based-F1":     ["R-based-F1", "R_based_F1", "Rbased-F1", "RBased-F1", "Range-F1", "Range_F1", "R-F1"],
    "Affiliation-F":  ["Affiliation-F", "Affiliation_F", "affiliation_f", "AffiliationF", "Aff-F"],
}

MODEL_CATEGORIES = {
    "IForest": "Classical", "LOF": "Classical", "KNN": "Classical",
    "HBOS": "Classical", "COPOD": "Classical", "CBLOF": "Classical",
    "EIF": "Classical", "KMeansAD": "Classical", "PCA": "Classical",
    "AutoEncoder": "Reconstruction", "USAD": "Reconstruction", "OmniAnomaly": "Reconstruction",
    "LSTMAD": "Forecasting", "CNN": "Forecasting", "FITS": "Forecasting",
    "TranAD": "Transformer", "AnomalyTransformer": "Transformer",
    "PatchTST": "Transformer", "TimesNet": "Transformer",
    "MTGFLOW": "Density",
    "OFA": "Foundation",
}

CATEGORY_COLORS = {
    "Classical": "#0891B2", "Reconstruction": "#7C3AED", "Forecasting": "#059669",
    "Transformer": "#DC2626", "Density": "#D97706", "Foundation": "#6B21A8",
}


def dataset_short_name(filename: str) -> str:
    """Map a TSB-AD filename to a short dataset family name (e.g. 'SMD')."""
    stem = Path(str(filename)).stem
    stem = stem.split("__")[0]   # strip any __seedN suffix
    parts = stem.split("_")
    return parts[1] if len(parts) > 1 else stem


def _find_metric(source: dict, canonical_key: str):
    """Locate one metric in a dict, trying top-level then one nested level."""
    aliases = METRIC_ALIASES.get(canonical_key, [canonical_key])
    # Top-level
    for k in aliases:
        if k in source:
            try:
                return float(source[k])
            except (ValueError, TypeError):
                pass
    # One level of nesting
    for cont in ["metrics", "evaluation", "eval", "scores",
                 "evaluation_result", "results"]:
        sub = source.get(cont)
        if isinstance(sub, dict):
            for k in aliases:
                if k in sub:
                    try:
                        return float(sub[k])
                    except (ValueError, TypeError):
                        pass
    return None


def _extract_record(sidecar: dict, model_name: str, hint: str) -> dict:
    rec = {
        "model": model_name,
        "file": sidecar.get("file") or sidecar.get("filename") or hint,
        "seed": sidecar.get("seed"),
        "status": sidecar.get("status", "OK"),
        "wall_time": (sidecar.get("wall_time_seconds") or
                      sidecar.get("wall_time") or sidecar.get("wall") or
                      sidecar.get("time") or sidecar.get("runtime")),
        "error": sidecar.get("error"),
    }
    for mk in METRIC_KEYS:
        rec[mk] = _find_metric(sidecar, mk)
    return rec


def collect_all_results(results_dir: Path) -> pd.DataFrame:
    runs_dir = results_dir / "runs"
    if not runs_dir.exists():
        print(f"ERROR: {runs_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    rows: list[dict] = []
    diag_done = False
    fallback_models: list[str] = []

    for entry in sorted(runs_dir.iterdir()):
        if not entry.is_dir():
            continue
        model_name = entry.name
        sidecars = sorted(entry.glob("*.json"))
        model_rows: list[dict] = []

        for jp in sidecars:
            try:
                with open(jp) as f:
                    s = json.load(f)
            except Exception as e:
                print(f"  warn: could not read {jp}: {e}", file=sys.stderr)
                continue

            if not diag_done:
                print(f"\n[DIAGNOSTIC] first sidecar inspected: {jp.name}")
                print(f"[DIAGNOSTIC] top-level keys = {sorted(s.keys())}")
                for cont in ["metrics", "evaluation", "eval", "scores",
                             "evaluation_result", "results"]:
                    if isinstance(s.get(cont), dict):
                        print(f"[DIAGNOSTIC] '{cont}' sub-keys = {sorted(s[cont].keys())}")
                diag_done = True
                print()

            model_rows.append(_extract_record(s, model_name, jp.stem))

        # Did we get any metric values from sidecars?
        has_metrics = any(
            any(r.get(k) is not None for k in METRIC_KEYS) for r in model_rows
        )

        if has_metrics:
            rows.extend(model_rows)
            continue

        # Fallback to summary CSV
        if model_rows:
            fallback_models.append(model_name)

        summary_path = runs_dir / f"{model_name}_summary.csv"
        if summary_path.exists():
            try:
                df_s = pd.read_csv(summary_path)
                df_s.columns = [c.strip() for c in df_s.columns]
                df_s["model"] = model_name

                for cand in ["file", "filename", "File", "Filename",
                             "Dataset", "dataset"]:
                    if cand in df_s.columns and "file" not in df_s.columns:
                        df_s = df_s.rename(columns={cand: "file"})
                        break

                rename = {}
                for canon, aliases in METRIC_ALIASES.items():
                    for a in aliases:
                        if a in df_s.columns:
                            rename[a] = canon
                            break
                df_s = df_s.rename(columns=rename)

                for k in METRIC_KEYS:
                    if k not in df_s.columns:
                        df_s[k] = np.nan
                if "status" not in df_s.columns:
                    df_s["status"] = "OK"
                if "seed" not in df_s.columns:
                    df_s["seed"] = None
                if "wall_time" not in df_s.columns:
                    for c in ["wall", "time", "Time", "runtime"]:
                        if c in df_s.columns:
                            df_s = df_s.rename(columns={c: "wall_time"})
                            break
                if "wall_time" not in df_s.columns:
                    df_s["wall_time"] = None

                keep = ["model", "file", "seed", "status", "wall_time"] + METRIC_KEYS
                keep = [c for c in keep if c in df_s.columns]
                rows.extend(df_s[keep].to_dict(orient="records"))
            except Exception as e:
                print(f"  warn: could not read {summary_path}: {e}", file=sys.stderr)
        elif model_rows:
            # No summary CSV; keep the sidecar records (will be all-NaN for metrics)
            rows.extend(model_rows)

    if not rows:
        print("ERROR: no results found", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows)
    df["dataset"] = df["file"].astype(str).apply(dataset_short_name)
    for k in METRIC_KEYS:
        if k in df.columns:
            df[k] = pd.to_numeric(df[k], errors="coerce")

    n_with_metric = df["AUC-PR"].notna().sum()
    print(f"Loaded {len(df)} cells across {df['model'].nunique()} models "
          f"and {df['dataset'].nunique()} datasets.")
    print(f"Cells with usable AUC-PR: {n_with_metric}")
    if fallback_models:
        print(f"Fell back to summary CSV for: {', '.join(sorted(set(fallback_models)))}")
    return df


_SUCCESS_STATUSES = {"success", "ok", "completed", "done"}


def _is_success(status) -> bool:
    """Treat anything not explicitly a success token as an error."""
    if status is None:
        return True  # legacy sidecars with no status field — assume success
    return str(status).strip().lower() in _SUCCESS_STATUSES


def per_model_summary(df: pd.DataFrame) -> pd.DataFrame:
    df_ok = df[df["status"].map(_is_success)] if "status" in df.columns else df
    agg = df_ok.groupby("model")[METRIC_KEYS].agg(["mean", "std", "count"]).round(4)
    agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
    agg.insert(0, "category", agg.index.map(MODEL_CATEGORIES))
    if "AUC-PR_mean" in agg.columns and agg["AUC-PR_mean"].notna().any():
        agg = agg.sort_values("AUC-PR_mean", ascending=False, na_position="last")
    return agg


def per_model_per_dataset(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    df_ok = df[df["status"].map(_is_success)] if "status" in df.columns else df
    return df_ok.pivot_table(index="model", columns="dataset",
                              values=metric, aggfunc="mean")


def _order_models(models):
    cat_order = ["Classical", "Reconstruction", "Forecasting",
                  "Transformer", "Density", "Foundation"]
    def key(m):
        cat = MODEL_CATEGORIES.get(m, "Other")
        ci = cat_order.index(cat) if cat in cat_order else len(cat_order)
        return (ci, m)
    return sorted(models, key=key)


def plot_heatmap(pivot, metric, out_path):
    if pivot.empty or pivot.isna().all().all():
        print(f"  skip {out_path.name}: no data")
        return
    ordered = _order_models(list(pivot.index))
    pivot = pivot.loc[ordered]
    fig_h = max(4, 0.35 * len(pivot) + 1.5)
    fig_w = max(7, 0.7 * len(pivot.columns) + 3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(pivot.values, cmap=plt.get_cmap("YlGnBu"),
                   aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(f"{metric} — mean across 5 seeds, per (model, dataset)",
                 fontsize=14, weight="bold", pad=15)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.iat[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color=("white" if v > 0.5 else "black"), fontsize=8)
            else:
                ax.text(j, i, "—", ha="center", va="center",
                        color="lightgray", fontsize=10)
    for tick, model in zip(ax.get_yticklabels(), pivot.index):
        cat = MODEL_CATEGORIES.get(model, "Other")
        tick.set_color(CATEGORY_COLORS.get(cat, "black"))
        tick.set_fontweight("bold")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(metric, rotation=270, labelpad=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_ranking(df, metric, out_path):
    df_ok = df[df["status"].map(_is_success)] if "status" in df.columns else df
    g = df_ok.groupby("model")[metric].agg(["mean", "std", "count"]).dropna(subset=["mean"])
    if g.empty:
        print(f"  skip {out_path.name}: no data")
        return
    g = g.reindex(_order_models(list(g.index)))
    colors = [CATEGORY_COLORS.get(MODEL_CATEGORIES.get(m, "Other"), "#999999")
              for m in g.index]
    fig_w = max(8, 0.45 * len(g) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    x = np.arange(len(g))
    ax.bar(x, g["mean"], yerr=g["std"].fillna(0), color=colors,
           edgecolor="white", linewidth=0.8, capsize=3, alpha=0.92)
    ax.set_xticks(x)
    ax.set_xticklabels(g.index, rotation=45, ha="right")
    ax.set_ylabel(f"Mean {metric}")
    ax.set_title(f"{metric} — mean ± std across all (dataset, seed) cells",
                 fontsize=14, weight="bold", pad=15)
    ax.set_ylim(0, max(1.0, (g["mean"] + g["std"].fillna(0)).max() * 1.1))
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=c, label=cat)
        for cat, c in CATEGORY_COLORS.items()
        if cat in {MODEL_CATEGORIES.get(m, "Other") for m in g.index}
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              fontsize=9, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_coverage(df, out_path):
    df = df.copy()
    df["is_ok"] = (df["status"].map(_is_success)) if "status" in df.columns else True
    counts = df.groupby(["model", "dataset"]).agg(
        ok=("is_ok", "sum"), total=("is_ok", "count")
    )
    pivot_ok = counts["ok"].unstack(fill_value=0)
    pivot_total = counts["total"].unstack(fill_value=0)
    if pivot_ok.empty:
        print(f"  skip {out_path.name}: no data")
        return
    ordered = _order_models(list(pivot_ok.index))
    pivot_ok = pivot_ok.loc[ordered]
    pivot_total = pivot_total.loc[ordered]
    completion_pct = (pivot_ok / pivot_total.replace(0, np.nan)).fillna(0)
    fig_h = max(4, 0.32 * len(pivot_ok) + 1.5)
    fig_w = max(7, 0.7 * len(pivot_ok.columns) + 3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "coverage", ["#DC2626", "#FBBF24", "#10B981"]
    )
    ax.imshow(completion_pct.values, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot_ok.columns)))
    ax.set_xticklabels(pivot_ok.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot_ok.index)))
    ax.set_yticklabels(pivot_ok.index)
    ax.set_title("Coverage — completed cells / total cells per (model, dataset)",
                 fontsize=14, weight="bold", pad=15)
    for i in range(len(pivot_ok)):
        for j in range(len(pivot_ok.columns)):
            ok = int(pivot_ok.iat[i, j])
            tot = int(pivot_total.iat[i, j])
            text = f"{ok}/{tot}" if tot > 0 else "—"
            color = "white" if completion_pct.iat[i, j] < 0.6 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8)
    for tick, model in zip(ax.get_yticklabels(), pivot_ok.index):
        cat = MODEL_CATEGORIES.get(model, "Other")
        tick.set_color(CATEGORY_COLORS.get(cat, "black"))
        tick.set_fontweight("bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_mtgflow_comparison(df, metric, out_path):
    if "MTGFLOW" not in df["model"].unique():
        print(f"  skip {out_path.name}: MTGFLOW not in results")
        return
    df_ok = df[df["status"].map(_is_success)] if "status" in df.columns else df
    g = df_ok.groupby(["model", "dataset"])[metric].mean().unstack()
    if "MTGFLOW" not in g.index or g.loc["MTGFLOW"].dropna().empty:
        print(f"  skip {out_path.name}: MTGFLOW has no usable {metric} values")
        return
    other = g.drop("MTGFLOW")
    if other.empty:
        print(f"  skip {out_path.name}: no baselines for comparison")
        return
    valid_cols = other.dropna(how="all", axis=1).columns
    if len(valid_cols) == 0:
        print(f"  skip {out_path.name}: no overlapping datasets")
        return
    other_v = other[valid_cols]
    best_other = other_v.max(axis=0, skipna=True)
    best_other_model = other_v.idxmax(axis=0, skipna=True)
    mtg = g.loc["MTGFLOW"].reindex(valid_cols)

    fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(valid_cols) + 2), 5))
    x = np.arange(len(valid_cols))
    w = 0.38
    ax.bar(x - w / 2, mtg.values, w, label="MTGFLOW",
           color=CATEGORY_COLORS["Density"], edgecolor="white", linewidth=0.8)
    ax.bar(x + w / 2, best_other.values, w, label="Best non-MTGFLOW baseline",
           color="#64748B", edgecolor="white", linewidth=0.8)
    for i in range(len(valid_cols)):
        if pd.notna(best_other.iloc[i]) and pd.notna(best_other_model.iloc[i]):
            ax.text(i + w / 2, best_other.iloc[i] + 0.01,
                    str(best_other_model.iloc[i]),
                    ha="center", fontsize=8, color="#475569")
    ax.set_xticks(x)
    ax.set_xticklabels(valid_cols, rotation=45, ha="right")
    ax.set_ylabel(f"Mean {metric}")
    ax.set_title(f"MTGFLOW vs best baseline per dataset — {metric}",
                 fontsize=14, weight="bold", pad=15)
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right", fontsize=10, frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", type=Path, default=Path("results"))
    parser.add_argument("--output_dir", type=Path, default=Path("figures"))
    parser.add_argument("--dedupe_deterministic", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Reading results from: {args.results_dir.resolve()}")
    print(f"Writing outputs to:   {args.output_dir.resolve()}")

    df = collect_all_results(args.results_dir)

    if args.dedupe_deterministic:
        DET = {"PCA", "KNN", "HBOS", "COPOD", "LOF", "CBLOF"}
        before = len(df)
        df = df.sort_values(["model", "file", "seed"])
        mask = df["model"].isin(DET)
        df_det = df[mask].drop_duplicates(subset=["model", "file"], keep="first")
        df = pd.concat([df[~mask], df_det], ignore_index=True)
        print(f"Deduped deterministic models: {before} -> {len(df)} rows.")

    df.to_csv(args.output_dir / "master_summary.csv", index=False)
    print(f"  wrote {args.output_dir / 'master_summary.csv'}")

    summary = per_model_summary(df)
    summary.to_csv(args.output_dir / "per_model_summary.csv")
    print(f"  wrote {args.output_dir / 'per_model_summary.csv'}")

    print("\nPer-model mean ± std (top metrics):")
    cols = [c for c in ["category", "AUC-PR_mean", "AUC-PR_std",
                         "AUC-ROC_mean", "AUC-ROC_std", "AUC-PR_count"]
            if c in summary.columns]
    print(summary[cols].to_string())

    plot_heatmap(per_model_per_dataset(df, "AUC-PR"),
                 "AUC-PR", args.output_dir / "heatmap_aucpr.png")
    plot_heatmap(per_model_per_dataset(df, "AUC-ROC"),
                 "AUC-ROC", args.output_dir / "heatmap_aucroc.png")
    plot_ranking(df, "AUC-PR", args.output_dir / "ranking_aucpr.png")
    plot_ranking(df, "AUC-ROC", args.output_dir / "ranking_aucroc.png")
    plot_coverage(df, args.output_dir / "coverage_report.png")
    plot_mtgflow_comparison(df, "AUC-PR",
                            args.output_dir / "mtgflow_vs_baseline_aucpr.png")
    plot_mtgflow_comparison(df, "AUC-ROC",
                            args.output_dir / "mtgflow_vs_baseline_aucroc.png")
    print("\nDone.")


if __name__ == "__main__":
    main()