#!/usr/bin/env python3
r"""
generate_latex_tables.py — Emit publication-quality LaTeX tables from MTAD results.

Reads per-cell JSON sidecars from results/runs/<MODEL>/*.json and produces:
  - One LaTeX table per metric (AUC-PR, AUC-ROC, VUS-PR, VUS-ROC, ...)
  - One ranking table summarising mean ± std across all datasets per model
  - One combined "main results" table (AUC-PR + AUC-ROC mean ± std side-by-side)

Each table:
  - Rows grouped by methodology family (Classical / Reconstruction / ...)
  - Best value per column bolded
  - Second-best per column underlined
  - Cells with insufficient data shown as "—"
  - Uses booktabs for clean rules

Output files land in <output_dir>/latex/ as ready-to-\input .tex files,
plus a single mtad_tables.tex aggregator that can be \include'd from a paper.

Usage:
    python generate_latex_tables.py --results_dir results/ --output_dir figures/

Compile-time requirements in your manuscript preamble:
    \\usepackage{booktabs}
    \\usepackage{multirow}     % only if you keep the category column merged
    \\usepackage{rotating}     % only for sideways tables (not used by default)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Constants — match generate_analysis.py
# -----------------------------------------------------------------------------
METRIC_KEYS = [
    "AUC-PR", "AUC-ROC", "VUS-PR", "VUS-ROC",
    "Standard-F1", "PA-F1", "Event-based-F1", "R-based-F1", "Affiliation-F",
]

# LaTeX-safe versions of the metric names for headers
METRIC_LATEX = {
    "AUC-PR":         r"AUC-PR",
    "AUC-ROC":        r"AUC-ROC",
    "VUS-PR":         r"VUS-PR",
    "VUS-ROC":        r"VUS-ROC",
    "Standard-F1":    r"$F_1$",
    "PA-F1":          r"$F_1^{\text{PA}}$",
    "Event-based-F1": r"$F_1^{\text{evt}}$",
    "R-based-F1":     r"$F_1^{R}$",
    "Affiliation-F":  r"$F^{\text{aff}}$",
}

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

CATEGORY_ORDER = ["Classical", "Reconstruction", "Forecasting",
                  "Transformer", "Density", "Foundation"]

# Bold tag for our new method — change this if your novel method has a different name
HIGHLIGHT_MODEL = "MTGFLOW"

_SUCCESS_STATUSES = {"success", "ok", "completed", "done"}


# -----------------------------------------------------------------------------
# Loading (same logic as the patched generate_analysis.py)
# -----------------------------------------------------------------------------
def dataset_short_name(filename: str) -> str:
    stem = Path(str(filename)).stem.split("__")[0]
    parts = stem.split("_")
    return parts[1] if len(parts) > 1 else stem


def _find_metric(source: dict, canonical_key: str):
    aliases = METRIC_ALIASES.get(canonical_key, [canonical_key])
    for k in aliases:
        if k in source:
            try:
                return float(source[k])
            except (ValueError, TypeError):
                pass
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


def _is_success(status) -> bool:
    if status is None:
        return True
    return str(status).strip().lower() in _SUCCESS_STATUSES


def collect_all_results(results_dir: Path) -> pd.DataFrame:
    runs_dir = results_dir / "runs"
    if not runs_dir.exists():
        print(f"ERROR: {runs_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    rows: list[dict] = []
    for entry in sorted(runs_dir.iterdir()):
        if not entry.is_dir():
            continue
        model_name = entry.name
        for jp in sorted(entry.glob("*.json")):
            try:
                with open(jp) as f:
                    s = json.load(f)
            except Exception as e:
                print(f"  warn: skipping {jp.name}: {e}", file=sys.stderr)
                continue
            rec = {
                "model": model_name,
                "file": s.get("file") or s.get("filename") or jp.stem,
                "seed": s.get("seed"),
                "status": s.get("status", "success"),
            }
            for mk in METRIC_KEYS:
                rec[mk] = _find_metric(s, mk)
            rows.append(rec)

    if not rows:
        print("ERROR: no sidecars found.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows)
    df["dataset"] = df["file"].astype(str).apply(dataset_short_name)
    for k in METRIC_KEYS:
        if k in df.columns:
            df[k] = pd.to_numeric(df[k], errors="coerce")
    return df


# -----------------------------------------------------------------------------
# LaTeX helpers
# -----------------------------------------------------------------------------
def _order_models(models):
    def key(m):
        cat = MODEL_CATEGORIES.get(m, "Other")
        ci = CATEGORY_ORDER.index(cat) if cat in CATEGORY_ORDER else len(CATEGORY_ORDER)
        return (ci, m)
    return sorted(models, key=key)


def _esc(s: str) -> str:
    """Escape LaTeX-special characters in a string (table cells)."""
    s = str(s)
    return (s.replace("\\", r"\\")
             .replace("&", r"\&")
             .replace("%", r"\%")
             .replace("_", r"\_")
             .replace("#", r"\#")
             .replace("{", r"\{")
             .replace("}", r"\}")
             .replace("$", r"\$"))


def _format_cell(val, is_best=False, is_second=False, precision=3) -> str:
    """Format a numeric cell with best/second-best decoration."""
    if pd.isna(val):
        return "---"
    s = f"{val:.{precision}f}"
    if is_best:
        return r"\textbf{" + s + "}"
    if is_second:
        return r"\underline{" + s + "}"
    return s


def _format_mean_std(mean, std, is_best=False, is_second=False, precision=3) -> str:
    """Format 'mean ± std' for a ranking table cell."""
    if pd.isna(mean):
        return "---"
    m = f"{mean:.{precision}f}"
    if pd.isna(std):
        body = m
    else:
        body = m + r"$\pm$" + f"{std:.{precision}f}"
    if is_best:
        return r"\textbf{" + body + "}"
    if is_second:
        return r"\underline{" + body + "}"
    return body


def _best_indices(series: pd.Series):
    """Return (best_idx, second_idx) — handle NaNs and ties cleanly."""
    s = series.dropna()
    if len(s) == 0:
        return None, None
    sorted_s = s.sort_values(ascending=False)
    best = sorted_s.index[0]
    second = sorted_s.index[1] if len(sorted_s) > 1 else None
    return best, second


# -----------------------------------------------------------------------------
# Per-metric tables (model × dataset)
# -----------------------------------------------------------------------------
def emit_per_metric_table(df: pd.DataFrame, metric: str,
                          caption: str, label: str) -> str:
    """Emit a LaTeX table: rows = models (grouped by category), cols = datasets.
    Cell values are mean of metric across seeds. Best per column bolded, second underlined."""

    df_ok = df[df["status"].map(_is_success)] if "status" in df.columns else df
    pivot = df_ok.pivot_table(index="model", columns="dataset",
                              values=metric, aggfunc="mean")
    if pivot.empty:
        return ""

    ordered_models = _order_models(list(pivot.index))
    pivot = pivot.loc[ordered_models]
    datasets = list(pivot.columns)
    n_cols = len(datasets)

    # Identify per-column best and second
    best_per_col = {}
    second_per_col = {}
    for ds in datasets:
        b, s = _best_indices(pivot[ds])
        best_per_col[ds] = b
        second_per_col[ds] = s

    # Begin assembling LaTeX
    col_spec = "l" + "c" * n_cols
    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{tab:{label}}}")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    # Header row
    header = ["Method"] + [_esc(ds) for ds in datasets]
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")

    # Iterate by category so we can insert category labels and \midrule between groups
    last_cat = None
    for i, model in enumerate(ordered_models):
        cat = MODEL_CATEGORIES.get(model, "Other")
        if cat != last_cat:
            if last_cat is not None:
                lines.append(r"\midrule")
            lines.append(rf"\multicolumn{{{n_cols + 1}}}{{l}}{{\textit{{{cat}}}}} \\")
            last_cat = cat
        row_cells = []
        # Method name — highlight our novel method
        model_label = _esc(model)
        if model == HIGHLIGHT_MODEL:
            model_label = r"\textit{" + model_label + "}"
        row_cells.append(model_label)
        for ds in datasets:
            v = pivot.at[model, ds]
            row_cells.append(_format_cell(
                v,
                is_best=(model == best_per_col[ds]),
                is_second=(model == second_per_col[ds]),
                precision=3,
            ))
        lines.append(" & ".join(row_cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


# -----------------------------------------------------------------------------
# Ranking table — one row per model, columns are metrics with mean ± std
# -----------------------------------------------------------------------------
def emit_ranking_table(df: pd.DataFrame, metrics: list[str],
                       caption: str, label: str) -> str:
    """Per-model mean ± std across all (dataset, seed) cells, for several metrics."""
    df_ok = df[df["status"].map(_is_success)] if "status" in df.columns else df
    agg = df_ok.groupby("model")[metrics].agg(["mean", "std"])

    # Best/second-best per metric across all models
    best, second = {}, {}
    for m in metrics:
        b, s = _best_indices(agg[(m, "mean")])
        best[m] = b
        second[m] = s

    ordered_models = _order_models(list(agg.index))
    n_cols = len(metrics)

    col_spec = "l" + "c" * n_cols
    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{tab:{label}}}")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    header = ["Method"] + [METRIC_LATEX.get(m, m) for m in metrics]
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")

    last_cat = None
    for model in ordered_models:
        cat = MODEL_CATEGORIES.get(model, "Other")
        if cat != last_cat:
            if last_cat is not None:
                lines.append(r"\midrule")
            lines.append(rf"\multicolumn{{{n_cols + 1}}}{{l}}{{\textit{{{cat}}}}} \\")
            last_cat = cat

        model_label = _esc(model)
        if model == HIGHLIGHT_MODEL:
            model_label = r"\textit{" + model_label + "}"
        row_cells = [model_label]
        for m in metrics:
            mean = agg.at[model, (m, "mean")] if (m, "mean") in agg.columns else np.nan
            std = agg.at[model, (m, "std")] if (m, "std") in agg.columns else np.nan
            row_cells.append(_format_mean_std(
                mean, std,
                is_best=(model == best[m]),
                is_second=(model == second[m]),
                precision=3,
            ))
        lines.append(" & ".join(row_cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


# -----------------------------------------------------------------------------
# Combined "main results" table — two metrics side by side
# -----------------------------------------------------------------------------
def emit_main_results_table(df: pd.DataFrame,
                            primary: str = "AUC-PR",
                            secondary: str = "AUC-ROC") -> str:
    """Per-model: primary metric and secondary metric, mean ± std side by side."""
    df_ok = df[df["status"].map(_is_success)] if "status" in df.columns else df
    agg = df_ok.groupby("model")[[primary, secondary]].agg(["mean", "std"])

    best_p, second_p = _best_indices(agg[(primary, "mean")])
    best_s, second_s = _best_indices(agg[(secondary, "mean")])

    ordered_models = _order_models(list(agg.index))

    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        rf"\caption{{Main results: per-model mean $\pm$ std of {METRIC_LATEX.get(primary, primary)} "
        rf"and {METRIC_LATEX.get(secondary, secondary)} across all (dataset, seed) cells. "
        r"Best in \textbf{bold}, second-best \underline{underlined}. "
        rf"{HIGHLIGHT_MODEL} (in \textit{{italics}}) is the method introduced in this work.}}"
    )
    lines.append(r"\label{tab:main-results}")
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\toprule")
    lines.append(
        " & ".join([
            "Method",
            METRIC_LATEX.get(primary, primary),
            METRIC_LATEX.get(secondary, secondary),
        ]) + r" \\"
    )
    lines.append(r"\midrule")

    last_cat = None
    for model in ordered_models:
        cat = MODEL_CATEGORIES.get(model, "Other")
        if cat != last_cat:
            if last_cat is not None:
                lines.append(r"\midrule")
            lines.append(rf"\multicolumn{{3}}{{l}}{{\textit{{{cat}}}}} \\")
            last_cat = cat

        model_label = _esc(model)
        if model == HIGHLIGHT_MODEL:
            model_label = r"\textit{" + model_label + "}"
        p_mean = agg.at[model, (primary, "mean")]
        p_std = agg.at[model, (primary, "std")]
        s_mean = agg.at[model, (secondary, "mean")]
        s_std = agg.at[model, (secondary, "std")]
        row = [
            model_label,
            _format_mean_std(p_mean, p_std,
                             is_best=(model == best_p),
                             is_second=(model == second_p)),
            _format_mean_std(s_mean, s_std,
                             is_best=(model == best_s),
                             is_second=(model == second_s)),
        ]
        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


# -----------------------------------------------------------------------------
# Coverage table — completed / total cells per (model, dataset)
# -----------------------------------------------------------------------------
def emit_coverage_table(df: pd.DataFrame) -> str:
    df2 = df.copy()
    df2["is_ok"] = df2["status"].map(_is_success) if "status" in df2.columns else True
    counts = df2.groupby(["model", "dataset"]).agg(
        ok=("is_ok", "sum"), total=("is_ok", "count")
    )
    pivot_ok = counts["ok"].unstack(fill_value=0)
    pivot_total = counts["total"].unstack(fill_value=0)
    if pivot_ok.empty:
        return ""

    ordered_models = _order_models(list(pivot_ok.index))
    pivot_ok = pivot_ok.loc[ordered_models]
    pivot_total = pivot_total.loc[ordered_models]
    datasets = list(pivot_ok.columns)
    n_cols = len(datasets)

    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{Coverage: completed cells out of total cells per (model, dataset). "
        r"A cell counts as completed when its sidecar's status is success. "
        r"Gaps reflect documented architectural limitations.}"
    )
    lines.append(r"\label{tab:coverage}")
    lines.append(rf"\begin{{tabular}}{{l{'c' * n_cols}}}")
    lines.append(r"\toprule")
    lines.append(" & ".join(["Method"] + [_esc(ds) for ds in datasets]) + r" \\")
    lines.append(r"\midrule")

    last_cat = None
    for model in ordered_models:
        cat = MODEL_CATEGORIES.get(model, "Other")
        if cat != last_cat:
            if last_cat is not None:
                lines.append(r"\midrule")
            lines.append(rf"\multicolumn{{{n_cols + 1}}}{{l}}{{\textit{{{cat}}}}} \\")
            last_cat = cat

        model_label = _esc(model)
        if model == HIGHLIGHT_MODEL:
            model_label = r"\textit{" + model_label + "}"
        row = [model_label]
        for ds in datasets:
            ok = int(pivot_ok.at[model, ds])
            total = int(pivot_total.at[model, ds])
            if total == 0:
                row.append("---")
            elif ok == total:
                row.append(f"{ok}/{total}")
            else:
                row.append(r"\textbf{" + f"{ok}/{total}" + "}")
        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", type=Path, default=Path("results"))
    parser.add_argument("--output_dir", type=Path, default=Path("figures"))
    parser.add_argument("--dedupe_deterministic", action="store_true",
                        help="For deterministic classical models, keep only one seed.")
    args = parser.parse_args()

    latex_dir = args.output_dir / "latex"
    latex_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing LaTeX to: {latex_dir.resolve()}")

    df = collect_all_results(args.results_dir)
    print(f"Loaded {len(df)} cells across {df['model'].nunique()} models.")

    if args.dedupe_deterministic:
        DET = {"PCA", "KNN", "HBOS", "COPOD", "LOF", "CBLOF"}
        before = len(df)
        df = df.sort_values(["model", "file", "seed"])
        mask = df["model"].isin(DET)
        df_det = df[mask].drop_duplicates(subset=["model", "file"], keep="first")
        df = pd.concat([df[~mask], df_det], ignore_index=True)
        print(f"Deduped deterministic models: {before} -> {len(df)} rows.")

    # === 1. Main results table (AUC-PR + AUC-ROC side by side) ==============
    tex = emit_main_results_table(df)
    (latex_dir / "table_main_results.tex").write_text(tex)
    print(f"  wrote {latex_dir / 'table_main_results.tex'}")

    # === 2. Ranking table (all metrics, mean ± std per model) ===============
    tex = emit_ranking_table(
        df, METRIC_KEYS,
        caption=(r"Per-model mean $\pm$ std of all nine TSB-AD metrics "
                 r"across 10 datasets and 5 random seeds. "
                 r"Best in \textbf{bold}, second-best \underline{underlined}."),
        label="ranking-all-metrics",
    )
    (latex_dir / "table_ranking_all_metrics.tex").write_text(tex)
    print(f"  wrote {latex_dir / 'table_ranking_all_metrics.tex'}")

    # === 3. One table per metric (model × dataset) ==========================
    for metric in METRIC_KEYS:
        m_label = metric.replace(" ", "_").replace("/", "_")
        caption = (
            rf"{METRIC_LATEX.get(metric, metric)} per (model, dataset), "
            r"mean across 5 seeds. Best per column in \textbf{bold}, "
            r"second-best \underline{underlined}, missing cells (---) indicate "
            r"failed cells (e.g. OOM on high-dimensional inputs)."
        )
        tex = emit_per_metric_table(df, metric, caption=caption,
                                     label=f"by-dataset-{m_label.lower()}")
        if tex:
            out_path = latex_dir / f"table_by_dataset_{m_label}.tex"
            out_path.write_text(tex)
            print(f"  wrote {out_path}")

    # === 4. Coverage table ==================================================
    tex = emit_coverage_table(df)
    if tex:
        (latex_dir / "table_coverage.tex").write_text(tex)
        print(f"  wrote {latex_dir / 'table_coverage.tex'}")

    # === 5. Master aggregator file (\input each table) =====================
    master = []
    master.append("% Auto-generated by generate_latex_tables.py")
    master.append("% Requires in your preamble:")
    master.append(r"%   \usepackage{booktabs}")
    master.append("")
    master.append(r"\input{latex/table_main_results.tex}")
    master.append(r"\input{latex/table_ranking_all_metrics.tex}")
    for metric in METRIC_KEYS:
        m_label = metric.replace(" ", "_").replace("/", "_")
        master.append(rf"\input{{latex/table_by_dataset_{m_label}.tex}}")
    master.append(r"\input{latex/table_coverage.tex}")
    (args.output_dir / "mtad_tables.tex").write_text("\n".join(master) + "\n")
    print(f"  wrote {args.output_dir / 'mtad_tables.tex'}")

    print("\nDone.  Use the tables in your manuscript like:")
    print(r"    \input{figures/mtad_tables.tex}        % include all tables")
    print(r"    \input{figures/latex/table_main_results.tex}  % just the headline table")


if __name__ == "__main__":
    main()