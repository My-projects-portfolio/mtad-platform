#!/usr/bin/env python3
"""
visualize_dataset.py — Plot a TSB-AD-M dataset showing channels, train/test
split, and anomaly regions.

Reads a TSB-AD CSV (numeric feature columns + trailing Label column), and
produces a multi-panel PNG showing:
  - The first N channels of the multivariate series (default: 6)
  - A dashed vertical line at the train/test split (from the filename's tr_NNNN)
  - Red shaded regions for every anomaly interval in the label column
  - A summary panel at the bottom showing the binary anomaly label timeline

Usage:
    python visualize_dataset.py \\
        --dataset_dir TSB-AD/Datasets/TSB-AD-M/ \\
        --filename 057_SMD_id_1_Facility_tr_4529_1st_4629.csv \\
        --output_dir figures/ \\
        --max_channels 6
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
except ImportError:
    print("ERROR: matplotlib is required.  pip install matplotlib", file=sys.stderr)
    sys.exit(1)


def parse_filename(filename: str) -> dict:
    """Extract dataset family, train cutoff, and first-anomaly index from filename.

    TSB-AD format: NNN_FAMILY_id_K_DOMAIN_tr_TRAINIDX_1st_FIRSTANOM.csv
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    info = {"family": parts[1] if len(parts) > 1 else "?", "stem": stem}

    m = re.search(r"_tr_(\d+)_", stem)
    info["train_index"] = int(m.group(1)) if m else None

    m = re.search(r"_1st_(\d+)", stem)
    info["first_anomaly"] = int(m.group(1)) if m else None
    return info


def find_anomaly_intervals(label: np.ndarray) -> list[tuple[int, int]]:
    """Return list of (start, end) intervals where label == 1 (inclusive)."""
    intervals: list[tuple[int, int]] = []
    in_anom = False
    start = 0
    for i, v in enumerate(label):
        if v and not in_anom:
            start = i
            in_anom = True
        elif not v and in_anom:
            intervals.append((start, i - 1))
            in_anom = False
    if in_anom:
        intervals.append((start, len(label) - 1))
    return intervals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=Path,
                        default=Path("TSB-AD/Datasets/TSB-AD-M/"))
    parser.add_argument("--filename", type=str, required=True,
                        help="CSV filename inside dataset_dir (with .csv extension).")
    parser.add_argument("--output_dir", type=Path, default=Path("figures"))
    parser.add_argument("--max_channels", type=int, default=6,
                        help="How many channels to plot.  Default: 6")
    parser.add_argument("--standardize", action="store_true",
                        help="Per-channel mean/std normalisation before plotting.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.dataset_dir / args.filename
    if not csv_path.exists():
        print(f"ERROR: file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {csv_path}")
    df = pd.read_csv(csv_path).dropna()
    if "Label" not in df.columns:
        # TSB-AD's label column may be named differently in some files
        candidate = [c for c in df.columns if c.lower() == "label"]
        if not candidate:
            print("ERROR: no Label column found.", file=sys.stderr)
            sys.exit(1)
        label_col = candidate[0]
    else:
        label_col = "Label"

    data = df.drop(columns=[label_col]).values.astype(float)
    label = df[label_col].astype(int).values
    n_samples, n_features = data.shape

    info = parse_filename(args.filename)
    train_idx = info["train_index"]
    first_anom = info["first_anomaly"]

    print(f"  shape       = ({n_samples}, {n_features})")
    print(f"  dataset     = {info['family']}")
    print(f"  train_idx   = {train_idx}")
    print(f"  first_anom  = {first_anom}")
    print(f"  anomaly %   = {100 * label.mean():.2f}")

    intervals = find_anomaly_intervals(label)
    print(f"  intervals   = {len(intervals)}  (total {label.sum()} anomalous timestamps)")

    if args.standardize:
        mu = data.mean(axis=0, keepdims=True)
        sigma = data.std(axis=0, keepdims=True) + 1e-9
        data = (data - mu) / sigma

    # ---- Plot ----
    n_panels = min(args.max_channels, n_features) + 1  # +1 for label panel
    fig, axes = plt.subplots(n_panels, 1, figsize=(13, 1.4 * n_panels + 1),
                              sharex=True, gridspec_kw={"hspace": 0.15})
    if n_panels == 1:
        axes = [axes]

    t = np.arange(n_samples)

    # Channels
    for ch in range(min(args.max_channels, n_features)):
        ax = axes[ch]
        ax.plot(t, data[:, ch], color="#0F172A", linewidth=0.7)
        ax.set_ylabel(f"ch {ch}", fontsize=10, rotation=0, ha="right", va="center", labelpad=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=9)

        # anomaly bands
        for (s, e) in intervals:
            ax.axvspan(s, e, color="#EF4444", alpha=0.18, lw=0)

        # train/test split
        if train_idx is not None:
            ax.axvline(train_idx, color="#0891B2", linestyle="--",
                       linewidth=1.2, alpha=0.9)

    # Label panel
    ax = axes[-1]
    ax.fill_between(t, 0, label.astype(float), color="#EF4444",
                    alpha=0.85, linewidth=0, step="post")
    ax.set_ylim(-0.05, 1.1)
    ax.set_yticks([0, 1])
    ax.set_ylabel("Label", fontsize=10, rotation=0, ha="right", va="center", labelpad=18)
    ax.set_xlabel("Timestep", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if train_idx is not None:
        ax.axvline(train_idx, color="#0891B2", linestyle="--", linewidth=1.2, alpha=0.9)
    ax.tick_params(labelsize=9)

    # Title
    title = (f"{info['family']} — {args.filename}\n"
             f"shape ({n_samples} timesteps, {n_features} channels) · "
             f"train cutoff = {train_idx} · anomaly fraction = {100*label.mean():.2f}%")
    fig.suptitle(title, fontsize=12, y=0.995)

    # Legend (annotation block)
    legend_text = ("─ ─  train / test split\n"
                   "█    anomaly intervals (red)\n"
                   f"first anomaly at row {first_anom}")
    fig.text(0.99, 0.99, legend_text, ha="right", va="top",
             fontsize=9, color="#475569",
             bbox=dict(facecolor="white", edgecolor="#CBD5E1", boxstyle="round,pad=0.4"))

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_name = f"dataset_overview_{Path(args.filename).stem}.png"
    out_path = args.output_dir / out_name
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nWrote {out_path}")
    if n_features > args.max_channels:
        print(f"Showing first {args.max_channels} of {n_features} channels.  "
              f"Pass --max_channels {n_features} to plot all.")


if __name__ == "__main__":
    main()