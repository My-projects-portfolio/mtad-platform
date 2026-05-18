#!/usr/bin/env python3
"""
run_baseline.py

Multi-seed wrapper around TSB-AD's existing runner machinery.

For each (file, seed) pair:
  - Re-seeds torch / numpy / random
  - Calls TSB-AD's run_Unsupervise_AD or run_Semisupervise_AD with the model's
    Optimal HP dict — filtering out any keys the wrapper function doesn't accept
  - Captures wall time, peak GPU memory, status, and the 9 TSB-AD detection metrics
  - Saves the raw score array (.npy) and a JSON sidecar with full metadata
  - On exception, writes a status=error sidecar with traceback — never the silent
    zero-row case from TSB-AD's stock runner

Resume behavior:
  - Cells whose sidecar exists with status=success are skipped
  - Cells whose sidecar exists with status=error are re-run (and overwritten)
  - Pass --force to re-run all cells regardless of existing sidecars

A note on seeds for classical models:
  TSB-AD's stock wrappers for IForest, LOF, OCSVM do NOT forward random_state
  to the underlying class — they're effectively deterministic. Running 5 seeds
  for these models produces 5 identical results. Use --seeds 42 (or any single
  value) for those. Genuine multi-seed runs (5 seeds) are meaningful for the
  deep models: AE, USAD, OmniAnomaly, TranAD, AnomalyTransformer, LSTMAD, TimesNet.

Outputs:
  <output_dir>/runs/<Model>/<file_stem>__seed<N>.json     # per-run record
  <output_dir>/scores/<Model>/<file_stem>__seed<N>.npy    # raw anomaly scores
  <output_dir>/runs/<Model>_summary.csv                   # flat aggregate, rebuilt each run

Usage:
  python run_baseline.py \
      --model IForest \
      --file_list extensions/configs/dev_subset.csv \
      --dataset_dir TSB-AD/Datasets/TSB-AD-M/ \
      --output_dir results/ \
      --seeds 42
"""

import argparse
import inspect
import json
import platform
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.HP_list import Optimal_Multi_algo_HP_dict
from TSB_AD import model_wrapper as _model_wrapper
from TSB_AD.model_wrapper import (
    Semisupervise_AD_Pool,
    Unsupervise_AD_Pool,
    run_Semisupervise_AD,
    run_Unsupervise_AD,
)
from TSB_AD.utils.slidingWindows import find_length_rank


# Canonical seed list. Five seeds for stochastic deep models; reusing the same
# list across models enables paired statistical comparisons.
CANONICAL_SEEDS = [13, 17, 42, 1337, 2024]


def set_all_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def env_snapshot() -> dict:
    gpu_name = None
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = "cuda-available"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_name": gpu_name,
    }


def filter_kwargs_for_wrapper(model_name, hp_dict):
    """Drop HP keys the wrapper function ``run_<model_name>`` doesn't accept.

    Returns (filtered_hp, dropped_keys). Defensive measure: if the HP dict
    contains extras (e.g. ``random_state`` accidentally injected, or a key
    only the underlying class accepts), they get dropped here instead of
    crashing the run.
    """
    func = getattr(_model_wrapper, f"run_{model_name}", None)
    if func is None:
        return dict(hp_dict), []
    sig = inspect.signature(func)
    accepted = set(sig.parameters.keys())
    filtered = {k: v for k, v in hp_dict.items() if k in accepted}
    dropped = sorted(set(hp_dict.keys()) - accepted)
    return filtered, dropped


def sidecar_path(output_dir, model, filename, seed):
    stem = filename.replace(".csv", "")
    return output_dir / "runs" / model / f"{stem}__seed{seed}.json"


def score_path(output_dir, model, filename, seed):
    stem = filename.replace(".csv", "")
    return output_dir / "scores" / model / f"{stem}__seed{seed}.npy"


def run_one(model_name, filename, dataset_dir, seed, output_dir):
    """Run a single (model, file, seed) cell and write its sidecar.

    If a success sidecar already exists, return the cached record without
    re-running. Error sidecars are overwritten on re-run.
    """
    sc_path = sidecar_path(output_dir, model_name, filename, seed)
    sp_path = score_path(output_dir, model_name, filename, seed)
    sc_path.parent.mkdir(parents=True, exist_ok=True)
    sp_path.parent.mkdir(parents=True, exist_ok=True)

    if sc_path.exists():
        with open(sc_path) as f:
            existing = json.load(f)
        if existing.get("status") == "success":
            return existing

    set_all_seeds(seed)

    hp_original = dict(Optimal_Multi_algo_HP_dict[model_name])
    hp_filtered, dropped_keys = filter_kwargs_for_wrapper(model_name, hp_original)
    if dropped_keys:
        print(f"    [warn] {model_name}: dropped HP keys not accepted by "
              f"run_{model_name}: {dropped_keys}")

    record = {
        "model": model_name,
        "file": filename,
        "seed": seed,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "env": env_snapshot(),
        "hp": hp_filtered,
        "hp_dropped_keys": dropped_keys,
        "status": "error",
    }

    try:
        file_path = dataset_dir / filename
        df = pd.read_csv(file_path).dropna()
        data = df.iloc[:, 0:-1].values.astype(float)
        label = df["Label"].astype(int).to_numpy()
        feats = data.shape[1]
        slidingWindow = find_length_rank(data[:, 0].reshape(-1, 1), rank=1)
        train_index = int(filename.split(".")[0].split("_")[-3])
        data_train = data[:train_index, :]

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        t_start = time.time()
        if model_name in Semisupervise_AD_Pool:
            output = run_Semisupervise_AD(model_name, data_train, data, **hp_filtered)
        elif model_name in Unsupervise_AD_Pool:
            output = run_Unsupervise_AD(model_name, data, **hp_filtered)
        else:
            raise ValueError(
                f"{model_name} is not in Semisupervise_AD_Pool or Unsupervise_AD_Pool"
            )
        elapsed = time.time() - t_start

        if not isinstance(output, np.ndarray):
            raise RuntimeError(f"Model returned non-array: {output!r}")

        peak_gpu_bytes = (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        )

        np.save(sp_path, output)
        metrics = get_metrics(output, label, slidingWindow=slidingWindow)

        record.update({
            "status": "success",
            "n_samples": int(len(label)),
            "n_features": int(feats),
            "sliding_window": int(slidingWindow),
            "train_index": int(train_index),
            "wall_time_seconds": float(elapsed),
            "peak_gpu_memory_bytes": int(peak_gpu_bytes),
            "score_path": str(sp_path.relative_to(output_dir)),
            "metrics": {k: float(v) for k, v in metrics.items()},
        })

    except Exception:
        record["traceback"] = traceback.format_exc()

    with open(sc_path, "w") as f:
        json.dump(record, f, indent=2)

    return record


def aggregate_summary_csv(output_dir, model_name):
    """Rebuild a flat CSV from all sidecars (success rows only)."""
    sidecar_dir = output_dir / "runs" / model_name
    rows = []
    for json_path in sorted(sidecar_dir.glob("*__seed*.json")):
        with open(json_path) as f:
            r = json.load(f)
        if r.get("status") != "success":
            continue
        row = {
            "file": r["file"],
            "seed": r["seed"],
            "wall_time_seconds": r.get("wall_time_seconds"),
            "peak_gpu_memory_bytes": r.get("peak_gpu_memory_bytes"),
            "n_samples": r.get("n_samples"),
            "n_features": r.get("n_features"),
            "sliding_window": r.get("sliding_window"),
            **r.get("metrics", {}),
        }
        rows.append(row)

    csv_path = output_dir / "runs" / f"{model_name}_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model name, e.g. IForest")
    parser.add_argument("--file_list", required=True, help="CSV with file_name column")
    parser.add_argument("--dataset_dir", required=True, help="Path to TSB-AD-M/")
    parser.add_argument("--output_dir", required=True, help="Root output (e.g. results/)")
    parser.add_argument(
        "--seeds",
        default=",".join(str(s) for s in CANONICAL_SEEDS),
        help=f"Comma-separated seeds (default: {CANONICAL_SEEDS})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run cells even if a success sidecar exists (overwrites)",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    seeds = [int(s) for s in args.seeds.split(",")]
    file_list = pd.read_csv(args.file_list)["file_name"].tolist()

    if args.model not in Optimal_Multi_algo_HP_dict:
        print(f"ERROR: {args.model} not in Optimal_Multi_algo_HP_dict", file=sys.stderr)
        sys.exit(1)

    # Surface up front what will actually be passed (and what was dropped).
    hp_orig = Optimal_Multi_algo_HP_dict[args.model]
    hp_filt, dropped = filter_kwargs_for_wrapper(args.model, hp_orig)

    total_cells = len(file_list) * len(seeds)
    print(f"Model:         {args.model}")
    print(f"HP (original): {hp_orig}")
    print(f"HP (used):     {hp_filt}")
    if dropped:
        print(f"HP (dropped):  {dropped}   <- not accepted by run_{args.model}")
    print(f"Files:         {len(file_list)}")
    print(f"Seeds:         {seeds}")
    print(f"Total cells:   {total_cells}")
    print(f"Force re-run:  {args.force}")
    print(f"Dataset dir:   {dataset_dir}")
    print(f"Output dir:    {output_dir}")
    print(f"CUDA:          {torch.cuda.is_available()}")
    print("-" * 80)

    successes = errors = skipped = reran_errors = 0
    t_overall = time.time()

    for fn in file_list:
        for seed in seeds:
            sc_path = sidecar_path(output_dir, args.model, fn, seed)

            if sc_path.exists() and not args.force:
                with open(sc_path) as f:
                    existing = json.load(f)
                if existing.get("status") == "success":
                    print(f"  [skip-OK] {fn} seed={seed}")
                    skipped += 1
                    continue
                # Existing error sidecar: re-run, overwriting.
                print(f"  [rerun]   {fn} seed={seed} (previous attempt errored)")
                reran_errors += 1
            else:
                print(f"  [run]     {fn} seed={seed}", flush=True)

            rec = run_one(args.model, fn, dataset_dir, seed, output_dir)
            if rec["status"] == "success":
                m = rec["metrics"]
                print(
                    f"            OK   t={rec['wall_time_seconds']:6.1f}s  "
                    f"AUC-PR={m.get('AUC-PR', float('nan')):.3f}  "
                    f"VUS-PR={m.get('VUS-PR', float('nan')):.3f}  "
                    f"peak_gpu_MB={rec['peak_gpu_memory_bytes']/1e6:.0f}"
                )
                successes += 1
            else:
                tb_lines = (rec.get("traceback") or "").strip().splitlines()
                msg = tb_lines[-1] if tb_lines else "?"
                print(f"            ERR  {msg}")
                errors += 1

    summary_path = aggregate_summary_csv(output_dir, args.model)
    elapsed = time.time() - t_overall

    print("-" * 80)
    print(f"Done. success={successes}  error={errors}  "
          f"skipped(OK)={skipped}  reran(prev-error)={reran_errors}  "
          f"total_time={elapsed/60:.1f}min")
    print(f"Sidecars: {output_dir}/runs/{args.model}/")
    print(f"Scores:   {output_dir}/scores/{args.model}/")
    print(f"Summary:  {summary_path}")


if __name__ == "__main__":
    main()