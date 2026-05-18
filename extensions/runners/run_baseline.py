#!/usr/bin/env python3
"""
run_baseline.py

Multi-seed wrapper around TSB-AD's runner with cost instrumentation.

For each (file, seed) pair this captures:
  - The 9 TSB-AD detection metrics
  - wall_time_seconds, plus fit_time_seconds and score_time_seconds separately
    (score_time is ~0 for unsupervised models — TSB-AD reads scores from
    clf.decision_scores_ directly rather than calling decision_function)
  - infer_per_window_ms — derived from score_time / n_test_windows for
    semisupervised models; None for unsupervised
  - peak_gpu_memory_bytes via torch.cuda.max_memory_allocated
  - param_count — sum of trainable parameters across all top-level
    torch.nn.Module instances created during the run (0 for sklearn-based
    classical models like IForest/LOF/OCSVM)
  - Full reproducibility metadata: git SHA, env, HP dict, seed, status, traceback

Resume behaviour: success sidecars are skipped; error sidecars are re-run and
overwritten. Pass --force to re-run all cells, including successes.

Note on multi-seed for classical models:
  TSB-AD's stock wrappers for IForest, LOF, OCSVM do not forward random_state
  to the underlying class — they're effectively deterministic. Run those with
  a single seed. Multi-seed is meaningful for the deep models.

Outputs:
  <output_dir>/runs/<Model>/<file_stem>__seed<N>.json     # per-run record
  <output_dir>/scores/<Model>/<file_stem>__seed<N>.npy    # raw anomaly scores
  <output_dir>/runs/<Model>_summary.csv                   # flat aggregate

Usage:
  python run_baseline.py --model IForest \
      --file_list extensions/configs/dev_subset.csv \
      --dataset_dir TSB-AD/Datasets/TSB-AD-M/ \
      --output_dir results/ --seeds 42
"""

import argparse
import importlib
import inspect
import json
import pkgutil
import platform
import random
import subprocess
import sys
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

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


# ---- sklearn 1.6+ compatibility patch for PyOD-style BaseDetector -----------

from sklearn.base import BaseEstimator as _SkBaseEstimator
from TSB_AD.models.base import BaseDetector as _TSBBaseDetector

if not hasattr(_TSBBaseDetector, "__sklearn_tags__"):
    def _sklearn_tags(self):
        return _SkBaseEstimator().__sklearn_tags__()
    _TSBBaseDetector.__sklearn_tags__ = _sklearn_tags


# ---- Instrumentation: fit / score time split + parameter count --------------
#
# Approach:
#   - Eagerly import all TSB-AD model modules at startup so every BaseDetector
#     subclass exists before we patch.
#   - Walk subclasses and wrap their fit / decision_function methods with timing
#     wrappers that accumulate into _RUN_STATE.
#   - Hook torch.nn.Module.__init__ to register every module created during a
#     run. Post-run we identify top-level modules and sum their trainable params.

_RUN_STATE = {
    "fit_time": 0.0,
    "score_time": 0.0,
    "modules": [],
}


def _reset_run_state():
    _RUN_STATE["fit_time"] = 0.0
    _RUN_STATE["score_time"] = 0.0
    _RUN_STATE["modules"] = []


def _eagerly_import_tsb_models():
    """Import every module under TSB_AD.models so all BaseDetector subclasses
    are present in the subclass tree before we wrap methods.

    Silences per-module import errors — some models depend on optional packages
    (foundation models, etc.) that may not be installed.
    """
    import TSB_AD.models as _models_pkg
    failed = []
    for info in pkgutil.iter_modules(_models_pkg.__path__):
        if info.name.startswith("_"):
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                importlib.import_module(f"TSB_AD.models.{info.name}")
        except Exception:
            failed.append(info.name)
    if failed:
        preview = failed[:5]
        suffix = " ..." if len(failed) > 5 else ""
        print(f"[setup] Skipped {len(failed)} model modules with import "
              f"errors (usually optional deps): {preview}{suffix}")


def _instrument_model_class_timing():
    """Wrap fit / decision_function on every model class in TSB_AD.models.*
    that defines both methods directly. This catches both BaseDetector
    subclasses AND stand-alone model classes (LSTMAD, xLSTMAD, etc.) that
    don't inherit from BaseDetector but still expose the same interface.
    Idempotent.
    """
    import TSB_AD.models as _models_pkg

    wrapped = 0
    for info in pkgutil.iter_modules(_models_pkg.__path__):
        if info.name.startswith("_") or info.name == "base":
            continue
        try:
            mod = importlib.import_module(f"TSB_AD.models.{info.name}")
        except Exception:
            continue

        for attr_name in dir(mod):
            obj = getattr(mod, attr_name, None)
            if not inspect.isclass(obj):
                continue
            # Only classes DEFINED in this module — skip re-exports
            if obj.__module__ != mod.__name__:
                continue
            # Must have BOTH fit and decision_function directly defined
            if "fit" not in obj.__dict__ or "decision_function" not in obj.__dict__:
                continue

            for method_name in ("fit", "decision_function"):
                original = obj.__dict__[method_name]
                if getattr(original, "_timing_wrapped", False):
                    continue

                def make_wrapper(orig, name):
                    def wrapper(self, *args, **kwargs):
                        t0 = time.time()
                        try:
                            return orig(self, *args, **kwargs)
                        finally:
                            elapsed = time.time() - t0
                            if name == "fit":
                                _RUN_STATE["fit_time"] += elapsed
                            else:
                                _RUN_STATE["score_time"] += elapsed
                    wrapper._timing_wrapped = True
                    wrapper.__name__ = orig.__name__
                    return wrapper

                setattr(obj, method_name, make_wrapper(original, method_name))
                wrapped += 1

    print(f"[setup] Instrumented {wrapped} model class methods for fit/score timing")

def _hook_nn_module_init():
    """Patch nn.Module.__init__ to register every instance into _RUN_STATE."""
    original_init = nn.Module.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        _RUN_STATE["modules"].append(self)

    nn.Module.__init__ = patched_init


def _compute_param_count():
    """Sum trainable parameters across the top-level modules created during
    this run. 'Top-level' = not contained as a sub-module of any other module
    we captured. This handles both single-model and multi-model architectures
    (e.g. USAD with two separate networks).
    """
    modules = _RUN_STATE["modules"]
    if not modules:
        return 0

    child_ids = set()
    for m in modules:
        for sub in m.modules():
            if sub is not m:
                child_ids.add(id(sub))

    roots = [m for m in modules if id(m) not in child_ids]
    total = 0
    for m in roots:
        for p in m.parameters():
            if p.requires_grad:
                total += p.numel()
    return int(total)


# Apply all instrumentation at import time, before the first run_one call.
_eagerly_import_tsb_models()
_instrument_model_class_timing()
_hook_nn_module_init()


# ---- Standard helpers --------------------------------------------------------

CANONICAL_SEEDS = [13, 17, 42, 1337, 2024]


def set_all_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def env_snapshot():
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

def run_one(model_name, filename, dataset_dir, seed, output_dir, force=False):
    sc_path = sidecar_path(output_dir, model_name, filename, seed)
    sp_path = score_path(output_dir, model_name, filename, seed)
    sc_path.parent.mkdir(parents=True, exist_ok=True)
    sp_path.parent.mkdir(parents=True, exist_ok=True)

    if sc_path.exists() and not force:
        with open(sc_path) as f:
            existing = json.load(f)
        if existing.get("status") == "success":
            return existing

    set_all_seeds(seed)
    _reset_run_state()

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
        wall_elapsed = time.time() - t_start

        if not isinstance(output, np.ndarray):
            raise RuntimeError(f"Model returned non-array: {output!r}")

        peak_gpu_bytes = (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        )
        param_count = _compute_param_count()
        fit_time = float(_RUN_STATE["fit_time"])
        score_time = float(_RUN_STATE["score_time"])

        n_test_windows = data.shape[0] - train_index
        if (model_name in Semisupervise_AD_Pool
                and n_test_windows > 0
                and score_time > 0):
            infer_per_window_ms = (score_time / n_test_windows) * 1000.0
        else:
            infer_per_window_ms = None

        np.save(sp_path, output)
        metrics = get_metrics(output, label, slidingWindow=slidingWindow)

        record.update({
            "status": "success",
            "n_samples": int(len(label)),
            "n_features": int(feats),
            "sliding_window": int(slidingWindow),
            "train_index": int(train_index),
            "wall_time_seconds": float(wall_elapsed),
            "fit_time_seconds": fit_time,
            "score_time_seconds": score_time,
            "infer_per_window_ms": infer_per_window_ms,
            "peak_gpu_memory_bytes": int(peak_gpu_bytes),
            "param_count": param_count,
            "score_path": str(sp_path.relative_to(output_dir)),
            "metrics": {k: float(v) for k, v in metrics.items()},
        })

    except Exception:
        record["traceback"] = traceback.format_exc()

    with open(sc_path, "w") as f:
        json.dump(record, f, indent=2)

    return record

def aggregate_summary_csv(output_dir, model_name):
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
            "fit_time_seconds": r.get("fit_time_seconds"),
            "score_time_seconds": r.get("score_time_seconds"),
            "infer_per_window_ms": r.get("infer_per_window_ms"),
            "peak_gpu_memory_bytes": r.get("peak_gpu_memory_bytes"),
            "param_count": r.get("param_count"),
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
    parser.add_argument("--model", required=True)
    parser.add_argument("--file_list", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seeds",
                        default=",".join(str(s) for s in CANONICAL_SEEDS))
    parser.add_argument("--force", action="store_true",
                        help="Re-run cells even if a success sidecar exists")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    seeds = [int(s) for s in args.seeds.split(",")]
    file_list = pd.read_csv(args.file_list)["file_name"].tolist()

    if args.model not in Optimal_Multi_algo_HP_dict:
        print(f"ERROR: {args.model} not in Optimal_Multi_algo_HP_dict",
              file=sys.stderr)
        sys.exit(1)

    hp_orig = Optimal_Multi_algo_HP_dict[args.model]
    hp_filt, dropped = filter_kwargs_for_wrapper(args.model, hp_orig)

    print(f"Model:         {args.model}")
    print(f"HP (original): {hp_orig}")
    print(f"HP (used):     {hp_filt}")
    if dropped:
        print(f"HP (dropped):  {dropped}   <- not accepted by run_{args.model}")
    print(f"Files:         {len(file_list)}")
    print(f"Seeds:         {seeds}")
    print(f"Total cells:   {len(file_list) * len(seeds)}")
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
                print(f"  [rerun]   {fn} seed={seed} (previous attempt errored)")
                reran_errors += 1
            else:
                print(f"  [run]     {fn} seed={seed}", flush=True)

            rec = run_one(args.model, fn, dataset_dir, seed, output_dir, force=args.force)
            if rec["status"] == "success":
                m = rec["metrics"]
                ipw = rec.get("infer_per_window_ms")
                ipw_str = f"  infer/win={ipw:.2f}ms" if ipw is not None else ""
                print(
                    f"            OK   wall={rec['wall_time_seconds']:6.1f}s  "
                    f"fit={rec['fit_time_seconds']:5.1f}s  "
                    f"AUC-PR={m.get('AUC-PR', float('nan')):.3f}  "
                    f"params={rec['param_count']}  "
                    f"peak_gpu_MB={rec['peak_gpu_memory_bytes']/1e6:.0f}"
                    f"{ipw_str}"
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