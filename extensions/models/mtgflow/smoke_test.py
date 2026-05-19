#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke test for the MTGFLOW integration.

Runs MTGFLOW with deliberately short training (5 epochs) on a single TSB-AD-M
file. Checks the five correctness conditions from the integration plan:

  1. Model imports and instantiates cleanly.
  2. Training loss decreases over epochs.
  3. Score array has the right shape, dtype, and no NaN/Inf.
  4. AUC-ROC is above 0.5 (non-trivial signal).
  5. AUC-PR is above the naive baseline (the anomaly ratio).

If all five pass, the integration is sound and the full 10-dataset × 5-seed
sweep can proceed. If any fail, fix the wrapper before scaling up.

Usage (from the platform root, with the TSB-AD venv active):

    python -m extensions.models.mtgflow.smoke_test \\
        --dataset_dir TSB-AD/Datasets/TSB-AD-M/ \\
        --filename 057_SMD_id_1_Facility_tr_4529_1st_4629.csv \\
        --epochs 5 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch

# Register MTGFLOW with TSB-AD before importing anything that uses it.
from extensions.registry import register_all
register_all()

from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.model_wrapper import run_Semisupervise_AD, Semisupervise_AD_Pool
from TSB_AD.utils.slidingWindows import find_length_rank


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _check(label: str, ok: bool, detail: str = "") -> bool:
    mark = "PASS" if ok else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{mark}] {label}{suffix}")
    return ok


def main():
    parser = argparse.ArgumentParser(description="MTGFLOW smoke test")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="TSB-AD/Datasets/TSB-AD-M/",
        help="Directory containing TSB-AD-M CSV files",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="057_SMD_id_1_Facility_tr_4529_1st_4629.csv",
        help="Single CSV filename inside dataset_dir",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Short epoch count for fast smoke test (full sweep uses 40)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING"),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    )

    print("=" * 72)
    print("MTGFLOW integration smoke test")
    print("=" * 72)
    print(f"Dataset:   {args.filename}")
    print(f"Epochs:    {args.epochs}")
    print(f"Seed:      {args.seed}")
    print(f"CUDA:      {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU:       {torch.cuda.get_device_name(0)}")
    print()

    _set_seed(args.seed)

    # Pool membership: did register_all() actually register the model?
    if not _check(
        "MTGFLOW present in Semisupervise_AD_Pool",
        "MTGFLOW" in Semisupervise_AD_Pool,
        f"pool size = {len(Semisupervise_AD_Pool)}",
    ):
        sys.exit(1)

    # Load data the same way Run_Detector_M.py does.
    file_path = os.path.join(args.dataset_dir, args.filename)
    df = pd.read_csv(file_path).dropna()
    data = df.iloc[:, 0:-1].values.astype(float)
    label = df["Label"].astype(int).to_numpy()
    train_index = int(args.filename.split(".")[0].split("_")[-3])
    data_train = data[:train_index, :]
    n_samples, n_features = data.shape
    print(f"Series:    n_samples={n_samples}, n_features={n_features}")
    print(f"Train cut: {train_index} rows  (anomaly ratio in test region: "
          f"{label[train_index:].mean():.4f})")
    print()

    slidingWindow = find_length_rank(data[:, 0].reshape(-1, 1), rank=1)
    print(f"Estimated sliding window for metrics: {slidingWindow}")
    print()

    # Run MTGFLOW via TSB-AD's dispatch.
    print("Running MTGFLOW ...")
    t0 = time.time()
    scores = run_Semisupervise_AD(
        "MTGFLOW", data_train, data, epochs=args.epochs, verbose=True
    )
    elapsed = time.time() - t0
    print(f"  ... done in {elapsed:.1f}s")
    print()

    # === Correctness checks =====================================================
    all_ok = True

    # Check 1: scores is an ndarray of correct shape/dtype.
    all_ok &= _check(
        "scores is np.ndarray", isinstance(scores, np.ndarray),
        detail=f"got {type(scores).__name__}"
    )
    if not isinstance(scores, np.ndarray):
        print("Dispatch returned an error string instead of scores:")
        print(f"  {scores!r}")
        sys.exit(1)

    all_ok &= _check(
        "scores.shape == (n_samples,)",
        scores.shape == (n_samples,),
        detail=f"got {scores.shape}",
    )
    all_ok &= _check(
        "scores has no NaN", not np.isnan(scores).any()
    )
    all_ok &= _check(
        "scores has no Inf", not np.isinf(scores).any()
    )

    # Check 2: scores are in [0, 1] (MinMaxScaler applied by runner).
    all_ok &= _check(
        "scores within [0, 1]",
        (scores.min() >= -1e-6) and (scores.max() <= 1 + 1e-6),
        detail=f"range = [{scores.min():.4f}, {scores.max():.4f}]",
    )

    # Check 3: metrics finish and look non-trivial.
    metrics = get_metrics(scores, label, slidingWindow=slidingWindow)
    print()
    print("TSB-AD metrics:")
    for k, v in metrics.items():
        print(f"  {k:<24} = {v:.4f}")

    anomaly_ratio = label.mean()
    all_ok &= _check(
        "AUC-ROC > 0.5", metrics["AUC-ROC"] > 0.5,
        detail=f"AUC-ROC = {metrics['AUC-ROC']:.4f}"
    )
    all_ok &= _check(
        "AUC-PR > anomaly ratio",
        metrics["AUC-PR"] > anomaly_ratio,
        detail=f"AUC-PR = {metrics['AUC-PR']:.4f}, baseline = {anomaly_ratio:.4f}",
    )

    # Check 4: the (separately captured) training loss should have dropped.
    # We can read it back from the detector object only if we ran it directly;
    # the runner wrapper hides it. So instead instantiate the detector once
    # and read its history as a sanity check.
    print()
    print("Loss-curve check (separate run, 3 epochs):")
    _set_seed(args.seed)
    from extensions.models.mtgflow import MTGFLOW_AD

    detector = MTGFLOW_AD(epochs=3, verbose=False)
    detector.fit(data_train)
    history = detector.train_loss_history_
    print(f"  per-epoch neg-log-lik: {[round(x, 3) for x in history]}")
    decreased = len(history) >= 2 and history[-1] < history[0]
    all_ok &= _check(
        "training loss decreases (epoch[-1] < epoch[0])",
        decreased,
        detail=f"{history[0]:.3f} -> {history[-1]:.3f}",
    )

    print()
    print("=" * 72)
    if all_ok:
        print("SMOKE TEST PASSED — ready for the full sweep.")
        sys.exit(0)
    else:
        print("SMOKE TEST FAILED — investigate before running the full sweep.")
        sys.exit(2)


if __name__ == "__main__":
    main()