#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke test for the HiRE-Flow integration.

Five checks (modelled on extensions/models/mtgflow/smoke_test.py):

  1. Import and instantiate HiREFlow_AD with default HPs.  No errors.
  2. Forward shape: fit on synthetic (2000, 10) for 2 epochs; decision_function
     returns ndarray of shape (2000,) with finite values.
  3. Param count includes contributions from all six new modules
     (GraphEncoder, GRU_fast, GRU_slow, GraphPredictor, Combine, GateMLP)
     and the four reused MTGFLOW components.
  4. Aux loss is active: L_aux > 0 and GraphPredictor.weight receives a
     non-zero gradient after a single forward-backward pass.
  5. End-to-end on TSB-AD format: load the bundled SMD 057 CSV, call
     run_HiREFlow with epochs=2, assert scores have the correct shape and
     are finite.

Usage (from the platform root, venv active):

    python -m extensions.models.hire_flow.smoke_test \\
        --dataset_dir TSB-AD/Datasets/TSB-AD-M/ \\
        --filename 057_SMD_id_1_Facility_tr_4529_1st_4629.csv \\
        --epochs 2 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from extensions.registry import register_all
register_all()

from TSB_AD.model_wrapper import Semisupervise_AD_Pool


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _check(label: str, ok: bool, detail: str = "") -> bool:
    mark = "PASS" if ok else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{mark}] {label}{suffix}")
    return ok


def _parse_cutoff(filename: str) -> int:
    """Extract the training cutoff from the TSB-AD filename convention."""
    import re
    m = re.search(r"_tr_(\d+)_", filename)
    if not m:
        raise ValueError(f"cannot parse train cutoff from {filename!r}")
    return int(m.group(1))


def main() -> int:
    parser = argparse.ArgumentParser(description="HiRE-Flow smoke test")
    parser.add_argument(
        "--dataset_dir",
        default="TSB-AD/Datasets/TSB-AD-M/",
    )
    parser.add_argument(
        "--filename",
        default="057_SMD_id_1_Facility_tr_4529_1st_4629.csv",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--log_level",
        default="WARNING",
        choices=("DEBUG", "INFO", "WARNING"),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    )

    print("=" * 70)
    print("HiRE-Flow smoke test")
    print("=" * 70)
    print(f"Epochs:  {args.epochs}")
    print(f"Seed:    {args.seed}")
    print(f"CUDA:    {torch.cuda.is_available()}")
    print()

    _set_seed(args.seed)
    results: list[bool] = []

    # ── Check 1: import and instantiation ─────────────────────────────────
    try:
        from extensions.models.hire_flow.detector import HiREFlow_AD
        from extensions.models.hire_flow.model import HiREFlowModel
        from extensions.models.hire_flow.runner import run_HiREFlow

        detector = HiREFlow_AD(verbose=False)
        ok1 = True
    except Exception as exc:
        ok1 = False
        print(f"  [FAIL] import/instantiation error: {exc}")
    results.append(_check("1. import and instantiate HiREFlow_AD", ok1))
    if not ok1:
        return 1

    # ── Check 2: forward shape on synthetic data ───────────────────────────
    print()
    print("  Check 2: forward shape (2000 × 10, 2 epochs) …")
    try:
        rng = np.random.default_rng(args.seed)
        X_syn = rng.standard_normal((2000, 10)).astype(np.float32)
        X_train_syn = X_syn[:1200]

        det2 = HiREFlow_AD(epochs=args.epochs, verbose=False)
        det2.fit(X_train_syn)
        scores2 = det2.decision_function(X_syn)

        ok2a = scores2.shape == (2000,)
        ok2b = np.isfinite(scores2).all()
        ok2c = len(det2.train_loss_history_) == args.epochs
        ok2 = ok2a and ok2b and ok2c
    except Exception as exc:
        ok2 = False
        print(f"    error: {exc}")
    results.append(_check(
        "2. fit + decision_function shape / finite / loss history",
        ok2,
        f"shape={getattr(scores2, 'shape', '?')} "
        f"finite={np.isfinite(scores2).all() if ok2 else '?'} "
        f"epochs={len(getattr(det2, 'train_loss_history_', []))}",
    ))

    # ── Check 3: param count includes all six new modules ─────────────────
    print()
    print("  Check 3: parameter count across all modules …")
    try:
        K, L, d_g, d_fast, d_slow, d_rank, H = 10, 60, 32, 32, 32, 8, 32
        model_c3 = HiREFlowModel(
            n_blocks=1, input_size=1, hidden_size=H, n_hidden=1,
            window_size=L, n_sensor=K,
            d_g=d_g, d_fast=d_fast, d_slow=d_slow, d_rank=d_rank,
        )
        total_params = sum(p.numel() for p in model_c3.parameters())

        new_module_params = sum([
            sum(p.numel() for p in model_c3.graph_encoder.parameters()),
            sum(p.numel() for p in model_c3.gru_fast.parameters()),
            sum(p.numel() for p in model_c3.gru_slow.parameters()),
            sum(p.numel() for p in model_c3.graph_predictor.parameters()),
            sum(p.numel() for p in model_c3.combine.parameters()),
            sum(p.numel() for p in model_c3.gate_mlp.parameters()),
        ])
        old_module_params = sum([
            sum(p.numel() for p in model_c3.attention.parameters()),
            sum(p.numel() for p in model_c3.rnn.parameters()),
            sum(p.numel() for p in model_c3.gcn.parameters()),
            sum(p.numel() for p in model_c3.nf.parameters()),
        ])

        ok3 = (
            total_params > 0
            and new_module_params > 0
            and old_module_params > 0
            and (new_module_params + old_module_params) == total_params
        )
    except Exception as exc:
        ok3 = False
        total_params = new_module_params = old_module_params = 0
        print(f"    error: {exc}")
    results.append(_check(
        "3. param count covers all 10 sub-modules",
        ok3,
        f"total={total_params:,}  new={new_module_params:,}  "
        f"mtgflow={old_module_params:,}",
    ))

    # ── Check 4: aux loss is active, GraphPredictor receives gradient ──────
    print()
    print("  Check 4: aux loss active + GraphPredictor gradient …")
    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        K4, L4, B4 = 10, 60, 4
        model_c4 = HiREFlowModel(
            n_blocks=1, input_size=1, hidden_size=32, n_hidden=1,
            window_size=L4, n_sensor=K4,
        ).to(device)

        rng4 = torch.Generator(device=device)
        rng4.manual_seed(args.seed)
        x_prev = torch.randn(B4, K4, L4, 1, device=device, generator=rng4)
        x_curr = torch.randn(B4, K4, L4, 1, device=device, generator=rng4)
        x_next = torch.randn(B4, K4, L4, 1, device=device, generator=rng4)
        r_fast = torch.zeros(1, 1, 32, device=device)
        r_slow = torch.zeros(1, 1, 32, device=device)

        log_prob, _, _, A_hat_upper, A_next_upper = model_c4.forward_train(
            x_prev, x_curr, x_next, r_fast, r_slow
        )
        L_flow = -log_prob.mean()
        L_aux = F.mse_loss(A_hat_upper, A_next_upper.detach())
        (L_flow + 0.1 * L_aux).backward()

        l_aux_val = float(L_aux.item())
        gp_grad = model_c4.graph_predictor.weight.grad
        ok4a = l_aux_val > 0.0
        ok4b = gp_grad is not None and float(gp_grad.abs().max().item()) > 0.0

        ok4 = ok4a and ok4b
    except Exception as exc:
        ok4 = False
        l_aux_val = -1.0
        ok4a = ok4b = False
        print(f"    error: {exc}")
    results.append(_check(
        "4. L_aux > 0 and GraphPredictor gets non-zero gradient",
        ok4,
        f"L_aux={l_aux_val:.4f}  grad_ok={ok4b}",
    ))

    # ── Check 5: end-to-end on TSB-AD SMD 057 ─────────────────────────────
    print()
    print(f"  Check 5: end-to-end on {args.filename} …")
    import os
    csv_path = os.path.join(args.dataset_dir, args.filename)

    try:
        df = pd.read_csv(csv_path)
        label_col = df.columns[-1]
        y = df[label_col].values.astype(int)
        X_full = df.drop(columns=[label_col]).values.astype(np.float64)
        cut = _parse_cutoff(args.filename)
        X_train_e2e = X_full[:cut]

        _set_seed(args.seed)
        t0 = time.time()
        scores_e2e = run_HiREFlow(
            data_train=X_train_e2e,
            data_test=X_full,
            epochs=args.epochs,
            verbose=False,
        )
        elapsed = time.time() - t0

        ok5a = scores_e2e.shape == (len(X_full),)
        ok5b = np.isfinite(scores_e2e).all()
        ok5c = elapsed < 300.0  # must finish in 5 min on CPU
        ok5 = ok5a and ok5b and ok5c

        # Report a quick AUC-ROC as sanity signal
        from sklearn.metrics import roc_auc_score, average_precision_score
        auc_roc = float(roc_auc_score(y, scores_e2e)) if y.sum() > 0 else float("nan")
        auc_pr  = float(average_precision_score(y, scores_e2e)) if y.sum() > 0 else float("nan")
    except Exception as exc:
        ok5 = False
        elapsed = -1.0
        auc_roc = auc_pr = float("nan")
        print(f"    error: {exc}")
    results.append(_check(
        "5. end-to-end run_HiREFlow on SMD 057",
        ok5,
        f"shape={getattr(scores_e2e, 'shape', '?')}  "
        f"AUC-ROC={auc_roc:.4f}  AUC-PR={auc_pr:.4f}  "
        f"wall={elapsed:.1f}s",
    ))

    # ── HiREFlow in pool? ──────────────────────────────────────────────────
    print()
    in_pool = "HiREFlow" in Semisupervise_AD_Pool
    results.append(_check(
        "  (bonus) HiREFlow registered in Semisupervise_AD_Pool",
        in_pool,
    ))

    # ── Summary ───────────────────────────────────────────────────────────
    n_pass = sum(results)
    n_total = len(results)
    print()
    print("=" * 70)
    if n_pass == n_total:
        print(f"All {n_total} checks PASS — integration is ready.")
    else:
        print(f"{n_pass}/{n_total} checks passed.")
    print("=" * 70)

    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())