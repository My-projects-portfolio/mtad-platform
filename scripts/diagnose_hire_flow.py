#!/usr/bin/env python3
"""Diagnose HiRE-Flow per METHOD.md "What to monitor during training".

Replicates the detector's exact training loop and logs three signals each
batch:

  1. alpha.mean()  — captured via forward hook on `model.gate_mlp`. If it
                     stays above ~0.95, the relational path is dormant.
  2. L_aux         — captured directly from the loss term. Should decrease.
  3. gnorm_fast    — ||grad|| over gru_fast parameters.
     gnorm_slow    — ||grad|| over gru_slow parameters. Should be smaller
                     (10× LR difference + smaller gradient signal).

Usage (from project root, venv active):

    python scripts/diagnose_hire_flow.py \\
        --dataset_dir TSB-AD/Datasets/TSB-AD-M/ \\
        --filename 057_SMD_id_1_Facility_tr_4529_1st_4629.csv \\
        --epochs 8
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extensions.models.hire_flow.detector import (
    _SingleWindowDataset,
    _TripletWindowDataset,
)
from extensions.models.hire_flow.model import HiREFlowModel, aux_loss_kl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", default="TSB-AD/Datasets/TSB-AD-M/")
    p.add_argument("--filename", default="057_SMD_id_1_Facility_tr_4529_1st_4629.csv")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--window_size", type=int, default=60)
    p.add_argument("--stride_size", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--lr_slow_mult", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--lambda_aux", type=float, default=0.1)
    p.add_argument("--grad_clip_norm", type=float, default=1.0)
    p.add_argument("--hidden_size", type=int, default=32)
    p.add_argument("--n_blocks", type=int, default=1)
    p.add_argument("--n_hidden", type=int, default=1)
    p.add_argument("--d_g", type=int, default=32)
    p.add_argument("--d_fast", type=int, default=32)
    p.add_argument("--d_slow", type=int, default=32)
    p.add_argument("--d_rank", type=int, default=8)
    p.add_argument("--rel_dropout", type=float, default=0.1)
    p.add_argument("--gate_init_bias", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _grad_norm(params) -> float:
    total = 0.0
    n = 0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum().item())
            n += 1
    return float(np.sqrt(total)) if n else 0.0


def main() -> int:
    args = parse_args()
    _set_seed(args.seed)

    csv_path = Path(args.dataset_dir) / args.filename
    df = pd.read_csv(csv_path)
    label_col = "Label"
    data_cols = [c for c in df.columns if c != label_col]
    full = df[data_cols].to_numpy(dtype=np.float64)

    stem = args.filename.rsplit(".", 1)[0]
    train_index = int(stem.split("_tr_")[-1].split("_")[0])
    X_train = full[:train_index]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  X_train: {X_train.shape}  epochs: {args.epochs}")

    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X_train).astype(np.float32)

    base_ds = _SingleWindowDataset(X_norm, args.window_size, args.stride_size)
    triplet_ds = _TripletWindowDataset(base_ds)
    loader = DataLoader(
        triplet_ds, batch_size=args.batch_size, shuffle=False, drop_last=False,
    )
    n_features = X_train.shape[1]
    print(f"windows: {len(base_ds)}  triplets: {len(triplet_ds)}  "
          f"batches/epoch: {len(loader)}  K (sensors): {n_features}")

    model = HiREFlowModel(
        n_blocks=args.n_blocks,
        input_size=1,
        hidden_size=args.hidden_size,
        n_hidden=args.n_hidden,
        window_size=args.window_size,
        n_sensor=n_features,
        dropout=0.0,
        batch_norm=False,
        d_g=args.d_g,
        d_fast=args.d_fast,
        d_slow=args.d_slow,
        d_rank=args.d_rank,
        rel_dropout=args.rel_dropout,
        gate_init_bias=args.gate_init_bias,
    ).to(device)

    # Two-parameter-group optimizer (must match detector)
    slow_params = list(model.gru_slow.parameters())
    slow_ids = {id(p) for p in slow_params}
    base_params = [p for p in model.parameters() if id(p) not in slow_ids]
    optimizer = torch.optim.Adam(
        [
            {"params": base_params, "lr": args.lr, "weight_decay": args.weight_decay},
            {"params": slow_params, "lr": args.lr * args.lr_slow_mult,
             "weight_decay": args.weight_decay},
        ]
    )

    # Forward hook on gate_mlp captures pre-sigmoid logits
    gate_logits_holder = {"v": None}

    def gate_hook(_module, _inp, out):
        gate_logits_holder["v"] = out.detach()

    handle = model.gate_mlp.register_forward_hook(gate_hook)

    fast_params = list(model.gru_fast.parameters())
    pred_params = list(model.graph_predictor.parameters())

    model.train()
    r_fast = torch.zeros(1, 1, args.d_fast, device=device)
    r_slow = torch.zeros(1, 1, args.d_slow, device=device)

    print()
    print(f"{'epoch':>5}  {'L_flow':>8}  {'L_aux':>8}  "
          f"{'alpha_mean':>10}  {'alpha_min':>9}  {'alpha_max':>9}  "
          f"{'gn_fast':>8}  {'gn_slow':>8}  {'gn_pred':>8}  {'ratio':>6}")
    print("-" * 110)

    for epoch in range(args.epochs):
        ep_L_flow, ep_L_aux = [], []
        ep_alpha_mean, ep_alpha_min, ep_alpha_max = [], [], []
        ep_gn_fast, ep_gn_slow, ep_gn_pred = [], [], []

        for x_prev, x_curr, x_next in loader:
            x_prev = x_prev.to(device)
            x_curr = x_curr.to(device)
            x_next = x_next.to(device)
            optimizer.zero_grad()

            log_prob, r_fast_new, r_slow_new, A_hat, A_next = model.forward_train(
                x_prev, x_curr, x_next, r_fast, r_slow,
            )
            L_flow = -log_prob.mean()
            L_aux = aux_loss_kl(A_hat, A_next.detach())  # match detector
            loss = L_flow + args.lambda_aux * L_aux

            if not torch.isfinite(loss):
                r_fast = r_fast_new.detach()
                r_slow = r_slow_new.detach()
                continue

            loss.backward()
            gn_fast = _grad_norm(fast_params)
            gn_slow = _grad_norm(slow_params)
            gn_pred = _grad_norm(pred_params)

            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            r_fast = r_fast_new.detach()
            r_slow = r_slow_new.detach()

            alpha = torch.sigmoid(gate_logits_holder["v"])
            ep_alpha_mean.append(float(alpha.mean().item()))
            ep_alpha_min.append(float(alpha.min().item()))
            ep_alpha_max.append(float(alpha.max().item()))
            ep_L_flow.append(float(L_flow.item()))
            ep_L_aux.append(float(L_aux.item()))
            ep_gn_fast.append(gn_fast)
            ep_gn_slow.append(gn_slow)
            ep_gn_pred.append(gn_pred)

        if ep_L_flow:
            mean_fast = float(np.mean(ep_gn_fast))
            mean_slow = float(np.mean(ep_gn_slow))
            ratio = mean_slow / mean_fast if mean_fast > 0 else float("nan")
            print(
                f"{epoch + 1:>5d}  "
                f"{np.mean(ep_L_flow):>8.4f}  {np.mean(ep_L_aux):>8.5f}  "
                f"{np.mean(ep_alpha_mean):>10.4f}  "
                f"{np.min(ep_alpha_min):>9.4f}  {np.max(ep_alpha_max):>9.4f}  "
                f"{mean_fast:>8.4f}  {mean_slow:>8.4f}  "
                f"{float(np.mean(ep_gn_pred)):>8.4f}  {ratio:>6.3f}"
            )

    handle.remove()

    print()
    print("Verdict guide:")
    print("  alpha_mean staying >0.95 across epochs  → relational path dormant")
    print("  L_aux flat across epochs                → graph predictor not learning")
    print("  ratio (slow/fast) ≈ 1.0                 → param-group split is broken")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
