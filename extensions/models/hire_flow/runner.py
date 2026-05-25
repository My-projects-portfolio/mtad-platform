# -*- coding: utf-8 -*-
"""Dispatch entry point for HiRE-Flow in TSB-AD's semi-supervised pool.

TSB-AD's ``run_Semisupervise_AD(model_name, data_train, data_test, **kwargs)``
looks up ``run_<model_name>`` in ``TSB_AD.model_wrapper``'s globals. The
registry hook attaches ``run_HiREFlow`` so this lookup resolves without
modifying TSB-AD core files.
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from .detector import HiREFlow_AD


def run_HiREFlow(
    data_train: np.ndarray,
    data_test: np.ndarray,
    # ── Architecture ──────────────────────────────────────────────────────
    n_blocks: int = 1,
    input_size: int = 1,
    hidden_size: int = 32,
    n_hidden: int = 1,
    # ── Windowing ─────────────────────────────────────────────────────────
    window_size: int = 60,
    stride_size: int = 10,
    batch_size: int = 512,
    # ── Optimisation ──────────────────────────────────────────────────────
    epochs: int = 40,
    lr: float = 2e-3,
    lr_slow_mult: float = 0.1,
    weight_decay: float = 5e-4,
    dropout: float = 0.0,
    batch_norm: bool = False,
    grad_clip_norm: float = 1.0,
    lambda_aux: float = 0.1,
    # ── Scoring ───────────────────────────────────────────────────────────
    score_aggregation: str = "mean",
    # ── HiRE-Flow HPs ─────────────────────────────────────────────────────
    d_g: int = 32,
    d_fast: int = 32,
    d_slow: int = 32,
    d_rank: int = 8,
    # ── HiRE-Flow v2 stability HPs (NEW) ──────────────────────────────────
    rel_dropout: float = 0.1,
    gate_init_bias: float = 2.0,
    # ── Runtime ───────────────────────────────────────────────────────────
    random_state=None,
    device=None,
    verbose: bool = True,
    **_ignored,
) -> np.ndarray:
    """Run HiRE-Flow end-to-end on a single TSB-AD multivariate series.

    Fits on the assumed-clean training portion, scores the full series, and
    returns per-timestep anomaly scores normalised to ``[0, 1]``.

    Parameters
    ----------
    data_train : ndarray (n_train, n_features)
    data_test  : ndarray (n_samples, n_features)  full series to score
    rel_dropout : float, default 0.1
        Dropout rate on the relational-path output before gated fusion.
        Set to 0.0 to disable (ablation).
    gate_init_bias : float, default 4.0
        Initial bias of the gate MLP. With ``gate_init_bias=4.0`` the gate
        starts at sigmoid(4) ≈ 0.98, so the model begins as approximately
        MTGFlow-with-A_prev and learns to open the relational path during
        training. Lowering toward 0.0 reverts to the v1 behavior (gate
        starts mixed ≈ 0.5).
    **kwargs   : see HiREFlow_AD.__init__ for the complete set

    Returns
    -------
    scores : ndarray (n_samples,) in [0, 1], higher = more anomalous
    """
    detector = HiREFlow_AD(
        n_blocks=n_blocks,
        input_size=input_size,
        hidden_size=hidden_size,
        n_hidden=n_hidden,
        window_size=window_size,
        stride_size=stride_size,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        lr_slow_mult=lr_slow_mult,
        weight_decay=weight_decay,
        dropout=dropout,
        batch_norm=batch_norm,
        grad_clip_norm=grad_clip_norm,
        lambda_aux=lambda_aux,
        score_aggregation=score_aggregation,
        d_g=d_g,
        d_fast=d_fast,
        d_slow=d_slow,
        d_rank=d_rank,
        rel_dropout=rel_dropout,
        gate_init_bias=gate_init_bias,
        random_state=random_state,
        device=device,
        verbose=verbose,
    )
    detector.fit(data_train)
    scores = detector.decision_function(data_test)
    return (
        MinMaxScaler(feature_range=(0, 1))
        .fit_transform(scores.reshape(-1, 1))
        .ravel()
    )