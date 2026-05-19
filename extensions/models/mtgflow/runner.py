# -*- coding: utf-8 -*-
"""Dispatch entry point for MTGFLOW in TSB-AD's semi-supervised pool.

TSB-AD's ``run_Semisupervise_AD(model_name, data_train, data_test, **kwargs)``
looks up ``run_<model_name>`` in ``TSB_AD.model_wrapper``'s globals. The
registry attaches the function below as ``model_wrapper.run_MTGFLOW`` so this
lookup succeeds.
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from .detector import MTGFLOW_AD


def run_MTGFLOW(
    data_train: np.ndarray,
    data_test: np.ndarray,
    n_blocks: int = 1,
    input_size: int = 1,
    hidden_size: int = 32,
    n_hidden: int = 1,
    window_size: int = 60,
    stride_size: int = 10,
    batch_size: int = 512,
    epochs: int = 40,
    lr: float = 2e-3,
    weight_decay: float = 5e-4,
    dropout: float = 0.0,
    batch_norm: bool = False,
    grad_clip_norm: float = 1.0,
    score_aggregation: str = "mean",
    device=None,
    verbose: bool = True,
    **_ignored,
) -> np.ndarray:
    """Run MTGFLOW end-to-end on a single TSB-AD multivariate series.

    Matches the TSB-AD semi-supervised runner contract: fit on the assumed-clean
    training portion, score the full series, return per-timestep anomaly scores
    in ``[0, 1]`` via a final MinMaxScaler (the same post-processing the other
    semi-supervised TSB-AD runners apply — see ``run_CHARM``).

    Parameters
    ----------
    data_train : ndarray of shape (n_train, n_features)
        Training portion, as sliced by TSB-AD's ``filename.split('_')[-3]`` rule.
    data_test : ndarray of shape (n_samples, n_features)
        Full series to score (n_samples >= n_train; includes the training rows).
    **kwargs
        See ``MTGFLOW_AD.__init__`` for the full set. Unknown keys are ignored
        so the same HP dict can carry settings for several runners.

    Returns
    -------
    scores : ndarray of shape (n_samples,), dtype float
        Per-timestep anomaly scores in [0, 1]. Higher = more anomalous.
    """
    detector = MTGFLOW_AD(
        n_blocks=n_blocks,
        input_size=input_size,
        hidden_size=hidden_size,
        n_hidden=n_hidden,
        window_size=window_size,
        stride_size=stride_size,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        dropout=dropout,
        batch_norm=batch_norm,
        grad_clip_norm=grad_clip_norm,
        score_aggregation=score_aggregation,
        device=device,
        verbose=verbose,
    )
    detector.fit(data_train)
    scores = detector.decision_function(data_test)

    # Match TSB-AD's convention: scores in [0,1] before the metrics call.
    scores = (
        MinMaxScaler(feature_range=(0, 1))
        .fit_transform(scores.reshape(-1, 1))
        .ravel()
    )
    return scores