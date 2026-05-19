# -*- coding: utf-8 -*-
"""MTGFLOW wrapped as a TSB-AD-compatible BaseDetector.

The model itself (LSTM + Graph Attention + Masked Autoregressive Flow) lives in
``_upstream/`` and is imported unmodified. This file only does I/O plumbing:

    TSB-AD CSV  →  (n_samples, n_features) ndarray  →
        →  sliding windows  →  (batch, K, L, D) tensor  →
            →  MTGFLOW.fit / MTGFLOW.test  →  per-window log_prob  →
                →  per-timestep anomaly score array of length n_samples

References:
    Zhou et al., "Detecting Multivariate Time Series Anomalies with Zero Known
    Label," AAAI 2023. https://arxiv.org/abs/2208.02108
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from TSB_AD.models.base import BaseDetector
from TSB_AD.utils.torch_utility import get_gpu

# The vendored upstream model. DO NOT modify the import; if it fails, run
# extensions/models/mtgflow/_upstream/vendor.sh first.
from ._upstream.MTGFLOW import MTGFLOW as _MTGFLOWModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal sliding-window Dataset
# ---------------------------------------------------------------------------
# Matches the windowing pattern of MTGFLOW's original SWat_dataset, but reads
# from a plain numpy array (TSB-AD's format) instead of a pandas DataFrame.
# ---------------------------------------------------------------------------
class _TSBADWindowDataset(Dataset):
    """Sliding-window view over a 2-D (n_samples, n_features) numpy array.

    Each item has shape (K, L, D) where:
        K = n_features  (number of "sensors" in MTGFLOW's terminology)
        L = window_size
        D = 1           (each sensor produces one scalar per timestep)
    """

    def __init__(self, data: np.ndarray, window_size: int, stride_size: int):
        assert data.ndim == 2, f"expected 2D data, got shape {data.shape}"
        assert len(data) >= window_size, (
            f"series of length {len(data)} is shorter than window_size={window_size}; "
            "cannot extract any windows"
        )
        self.data = data.astype(np.float32)
        self.window_size = window_size
        self.stride_size = stride_size
        # Window starts: 0, stride, 2*stride, ..., last_valid_start
        self.start_idx = np.arange(
            0, len(self.data) - self.window_size + 1, self.stride_size
        )

    def __len__(self) -> int:
        return len(self.start_idx)

    def __getitem__(self, index: int):
        start = self.start_idx[index]
        end = start + self.window_size
        # (L, K) -> (L, K, 1) -> transpose -> (K, L, 1)
        window = self.data[start:end].reshape(self.window_size, -1, 1)
        return torch.from_numpy(window).transpose(0, 1)


# ---------------------------------------------------------------------------
# The detector
# ---------------------------------------------------------------------------
class MTGFLOW_AD(BaseDetector):
    """MTGFLOW anomaly detector, TSB-AD-compatible.

    Parameters
    ----------
    n_blocks : int, default=1
        Number of MAF blocks stacked in the normalizing flow.
    input_size : int, default=1
        Per-sensor input dimension. Each TSB-AD feature column is one sensor
        producing one scalar per timestep, so this should stay at 1.
    hidden_size : int, default=32
        Hidden dim for the LSTM, GNN, and MAF.
    n_hidden : int, default=1
        Number of hidden layers inside each MADE.
    window_size : int, default=60
        Length of the sliding window in timesteps.
    stride_size : int, default=10
        Stride between consecutive windows. Smaller = more overlap, denser
        per-timestep scores at the cost of compute.
    batch_size : int, default=512
        Batches per gradient step.
    epochs : int, default=40
        Number of training epochs. The original repo uses different counts per
        dataset; 40 is a reasonable platform default and is exposed as an HP.
    lr : float, default=2e-3
        Adam learning rate (matches MTGFLOW's main.py default).
    weight_decay : float, default=5e-4
        Adam weight decay (matches MTGFLOW's main.py default).
    dropout : float, default=0.0
        Dropout in the LSTM (test-time uses 0 in the original code).
    batch_norm : bool, default=False
        BatchNorm between MAF blocks (matches MTGFLOW's main.py default).
    grad_clip_norm : float, default=1.0
        Max norm for gradient clipping. Common for normalizing-flow training.
    score_aggregation : str, default='mean'
        How to fold per-window scores back to per-timestep:
        ``'mean'`` — each timestep gets the mean of all window-scores covering
        it (smoothest, recommended for VUS metrics).
        ``'max'`` — each timestep gets the max of its covering windows.
    device : str or None, default=None
        Compute device. If None, uses TSB-AD's get_gpu helper.
    verbose : bool, default=True
        Whether to log per-epoch training loss.

    Attributes
    ----------
    decision_scores_ : ndarray of shape (n_samples,)
        Per-timestep anomaly scores. Higher = more anomalous.
    train_loss_history_ : list of float
        Per-epoch mean training loss. Useful sanity check (should decrease).
    model_ : torch.nn.Module
        The trained MTGFLOW model.
    scaler_ : StandardScaler
        Fit on training data; applied to all data at scoring time.
    """

    def __init__(
        self,
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
        device: Optional[str] = None,
        verbose: bool = True,
    ):
        super().__init__(contamination=0.1)

        self.n_blocks = n_blocks
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_hidden = n_hidden
        self.window_size = window_size
        self.stride_size = stride_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.grad_clip_norm = grad_clip_norm

        if score_aggregation not in ("mean", "max"):
            raise ValueError(
                f"score_aggregation must be 'mean' or 'max', got {score_aggregation!r}"
            )
        self.score_aggregation = score_aggregation

        if device is None:
            self.device = get_gpu(True)
        else:
            self.device = torch.device(device)

        self.verbose = verbose

        # Populated by fit()
        self.model_: Optional[nn.Module] = None
        self.scaler_: Optional[StandardScaler] = None
        self.n_sensor_: Optional[int] = None
        self.train_loss_history_: list[float] = []

    # ------------------------------------------------------------------ fit
    def fit(self, X: np.ndarray, y=None):
        """Train MTGFLOW on the (clean-assumed) training portion of a series.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training series in TSB-AD's row-major format.
        y : ignored
            Present for sklearn-style API compatibility; MTGFLOW is unsupervised.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"expected 2D input, got shape {X.shape}")
        n_samples, n_features = X.shape
        self.n_sensor_ = n_features

        if n_samples < self.window_size:
            raise ValueError(
                f"training series has {n_samples} timesteps but window_size="
                f"{self.window_size}; cannot extract any training windows"
            )

        # MTGFLOW's MAF is numerically sensitive — standardize per feature.
        self.scaler_ = StandardScaler()
        X_norm = self.scaler_.fit_transform(X).astype(np.float32)

        train_loader = DataLoader(
            _TSBADWindowDataset(X_norm, self.window_size, self.stride_size),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )
        if len(train_loader) == 0:
            raise ValueError(
                f"no training windows produced; n_samples={n_samples}, "
                f"window_size={self.window_size}, stride={self.stride_size}"
            )

        # Build the upstream model.
        self.model_ = _MTGFLOWModel(
            n_blocks=self.n_blocks,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            n_hidden=self.n_hidden,
            window_size=self.window_size,
            n_sensor=self.n_sensor_,
            dropout=self.dropout,
            model="MAF",
            batch_norm=self.batch_norm,
        ).to(self.device)

        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        self.train_loss_history_ = []
        self.model_.train()
        for epoch in range(self.epochs):
            epoch_loss_sum = 0.0
            n_batches = 0
            for x in train_loader:
                x = x.to(self.device)  # (B, K, L, D)
                optimizer.zero_grad()
                # model(x) returns scalar = mean log_prob over batch.
                # Loss is negative log-likelihood.
                loss = -self.model_(x)
                # Guard against the rare exploding-gradient case in MAF.
                if not torch.isfinite(loss):
                    logger.warning(
                        "non-finite loss at epoch %d batch %d; skipping batch",
                        epoch,
                        n_batches,
                    )
                    optimizer.zero_grad()
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model_.parameters(), self.grad_clip_norm
                )
                optimizer.step()
                epoch_loss_sum += float(loss.item())
                n_batches += 1

            epoch_mean = epoch_loss_sum / max(n_batches, 1)
            self.train_loss_history_.append(epoch_mean)
            if self.verbose:
                logger.info(
                    "MTGFLOW epoch %d/%d  train_neg_log_lik = %.4f",
                    epoch + 1,
                    self.epochs,
                    epoch_mean,
                )

        return self

    # ----------------------------------------------------------- score
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute per-timestep anomaly scores.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Series to score (typically the full series, including train + test).

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Higher = more anomalous.
        """
        if self.model_ is None or self.scaler_ is None:
            raise RuntimeError("decision_function called before fit")

        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"expected 2D input, got shape {X.shape}")
        n_samples, n_features = X.shape
        if n_features != self.n_sensor_:
            raise ValueError(
                f"feature count mismatch: fit on {self.n_sensor_}, scoring on "
                f"{n_features}"
            )

        # Apply the SAME scaler fit on training data — no refit at test time.
        X_norm = self.scaler_.transform(X).astype(np.float32)

        if n_samples < self.window_size:
            # Series shorter than one window: fall back to a constant score.
            # Use the training loss as a uniform "unknown" score.
            fallback = (
                self.train_loss_history_[-1] if self.train_loss_history_ else 0.0
            )
            return np.full(n_samples, fallback, dtype=np.float32)

        dataset = _TSBADWindowDataset(X_norm, self.window_size, self.stride_size)
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, drop_last=False
        )

        # Per-window log probabilities (higher = more normal).
        log_probs = []
        self.model_.eval()
        with torch.no_grad():
            for x in loader:
                x = x.to(self.device)  # (B, K, L, D)
                lp = self.model_.test(x)  # (B,)
                log_probs.append(lp.detach().cpu().numpy())
        log_probs = np.concatenate(log_probs, axis=0)  # (n_windows,)

        # Anomaly score per window = -log_prob (higher = more anomalous).
        win_scores = -log_probs.astype(np.float64)

        # Upsample per-window scores to per-timestep.
        timestep_scores = self._windows_to_timesteps(
            win_scores=win_scores,
            window_starts=dataset.start_idx,
            window_size=self.window_size,
            n_samples=n_samples,
            aggregation=self.score_aggregation,
        )

        self.decision_scores_ = timestep_scores
        return timestep_scores

    # --------------------------------------------------------- helpers
    @staticmethod
    def _windows_to_timesteps(
        win_scores: np.ndarray,
        window_starts: np.ndarray,
        window_size: int,
        n_samples: int,
        aggregation: str,
    ) -> np.ndarray:
        """Fold per-window anomaly scores back to per-timestep.

        Each timestep ``t`` is covered by zero or more overlapping windows.
        ``aggregation='mean'`` averages them; ``'max'`` takes the maximum.
        Uncovered timesteps at the tail (when ``stride_size > 1``) are
        forward-filled from the last covered value.
        """
        scores = np.zeros(n_samples, dtype=np.float64)
        counts = np.zeros(n_samples, dtype=np.int64)

        if aggregation == "mean":
            for i, s in enumerate(window_starts):
                e = s + window_size
                scores[s:e] += win_scores[i]
                counts[s:e] += 1
            covered = counts > 0
            scores[covered] = scores[covered] / counts[covered]
        else:  # 'max'
            scores.fill(-np.inf)
            for i, s in enumerate(window_starts):
                e = s + window_size
                np.maximum(scores[s:e], win_scores[i], out=scores[s:e])
                counts[s:e] += 1
            covered = counts > 0
            scores[~covered] = -np.inf  # placeholder

        # Forward-fill any tail timesteps that no window covered.
        if not covered.all():
            covered_idx = np.where(covered)[0]
            if len(covered_idx) > 0:
                last = covered_idx.max()
                fill_value = scores[last]
                scores[last + 1 :] = fill_value
            else:
                # Shouldn't happen given the length guard above, but be safe.
                scores[:] = 0.0

        return scores.astype(np.float32)