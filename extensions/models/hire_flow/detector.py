# -*- coding: utf-8 -*-
"""HiRE-Flow wrapped as a TSB-AD-compatible BaseDetector.

Training (single phase)
-----------------------
1. Fit a StandardScaler on the training data.
2. Build consecutive triplet windows (x_{t-1}, x_t, x_{t+1}).
3. For each epoch, reset GRU hidden states to zero and iterate over batches
   in temporal order (shuffle=False):
   a. Run HiREFlowModel.forward_train → (log_prob, r_fast_new, r_slow_new,
      A_hat_upper, A_next_upper).
   b. Compute L_flow = -mean(log_prob) and L_aux = MSE(A_hat, A_next).
   c. Loss = L_flow + lambda_aux * L_aux; backprop; clip grads; step.
   d. Detach r_fast and r_slow (truncated BPTT).
4. Save final GRU states for test warm-start.

Two parameter groups:
  - All parameters EXCEPT gru_slow:  lr = lr_base
  - gru_slow parameters only:        lr = lr_base * lr_slow_mult (default 0.1)

Scoring (offline, single pass in temporal order)
------------------------------------------------
1. Apply the training scaler (no refit).
2. Process consecutive window pairs (x_{t-1}, x_t) via forward_score.
3. r_slow is frozen at training-final value; r_fast updates each window.
4. Upsample per-window scores to per-timestep via mean-aggregation.

References
----------
HiRE-Flow: Hierarchical Relational Evolution Normalizing Flow for Multivariate
Time-Series Anomaly Detection. (paper TBD)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from TSB_AD.models.base import BaseDetector
from TSB_AD.utils.torch_utility import get_gpu

from .model import HiREFlowModel, aux_loss_kl

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class _SingleWindowDataset(Dataset):
    """Sliding-window view over a (n_samples, n_features) numpy array.

    Each item has shape (K, L, D) where K=n_features, L=window_size, D=1.
    """

    def __init__(self, data: np.ndarray, window_size: int, stride: int) -> None:
        assert data.ndim == 2
        self.data = data.astype(np.float32)
        self.window_size = window_size
        self.stride = stride
        self.start_idx = np.arange(
            0, len(data) - window_size + 1, stride, dtype=np.int64
        )

    def __len__(self) -> int:
        return len(self.start_idx)

    def __getitem__(self, i: int) -> torch.Tensor:
        s = self.start_idx[i]
        window = self.data[s : s + self.window_size]   # (L, K)
        # (L, K, 1) → transpose → (K, L, 1)
        return torch.from_numpy(
            window.reshape(self.window_size, -1, 1)
        ).transpose(0, 1)


class _TripletWindowDataset(Dataset):
    """Returns (x_{t-1}, x_t, x_{t+1}) consecutive window triplets.

    Must NOT be shuffled; temporal order is required for the GRU state.
    """

    def __init__(self, base: _SingleWindowDataset) -> None:
        self.base = base
        self.n = max(0, len(base) - 2)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        return self.base[i], self.base[i + 1], self.base[i + 2]


class _PairWindowDataset(Dataset):
    """Returns (x_{t-1}, x_t) pairs for test-time scoring.

    For i=0 the pair is (x_0, x_0) — the first window is used as its own
    predecessor, matching the test bootstrap rule.
    """

    def __init__(self, base: _SingleWindowDataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        prev = self.base[max(0, i - 1)]
        curr = self.base[i]
        return prev, curr


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class HiREFlow_AD(BaseDetector):
    """HiRE-Flow anomaly detector, TSB-AD-compatible.

    Parameters
    ----------
    n_blocks : int, default 1
        Number of MAF blocks in the normalizing flow.
    input_size : int, default 1
        Per-sensor feature dimension (1 for scalar series).
    hidden_size : int, default 32
        Hidden dimension for LSTM, GCN, MAF, Combine, and GateMLP.
    n_hidden : int, default 1
        Hidden layers per MADE block.
    window_size : int, default 60
        Sliding window length.
    stride_size : int, default 10
        Stride between consecutive windows.
    batch_size : int, default 512
        Windows per gradient step (also the GRU sequence length per batch).
    epochs : int, default 40
        Training epochs.
    lr : float, default 2e-3
        Base Adam learning rate (applied to all parameters except gru_slow).
    lr_slow_mult : float, default 0.1
        Learning-rate multiplier for gru_slow parameters.
    weight_decay : float, default 5e-4
        Adam weight decay.
    dropout : float, default 0.0
        LSTM dropout (no-op when num_layers=1).
    batch_norm : bool, default False
        Batch normalisation in MAF.
    grad_clip_norm : float, default 1.0
        Max gradient norm for clipping.
    lambda_aux : float, default 0.1
        Weight for the auxiliary graph-prediction loss.
    score_aggregation : str, default "mean"
        "mean" or "max" aggregation when folding window scores to timesteps.
    d_g : int, default 32
        GraphEncoder output dimension.
    d_fast : int, default 32
        GRU_fast hidden dimension.
    d_slow : int, default 32
        GRU_slow hidden dimension.
    d_rank : int, default 8
        Low-rank factor for GraphPredictor.
    random_state : int or None, default None
        Seed for reproducibility.
    device : str or None, default None
        Torch device string; auto-selected if None.
    verbose : bool, default True
        Log per-epoch training loss.

    Attributes
    ----------
    decision_scores_ : ndarray (n_samples,)
        Set by decision_function; higher = more anomalous.
    train_loss_history_ : list of float
        Mean total loss per epoch.
    model_ : HiREFlowModel or None
        The trained model.
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
        lr_slow_mult: float = 0.1,
        weight_decay: float = 5e-4,
        dropout: float = 0.0,
        batch_norm: bool = False,
        grad_clip_norm: float = 1.0,
        lambda_aux: float = 0.1,
        score_aggregation: str = "mean",
        d_g: int = 32,
        d_fast: int = 32,
        d_slow: int = 32,
        d_rank: int = 8,
        rel_dropout: float = 0.1,
        gate_init_bias: float = 2.0,
        random_state: Optional[int] = None,
        device: Optional[str] = None,
        verbose: bool = True,
    ) -> None:
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
        self.lr_slow_mult = lr_slow_mult
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.grad_clip_norm = grad_clip_norm
        self.lambda_aux = lambda_aux

        if score_aggregation not in ("mean", "max"):
            raise ValueError(
                f"score_aggregation must be 'mean' or 'max', "
                f"got {score_aggregation!r}"
            )
        self.score_aggregation = score_aggregation

        self.d_g = d_g
        self.d_fast = d_fast
        self.d_slow = d_slow
        self.d_rank = d_rank
        self.rel_dropout = rel_dropout
        self.gate_init_bias = gate_init_bias
        self.random_state = random_state

        self.device = (
            torch.device(device) if device is not None else get_gpu(True)
        )
        self.verbose = verbose

        # Populated by fit()
        self.model_: Optional[HiREFlowModel] = None
        self.scaler_: Optional[StandardScaler] = None
        self.n_sensor_: Optional[int] = None
        self.train_loss_history_: list[float] = []
        self._r_fast_final: Optional[torch.Tensor] = None
        self._r_slow_final: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ fit

    def fit(self, X: np.ndarray, y=None) -> "HiREFlow_AD":
        """Train HiRE-Flow on the assumed-clean training segment.

        Parameters
        ----------
        X : ndarray (n_samples, n_features)
        y : ignored

        Returns
        -------
        self
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_state)

        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"expected 2D input, got shape {X.shape}")
        n_samples, n_features = X.shape
        self.n_sensor_ = n_features

        if n_samples < self.window_size:
            raise ValueError(
                f"series has {n_samples} timesteps but window_size="
                f"{self.window_size}"
            )

        # Standardize on training data only
        self.scaler_ = StandardScaler()
        X_norm = self.scaler_.fit_transform(X).astype(np.float32)

        # Build dataset — need at least 3 windows for one triplet
        base_ds = _SingleWindowDataset(X_norm, self.window_size, self.stride_size)
        if len(base_ds) < 3:
            raise ValueError(
                f"need ≥3 windows to form one triplet; got {len(base_ds)}"
            )
        triplet_ds = _TripletWindowDataset(base_ds)
        loader = DataLoader(
            triplet_ds,
            batch_size=self.batch_size,
            shuffle=False,          # temporal order mandatory for GRU
            drop_last=False,
        )

        # Build model
        self.model_ = HiREFlowModel(
            n_blocks=self.n_blocks,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            n_hidden=self.n_hidden,
            window_size=self.window_size,
            n_sensor=n_features,
            dropout=self.dropout,
            batch_norm=self.batch_norm,
            d_g=self.d_g,
            d_fast=self.d_fast,
            d_slow=self.d_slow,
            d_rank=self.d_rank,
            rel_dropout=self.rel_dropout,
            gate_init_bias=self.gate_init_bias,
        ).to(self.device)

        # Two-parameter-group optimizer: gru_slow at reduced LR
        slow_params = list(self.model_.gru_slow.parameters())
        slow_ids = {id(p) for p in slow_params}
        base_params = [
            p for p in self.model_.parameters() if id(p) not in slow_ids
        ]
        optimizer = torch.optim.Adam(
            [
                {
                    "params": base_params,
                    "lr": self.lr,
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": slow_params,
                    "lr": self.lr * self.lr_slow_mult,
                    "weight_decay": self.weight_decay,
                },
            ]
        )

        self.train_loss_history_ = []
        self.model_.train()

        
        r_fast = torch.zeros(1, 1, self.d_fast, device=self.device)
          # initialise once
        r_slow = torch.zeros(1, 1, self.d_slow, device=self.device)
        for epoch in range(self.epochs):
            # Reset GRU states at epoch start (hidden = 0)
            #r_fast = torch.zeros(1, 1, self.d_fast, device=self.device)
            #r_slow = torch.zeros(1, 1, self.d_slow, device=self.device)

            epoch_loss_sum = 0.0
            n_batches = 0

            for x_prev, x_curr, x_next in loader:
                x_prev = x_prev.to(self.device)
                x_curr = x_curr.to(self.device)
                x_next = x_next.to(self.device)

                optimizer.zero_grad()

                log_prob, r_fast_new, r_slow_new, A_hat_upper, A_next_upper = (
                    self.model_.forward_train(
                        x_prev, x_curr, x_next, r_fast, r_slow
                    )
                )

                L_flow = -log_prob.mean()
                L_aux = aux_loss_kl(A_hat_upper, A_next_upper.detach())
                loss = L_flow + self.lambda_aux * L_aux

                if not torch.isfinite(loss):
                    logger.warning(
                        "HiREFlow non-finite loss at epoch %d; skipping batch",
                        epoch + 1,
                    )
                    optimizer.zero_grad()
                    # Still advance GRU state to maintain temporal coherence
                    r_fast = r_fast_new.detach()
                    r_slow = r_slow_new.detach()
                    continue

                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model_.parameters(), self.grad_clip_norm
                )
                optimizer.step()

                # Truncated BPTT: carry state value but stop gradient flow
                r_fast = r_fast_new.detach()
                r_slow = r_slow_new.detach()

                epoch_loss_sum += float(loss.item())
                n_batches += 1

            epoch_mean = epoch_loss_sum / max(n_batches, 1)
            self.train_loss_history_.append(epoch_mean)

            if self.verbose:
                logger.info(
                    "HiREFlow epoch %d/%d  loss=%.4f",
                    epoch + 1,
                    self.epochs,
                    epoch_mean,
                )

        # Save final GRU states for test warm-start
        self._r_fast_final = r_fast.detach().cpu()
        self._r_slow_final = r_slow.detach().cpu()

        self.model_.eval()
        return self

    # -------------------------------------------------------- decision_function

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute per-timestep anomaly scores.

        Processes the full series in temporal order with GRU warm-started from
        training.  r_slow is frozen at training-final value; r_fast updates
        each window.

        Parameters
        ----------
        X : ndarray (n_samples, n_features)

        Returns
        -------
        scores : ndarray (n_samples,) — higher = more anomalous
        """
        if self.model_ is None or self.scaler_ is None:
            raise RuntimeError("decision_function called before fit")

        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"expected 2D input, got shape {X.shape}")
        n_samples, n_features = X.shape

        if n_samples < self.window_size:
            fallback = (
                self.train_loss_history_[-1] if self.train_loss_history_ else 0.0
            )
            return np.full(n_samples, fallback, dtype=np.float32)

        X_norm = self.scaler_.transform(X).astype(np.float32)

        base_ds = _SingleWindowDataset(X_norm, self.window_size, self.stride_size)
        pair_ds = _PairWindowDataset(base_ds)
        loader = DataLoader(
            pair_ds,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )

        # Warm-start from final training states
        r_fast = self._r_fast_final.to(self.device)
        r_slow = self._r_slow_final.to(self.device)  # frozen throughout scoring

        win_scores: list[float] = []
        self.model_.eval()

        with torch.no_grad():
            for x_prev, x_curr in loader:
                x_prev = x_prev.to(self.device)
                x_curr = x_curr.to(self.device)
                B = x_curr.shape[0]

                log_prob, r_fast_new = self.model_.forward_score(
                    x_prev, x_curr, r_fast, r_slow
                )
                # r_slow is NOT updated (frozen) — only r_fast advances
                r_fast = r_fast_new

                win_scores.extend((-log_prob).tolist())

        win_scores_np = np.array(win_scores, dtype=np.float64)

        timestep_scores = self._windows_to_timesteps(
            win_scores=win_scores_np,
            window_starts=base_ds.start_idx,
            window_size=self.window_size,
            n_samples=n_samples,
            aggregation=self.score_aggregation,
        )
        self.decision_scores_ = timestep_scores
        return timestep_scores

    # --------------------------------------------------------------- helpers

    @staticmethod
    def _windows_to_timesteps(
        win_scores: np.ndarray,
        window_starts: np.ndarray,
        window_size: int,
        n_samples: int,
        aggregation: str,
    ) -> np.ndarray:
        """Fold per-window scores back to per-timestep scores.

        Each timestep is covered by one or more overlapping windows.
        Uncovered tail timesteps are forward-filled from the last covered
        value.

        Parameters
        ----------
        win_scores : (n_windows,) anomaly score per window
        window_starts : (n_windows,) start index of each window
        window_size : int
        n_samples : int
        aggregation : "mean" or "max"

        Returns
        -------
        scores : (n_samples,) dtype float32
        """
        scores = np.zeros(n_samples, dtype=np.float64)
        counts = np.zeros(n_samples, dtype=np.int64)

        if aggregation == "mean":
            for i, s in enumerate(window_starts):
                e = s + window_size
                scores[s:e] += win_scores[i]
                counts[s:e] += 1
            covered = counts > 0
            scores[covered] /= counts[covered]
        else:  # max
            scores.fill(-np.inf)
            for i, s in enumerate(window_starts):
                e = s + window_size
                np.maximum(scores[s:e], win_scores[i], out=scores[s:e])
                counts[s:e] += 1
            covered = counts > 0
            scores[~covered] = 0.0

        # Forward-fill any uncovered tail timesteps
        if not np.all(covered):
            last_valid = scores[covered][-1] if covered.any() else 0.0
            scores[~covered] = last_valid

        return scores.astype(np.float32)