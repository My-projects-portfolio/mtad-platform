# -*- coding: utf-8 -*-
"""HiRE-Flow neural network model — v2 with stability improvements.

Changes from v1 (in order of expected impact):

1. **Gate bias initialization** (`gate_init_bias=4.0`): the gate now starts
   with alpha ≈ 0.98 so the model behaves like MTGFlow-with-A_prev at init,
   then learns to open the relational path. This is the most important
   change for matching MTGFlow's training trajectory.

2. **Bug fix — test-time r_slow**: r_slow is truly frozen at test time
   (no GRU call; expanded to batch dimension).

3. **Numerical stability**: A_hat uses scaled softmax (UU^T / sqrt(d_rank)).

4. **Orthogonal GRU init** for recurrent weight matrices.

5. **Aux loss flexibility**: forward_train now returns the full (B, K, K)
   A_hat and A_next matrices so the runner can choose KL (recommended) or
   MSE (original) via the helper functions at the bottom of this file.

6. **Relational-path dropout** (`rel_dropout=0.1`): light regularization
   on h_rel before mixing.

7. **Shape assertions** in _upper_tri.

No change to the architecture's external contract: forward_train and
forward_score have the same signatures except that forward_train now
returns A_hat and A_next as full matrices instead of upper-triangle
vectors. The runner needs a one-line update (call aux_loss_kl or
aux_loss_mse on the returned matrices).
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from extensions.models.mtgflow._upstream.MTGFLOW import (
    GNN,
    ScaleDotProductAttention,
)
from extensions.models.mtgflow._upstream.NF import MAF


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _upper_tri(A: torch.Tensor) -> torch.Tensor:
    """Extract upper-triangle entries from a batch of square matrices.

    Parameters
    ----------
    A : Tensor of shape (B, K, K)

    Returns
    -------
    Tensor of shape (B, K*(K-1)//2)
    """
    assert A.ndim == 3, f"expected (B, K, K), got shape {A.shape}"
    assert A.shape[-1] == A.shape[-2], f"expected square, got {A.shape}"
    K = A.shape[-1]
    idx = torch.triu_indices(K, K, offset=1, device=A.device)
    return A[:, idx[0], idx[1]]


def _init_gru_orthogonal(gru: nn.GRU) -> None:
    """Apply orthogonal init to GRU recurrent weights, Xavier to input weights.

    PyTorch's default GRU initialization is uniform, which is unstable for
    deep recurrent computation graphs. Orthogonal recurrent weights preserve
    the norm of the hidden state across timesteps, which matters here because
    we run the GRU over an entire mini-batch as a single sequence.
    """
    for name, param in gru.named_parameters():
        if "weight_ih" in name:
            nn.init.xavier_uniform_(param)
        elif "weight_hh" in name:
            # GRU has 3 gates concatenated; init each block orthogonally.
            hidden_size = param.shape[1]
            for i in range(0, param.shape[0], hidden_size):
                nn.init.orthogonal_(param[i : i + hidden_size])
        elif "bias" in name:
            nn.init.zeros_(param)


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class GraphEncoder(nn.Module):
    """Compress the upper-triangle of a coupling graph to a compact vector."""

    def __init__(self, graph_dim: int, d_g: int = 32) -> None:
        super().__init__()
        mid = max(graph_dim // 2, d_g)
        self.mlp = nn.Sequential(
            nn.Linear(graph_dim, mid),
            nn.Tanh(),
            nn.Linear(mid, d_g),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class HiREFlowModel(nn.Module):
    """HiRE-Flow: Hierarchical Relational Evolution Normalizing Flow.

    See module docstring for the v2 change list.
    """

    def __init__(
        self,
        n_blocks: int,
        input_size: int,
        hidden_size: int,
        n_hidden: int,
        window_size: int,
        n_sensor: int,
        dropout: float = 0.0,
        batch_norm: bool = False,
        d_g: int = 32,
        d_fast: int = 32,
        d_slow: int = 32,
        d_rank: int = 8,
        rel_dropout: float = 0.1,
        gate_init_bias: float = 4.0,
    ) -> None:
        super().__init__()

        self.n_sensor = n_sensor
        self.hidden_size = hidden_size
        self.d_fast = d_fast
        self.d_slow = d_slow
        self.d_rank = d_rank

        # ── Reused MTGFLOW components ──────────────────────────────────────
        self.attention = ScaleDotProductAttention(window_size * input_size)
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            dropout=dropout,
        )
        self.gcn = GNN(input_size=hidden_size, hidden_size=hidden_size)
        self.nf = MAF(
            n_blocks,
            n_sensor,
            input_size,
            hidden_size,
            n_hidden,
            cond_label_size=hidden_size,
            batch_norm=batch_norm,
            activation="tanh",
        )

        # ── New HiRE-Flow modules ──────────────────────────────────────────
        graph_dim = n_sensor * (n_sensor - 1) // 2

        self.graph_encoder = GraphEncoder(graph_dim, d_g)

        self.gru_fast = nn.GRU(input_size=d_g, hidden_size=d_fast, batch_first=False)
        self.gru_slow = nn.GRU(input_size=d_g, hidden_size=d_slow, batch_first=False)
        _init_gru_orthogonal(self.gru_fast)
        _init_gru_orthogonal(self.gru_slow)

        self.graph_predictor = nn.Linear(d_fast, n_sensor * d_rank)

        self.combine = nn.Sequential(
            nn.Linear(d_fast + d_slow, hidden_size),
            nn.Tanh(),
        )
        self.rel_dropout = nn.Dropout(rel_dropout)

        self.gate_mlp = nn.Linear(hidden_size + d_fast + d_slow, hidden_size)
        # Critical: initialize gate bias high so alpha ≈ sigmoid(4) ≈ 0.98 at
        # start. The model begins as approximately MTGFlow-with-A_prev; the
        # relational path can only ADD information as training proceeds.
        nn.init.zeros_(self.gate_mlp.weight)
        nn.init.constant_(self.gate_mlp.bias, gate_init_bias)

    # ── Forward: training ──────────────────────────────────────────────────

    def forward_train(
        self,
        x_prev: torch.Tensor,
        x_curr: torch.Tensor,
        x_next: torch.Tensor,
        r_fast: torch.Tensor,
        r_slow: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training forward pass over a mini-batch of consecutive triplets.

        Parameters
        ----------
        x_prev, x_curr, x_next : Tensor (B, K, L, D)
            The triplet of consecutive windows. The mini-batch MUST be in
            temporal order (no shuffling) — the dual GRU treats the B items
            as a length-B sequence.
        r_fast : Tensor (1, 1, d_fast)
            GRU_fast hidden state, carried over from the previous batch.
        r_slow : Tensor (1, 1, d_slow)
            GRU_slow hidden state, carried over from the previous batch.

        Returns
        -------
        log_prob    : (B,)        per-window mean log-prob (higher = more normal)
        r_fast_new  : (1,1,d_fast) updated fast state — caller MUST detach
        r_slow_new  : (1,1,d_slow) updated slow state — caller MUST detach
        A_hat       : (B, K, K)   predicted next graph (full matrix)
        A_next      : (B, K, K)   actual next graph (full matrix)

        Notes
        -----
        A_hat and A_next are returned as full (B, K, K) matrices so the runner
        can compute the aux loss using either ``aux_loss_kl`` (recommended)
        or ``aux_loss_mse`` (original formulation).
        """
        B, K, L, D = x_curr.shape

        # ── 1. Graph construction ──────────────────────────────────────────
        A_prev, _ = self.attention(x_prev)
        A_curr, _ = self.attention(x_curr)
        A_next, _ = self.attention(x_next)

        # ── 2. Relational state update ─────────────────────────────────────
        g_t = self.graph_encoder(_upper_tri(A_curr))
        g_seq = g_t.unsqueeze(1)
        out_fast, r_fast_new = self.gru_fast(g_seq, r_fast)
        out_slow, r_slow_new = self.gru_slow(g_seq, r_slow)
        r_fast_t = out_fast.squeeze(1)
        r_slow_t = out_slow.squeeze(1)

        # ── 3. Auxiliary graph prediction (low-rank, scaled softmax) ───────
        U = self.graph_predictor(r_fast_t).reshape(B, K, self.d_rank)
        scores = (U @ U.transpose(-1, -2)) / math.sqrt(self.d_rank)
        A_hat = F.softmax(scores, dim=-1)

        # ── 4. Value branch: LSTM + GCN with CLEAN previous graph ──────────
        x_bkld = x_curr.reshape(B * K, L, D)
        h_temp, _ = self.rnn(x_bkld)
        h_temp = h_temp.reshape(B, K, L, self.hidden_size)
        h_spatial = self.gcn(h_temp, A_prev)

        # ── 5. Gated fusion ────────────────────────────────────────────────
        h_pool = h_spatial.mean(dim=[1, 2])
        h_rel = self.combine(torch.cat([r_fast_t, r_slow_t], dim=-1))
        h_rel = self.rel_dropout(h_rel)
        alpha = torch.sigmoid(
            self.gate_mlp(torch.cat([h_pool, r_fast_t, r_slow_t], dim=-1))
        )
        alpha_exp = alpha.unsqueeze(1).unsqueeze(1)
        h_rel_exp = h_rel.unsqueeze(1).unsqueeze(1)
        h_final = alpha_exp * h_spatial + (1.0 - alpha_exp) * h_rel_exp

        # ── 6. Density estimation via MAF ──────────────────────────────────
        h_flat = h_final.reshape(-1, self.hidden_size)
        x_flat = x_bkld.reshape(-1, D)
        log_prob = self.nf.log_prob(x_flat, K, L, h_flat)
        log_prob = log_prob.reshape(B, -1).mean(dim=1)

        return log_prob, r_fast_new, r_slow_new, A_hat, A_next

    # ── Forward: scoring ───────────────────────────────────────────────────

    def forward_score(
        self,
        x_prev: torch.Tensor,
        x_curr: torch.Tensor,
        r_fast: torch.Tensor,
        r_slow: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scoring forward pass (test time — no gradient, no A_next).

        Protocol differences vs. training:
        - r_fast updates each call and is returned (warm-started across batches).
        - r_slow is TRULY frozen: no GRU call. The training-final r_slow vector
          is expanded across the batch dimension and used as a constant
          relational context for every window in the test phase.

        Parameters
        ----------
        x_prev, x_curr : Tensor (B, K, L, D)
        r_fast : Tensor (1, 1, d_fast)  warm-started fast state
        r_slow : Tensor (1, 1, d_slow)  frozen slow state (never updated)

        Returns
        -------
        log_prob   : (B,)
        r_fast_new : (1, 1, d_fast)  for next batch
        """
        B, K, L, D = x_curr.shape

        A_prev, _ = self.attention(x_prev)
        A_curr, _ = self.attention(x_curr)

        g_t = self.graph_encoder(_upper_tri(A_curr))
        g_seq = g_t.unsqueeze(1)

        out_fast, r_fast_new = self.gru_fast(g_seq, r_fast)
        r_fast_t = out_fast.squeeze(1)

        # r_slow truly frozen: no GRU call. Broadcast the fixed state to (B, d_slow).
        r_slow_t = r_slow.squeeze(0).squeeze(0).expand(B, -1).contiguous()

        x_bkld = x_curr.reshape(B * K, L, D)
        h_temp, _ = self.rnn(x_bkld)
        h_temp = h_temp.reshape(B, K, L, self.hidden_size)
        h_spatial = self.gcn(h_temp, A_prev)

        h_pool = h_spatial.mean(dim=[1, 2])
        h_rel = self.combine(torch.cat([r_fast_t, r_slow_t], dim=-1))
        # rel_dropout is a no-op in eval mode (caller must set model.eval()).
        alpha = torch.sigmoid(
            self.gate_mlp(torch.cat([h_pool, r_fast_t, r_slow_t], dim=-1))
        )
        alpha_exp = alpha.unsqueeze(1).unsqueeze(1)
        h_rel_exp = h_rel.unsqueeze(1).unsqueeze(1)
        h_final = alpha_exp * h_spatial + (1.0 - alpha_exp) * h_rel_exp

        h_flat = h_final.reshape(-1, self.hidden_size)
        x_flat = x_bkld.reshape(-1, D)
        log_prob = self.nf.log_prob(x_flat, K, L, h_flat)
        log_prob = log_prob.reshape(B, -1).mean(dim=1)

        return log_prob, r_fast_new


# ---------------------------------------------------------------------------
# Auxiliary loss functions (runner chooses one)
# ---------------------------------------------------------------------------

def aux_loss_kl(A_hat: torch.Tensor, A_next: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Row-wise KL divergence between predicted and target graphs.

    Both inputs are (B, K, K) with row-stochastic last dim. This is the
    natural loss for matching probability distributions and is recommended
    over MSE for the auxiliary objective.

    Returns scalar averaged over batch and rows.
    """
    log_hat = torch.log(A_hat + eps)
    log_target = torch.log(A_next + eps)
    # KL(target || hat) = sum_j target[j] * (log target[j] - log hat[j])
    kl_per_row = (A_next * (log_target - log_hat)).sum(dim=-1)  # (B, K)
    return kl_per_row.mean()


def aux_loss_mse(A_hat: torch.Tensor, A_next: torch.Tensor) -> torch.Tensor:
    """MSE on upper triangles (original v1 formulation, kept for ablation)."""
    return F.mse_loss(_upper_tri(A_hat), _upper_tri(A_next))