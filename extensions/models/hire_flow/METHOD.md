# HiRE-Flow — Method specification for Claude Code

Read this before working on `extensions/models/hire_flow/`. It explains the
method, the design rationale, and the integration contract. The implementation
lives in `extensions/models/hire_flow/model.py` (the `nn.Module`),
`extensions/models/hire_flow/detector.py` (the TSB-AD `BaseDetector` wrapper),
and `extensions/models/hire_flow/runner.py` (the dispatch entry point). Read
`CLAUDE.md` first for the project's wrapper pattern and `extensions/models/mtgflow/`
for the reference implementation pattern that HiRE-Flow follows.

## What HiRE-Flow is

HiRE-Flow (Hierarchical Relational Evolution Normalizing Flow) is a density-based
multivariate time-series anomaly detector. It extends MTGFlow (Zhou et al., AAAI
2023) along a specific axis: explicit modeling of the *temporal dynamics of
inter-sensor coupling* across multiple timescales. The architecture combines
MTGFlow's LSTM + Graph Attention + Masked Autoregressive Flow backbone with a
new relational path consisting of a dual-timescale recurrent state over a
sequence of attention graphs, an auxiliary graph-prediction objective, and a
learned gated fusion of spatial and relational context.

## Why HiRE-Flow exists — the problem with MTGFlow

MTGFlow computes a conditioning vector `h` for its density estimator by
running a GCN over the same window it is about to score, with the GCN's
adjacency matrix `A_t` derived (via attention) from that same window. Formally
the model evaluates `p(x_t | h(x_t, A_t(x_t)))` — the conditioning is a
function of the input itself. When `x_t` is anomalous, `A_t` reflects the
anomaly, and `h` ends up describing the broken state. The flow assigns a
density to `x_t` conditional on context that already encodes the anomalous
behavior, so the score does not drop as much as it should. This is the **input
leakage problem** for graph-conditioned density models.

HiRE-Flow addresses leakage and adds long-range relational context without
modifying the MAF backbone.

## Architecture — four changes from MTGFlow

For each window position `t`, we work with a triplet of consecutive windows
`(x_{t-1}, x_t, x_{t+1})`. Each is shape `(K, L, D)` where K is the number
of sensors, L is the window length, and D=1.

**Change 1 — Lagged graph in the GCN.** The GCN is conditioned on
`A_prev = attention(x_{t-1})`, not on `A_curr = attention(x_t)`. The graph
that conditions the density estimator is computed from a *different* window
than the one being scored, breaking the leakage cycle. `h^spatial = GCN(LSTM(x_t), A_prev)`.

**Change 2 — Dual-timescale relational state.** A small MLP compresses the
upper triangle of `A_curr` into a vector `g_t = GraphEncoder(upper_tri(A_curr))`
of dimension `d_g=32`. Two GRUs consume `g_t`:

- `GRU_fast`: trained at the base learning rate (`2e-3`), captures
  short-range coupling fluctuations.
- `GRU_slow`: trained at `0.1×` the base learning rate (parameter group in the
  optimizer), captures longer-range regime drift.

Both produce 32-dim hidden states `r_fast_t` and `r_slow_t`.

**Change 3 — Auxiliary graph-prediction objective.** A low-rank linear layer
produces `U = GraphPredictor(r_fast_t)` of shape `(K, d_rank=8)`, from which
we reconstruct a predicted next graph:

```
A_hat = softmax( U @ U^T / sqrt(d_rank) , dim=-1 )    # (K, K)
```

The auxiliary loss is the row-wise KL divergence between `A_hat` and
`A_next = attention(x_{t+1})`. This forces `r_fast` to encode information
predictive of future coupling structure, preventing the well-known
representation-collapse failure mode of multi-encoder architectures (where
one branch is sufficient to minimize the primary loss, so the other branch
receives near-zero gradient and learns nothing).

**Change 4 — Gated fusion of spatial and relational context.** A learned
per-channel gate decides how much to trust each path:

```
h^pool   = mean(h^spatial, over K and L)              # (H,)
h^rel    = Combine([r_fast_t ; r_slow_t])             # (H,)
alpha    = sigmoid( GateMLP([h^pool ; r_fast_t ; r_slow_t]) )    # (H,) in [0,1]
h^final  = alpha ⊙ h^spatial + (1 - alpha) ⊙ h^rel    # broadcast over K, L
```

The gate is initialized so that `alpha ≈ 0.88` at the start of training (bias
of `GateMLP` set to `2.0`, weights zeroed). This means HiRE-Flow begins
mostly weighted toward the spatial path (≈ "MTGFlow with A_prev") and *learns*
to open the relational path during training. An earlier default of `4.0`
(sigmoid(4) ≈ 0.98) was tried but left the gate stuck near init — the
relational path stayed dormant for all 40 epochs on SMD 057. `2.0` gives
enough initial relational influence to receive useful gradient while still
biasing toward spatial as a safe starting point.

## Training objective

```
L_flow = -mean( log p(x_t | h^final) )          # MAF negative log-likelihood
L_aux  = KL_row( A_next  ||  A_hat )            # aux_loss_kl in model.py
L      = L_flow + lambda_aux * L_aux            # lambda_aux = 0.1
```

The detector uses `aux_loss_kl`. An MSE alternative (`aux_loss_mse`) exists
in `model.py` for ablations but is not the default — MSE on row-stochastic
(B, K, K) matrices collapses to scale ~1/K², producing near-zero gradient
on the GraphPredictor and leaving the relational path effectively unsupervised.

The MAF backbone, GCN, LSTM, attention, GraphEncoder, both GRUs, GraphPredictor,
Combine, and GateMLP are trained jointly from epoch 1 via a single optimizer
call. The optimizer uses **two parameter groups**: `gru_slow.parameters()` at
`lr × lr_slow_mult` (default `0.1`), and everything else at `lr`.

## Test-time protocol

At the end of training, we save `r_fast_final` and `r_slow_final` (the final
hidden states of both GRUs). At test time:

- `r_fast` is warm-started from `r_fast_final` and continues to update via
  `GRU_fast` on each new window. Its evolving state tracks the test segment's
  coupling fluctuations.
- `r_slow` is **truly frozen**: no GRU call at test time. The vector
  `r_slow_final` is expanded across the batch dimension and used as a constant
  relational context for every test window. This commits to a single regime
  context for the entire test phase.

The anomaly score is `S_t = -log p(x_t | h^final)`. Per-window scores are
folded into per-timestep scores by averaging the scores of all windows
covering each timestep (same logic as `MTGFLOW_AD._windows_to_timesteps`).

## Causality note

Computing `A_hat` and the aux loss requires observing `x_{t+1}`. This means
scoring window `t` at deployment requires the next window's data — a
one-window lookahead, acceptable under the standard TSB-AD offline evaluation
protocol but introducing a one-step latency for real-time deployment. Mention
this in the paper.

## Implementation files

- **`model.py`** — `HiREFlowModel(nn.Module)` composes the architecture.
  Vendored MTGFlow components (`LSTM`, `GNN`, `MAF`, `ScaleDotProductAttention`)
  are imported unmodified from `extensions.models.mtgflow._upstream`. New
  modules: `GraphEncoder`, `GRU_fast`, `GRU_slow`, `GraphPredictor`, `Combine`,
  `GateMLP`. The file also exports `aux_loss_kl` and `aux_loss_mse` helpers.
- **`detector.py`** — `HiREFlow_AD(BaseDetector)` is the TSB-AD wrapper. It
  handles scaling (StandardScaler fit on train only), triplet construction
  from the input array, the training loop with consecutive-batch ordering and
  parameter-group optimizer, state initialization and persistence
  (`r_fast_final_`, `r_slow_final_`), and per-window-to-per-timestep score
  folding. Mirrors `MTGFLOW_AD` from the MTGFLOW integration but with the
  triplet contract.
- **`runner.py`** — `run_HiREFlow(data_train, data_test, **hp)` is the
  dispatch entry point that TSB-AD's `model_wrapper` lookup resolves to. It
  instantiates the detector, calls `fit` then `decision_function`, and
  returns `[0, 1]`-scaled scores.
- **`__init__.py`** — registry hook that attaches `run_HiREFlow` to TSB-AD's
  `Semisupervise_AD_Pool` and `model_wrapper` at import time, following the
  same pattern as `extensions/models/mtgflow/__init__.py`.

## Critical contracts the implementation depends on

**Batches must contain temporally consecutive triplets.** The dual GRU treats
the B items in a mini-batch as a length-B sequence (using `batch_first=False`
with `n_sequences=1`). This means the dataloader must use `shuffle=False` and
must yield contiguous index ranges. Shuffling silently corrupts the
relational state computation. The detector wrapper is responsible for
enforcing this.

**GRU hidden states must be detached between batches.** After
`forward_train` returns `r_fast_new` and `r_slow_new`, the caller must call
`.detach()` on both before passing them to the next batch. Without detachment
the computation graph grows unbounded across batches and the run OOMs.

**Optimizer parameter groups must split `gru_slow`.** The `lr_slow_mult`
hyperparameter only takes effect if the optimizer is constructed with two
parameter groups:

```python
slow_params = list(model.gru_slow.parameters())
slow_ids = {id(p) for p in slow_params}
other_params = [p for p in model.parameters() if id(p) not in slow_ids]
optimizer = torch.optim.Adam(
    [
        {"params": other_params, "lr": lr},
        {"params": slow_params,  "lr": lr * lr_slow_mult},
    ],
    weight_decay=weight_decay,
)
```

A single-group optimizer trains both GRUs at the base learning rate, which
defeats the dual-timescale design.

**Test-time `r_slow` must not call the GRU.** `forward_score` expands the
frozen `r_slow` vector to the batch dimension; it does not run `GRU_slow`.
This is what "truly frozen" means in the test protocol.

## Hyperparameters (pinned defaults)

| Symbol            | Value      | Meaning                                              |
|-------------------|------------|------------------------------------------------------|
| `window_size`     | 60         | Sliding window length L                              |
| `stride_size`     | 10         | Window stride                                        |
| `batch_size`      | 512        | Triplets per mini-batch                              |
| `epochs`          | 40         | Training epochs                                      |
| `lr`              | `2e-3`     | Base learning rate                                   |
| `lr_slow_mult`    | `0.1`      | Slow GRU LR multiplier                               |
| `weight_decay`    | `5e-4`     | Adam weight decay                                    |
| `grad_clip_norm`  | `1.0`      | Gradient clipping max norm                           |
| `lambda_aux`      | `0.1`      | Auxiliary loss weight                                |
| `hidden_size`     | 32         | LSTM/GCN/MAF hidden dim                              |
| `n_blocks`        | 1          | MAF blocks                                           |
| `n_hidden`        | 1          | MADE hidden layers per MAF block                     |
| `d_g`             | 32         | GraphEncoder output dim                              |
| `d_fast`, `d_slow`| 32         | GRU hidden dims                                      |
| `d_rank`          | 8          | Low-rank dim for GraphPredictor                      |
| `rel_dropout`     | 0.1        | Dropout on h_rel before fusion                       |
| `gate_init_bias`  | 2.0        | Bias init so `alpha ≈ 0.88` at start                 |

## What to monitor during training

If results are unexpectedly poor, log these three quantities every ~100 batches:

1. **`alpha.mean()` over training.** With the default `gate_init_bias=2.0`
   alpha starts at ≈ 0.88 and is expected to drift down toward 0.83 over 40
   epochs on SMD 057, with per-channel `alpha_min` reaching ~0.55 (some
   channels going strongly relational) and `alpha_max` staying ~0.96 (others
   staying spatial). If it stays above ~0.95 throughout, the relational path
   is dormant and the multi-timescale story isn't operative. Run
   `scripts/diagnose_hire_flow.py` to inspect.
2. **`L_aux` over epochs.** Should decrease as training proceeds. If it stays
   flat at the initial value, the graph predictor isn't learning and the
   auxiliary objective is broken.
3. **Gradient norms for `gru_slow.parameters()` vs `gru_fast.parameters()`.**
   The slow norm should be smaller than the fast norm (smaller LR + less
   gradient signal). If they're equal, the parameter groups didn't split
   correctly.

## Known caveats

- The relational path can fail to engage (`alpha` stays high), in which case
  HiRE-Flow degenerates to "MTGFlow with A_prev" — still novel, but not the
  full method.
- GRU memory is bounded; the "long-range coupling" claim is sound at scales
  of hundreds of windows, not thousands. State-space models (S4, Mamba) are
  the natural future-work extension.
- Computing the residual `A_next - A_hat` requires next-window observation;
  a one-window scoring latency is inherent to the design.
- TSB-AD's filename-encoded `Train Index` is the train/test cut; assume the
  test segment is temporally contiguous with the training segment for the
  warm-start protocol to make sense.

## How to verify the implementation is wired correctly

Run `extensions/models/hire_flow/smoke_test.py`. It checks: (1) import and
instantiation, (2) forward shape on synthetic data, (3) non-zero parameter
count including all six new modules, (4) `L_aux > 0` and `GraphPredictor`
receives non-zero gradients, (5) end-to-end run on the bundled SMD dataset
produces a finite score array of correct length. All five must pass before
any sweep is started.
