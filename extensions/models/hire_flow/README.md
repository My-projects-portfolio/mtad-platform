# HiRE-Flow integration

**HiRE-Flow** (Hierarchical Relational Evolution Normalizing Flow) for
multivariate time-series anomaly detection, wrapped as a first-class TSB-AD
detector for the mtad-platform.

## Method

HiRE-Flow extends MTGFLOW (Zhou et al., AAAI 2023) in four ways:

1. **Lagged graph in the GCN.**  The GCN conditions on `A_prev = attention(x_{t-1})`
   rather than `A_curr = attention(x_t)`.  This breaks the self-referential loop
   where the graph used to score a window is computed from that same window.

2. **Dual-timescale relational state.**  Two GRUs process a compact graph
   encoding `g_t = GraphEncoder(upper_tri(A_curr))`:
   - `GRU_fast` (full learning rate) captures operational rhythm.
   - `GRU_slow` (learning rate × 0.1) captures slower structural drift.

3. **Auxiliary graph-prediction loss.**  A low-rank `GraphPredictor` predicts
   `A_{t+1}` from `r_fast`.  The auxiliary loss `‖Â_{t+1} − A_{t+1}‖²_F`
   forces the relational state to encode meaningful coupling history and prevents
   representation collapse.

4. **Gated fusion.**  A learned per-channel gate `α = σ(GateMLP([h_pool; r_fast;
   r_slow]))` mixes the spatial context `h_spatial` from the GCN with the
   relational context `h_rel = Combine(r_fast, r_slow)`.  The gate α is
   observable at inference time and serves as a diagnostic for anomaly onset.

All components — the four inherited MTGFLOW modules (attention, LSTM, GCN,
MAF) and the six new modules (GraphEncoder, GRU_fast, GRU_slow,
GraphPredictor, Combine, GateMLP) — are trained jointly in a single
optimisation pass from epoch 1.

## Layout

```
extensions/models/hire_flow/
├── model.py          HiREFlowModel nn.Module
├── detector.py       HiREFlow_AD(BaseDetector) wrapper
├── runner.py         run_HiREFlow(data_train, data_test, **hp)
├── smoke_test.py     5-check correctness gate
├── __init__.py       public API
├── README.md         this file
└── HiREFlow.meta.yaml  model metadata
```

The vendored MTGFLOW upstream (LSTM, GCN, ScaleDotProductAttention, MAF)
is imported read-only from `extensions/models/mtgflow/_upstream/`.
Do not copy or fork those files.

## Registration

Add to `extensions/registry.py`:

```python
_HIREFLOW_DEFAULT_HP = {
    **_MTGFLOW_DEFAULT_HP,
    "lr_slow_mult": 0.1,
    "lambda_aux":   0.1,
    "d_g":          32,
    "d_fast":       32,
    "d_slow":       32,
    "d_rank":       8,
}

def register_hire_flow() -> None:
    from TSB_AD import model_wrapper, HP_list
    from extensions.models.hire_flow import run_HiREFlow
    if "HiREFlow" not in model_wrapper.Semisupervise_AD_Pool:
        model_wrapper.Semisupervise_AD_Pool.append("HiREFlow")
    if not hasattr(model_wrapper, "run_HiREFlow"):
        model_wrapper.run_HiREFlow = run_HiREFlow
    if "HiREFlow" not in HP_list.Optimal_Multi_algo_HP_dict:
        HP_list.Optimal_Multi_algo_HP_dict["HiREFlow"] = dict(_HIREFLOW_DEFAULT_HP)
```

Then call `register_hire_flow()` inside `register_all()`.

## Smoke test

```bash
python -m extensions.models.hire_flow.smoke_test \
    --dataset_dir TSB-AD/Datasets/TSB-AD-M/ \
    --filename 057_SMD_id_1_Facility_tr_4529_1st_4629.csv \
    --epochs 2 \
    --seed 42
```

All five checks should pass in under 5 minutes on CPU.

## Citation

If you use this code, please cite:

- **MTGFLOW (upstream components):**
  Zhou, Q., Chen, J., Liu, H., He, S., & Meng, W. (2023).
  Detecting Multivariate Time Series Anomalies with Zero Known Label.
  *AAAI 2023*.  https://arxiv.org/abs/2208.02108

- **HiRE-Flow:**  TBD (paper in preparation).