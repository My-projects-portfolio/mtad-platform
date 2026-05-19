# MTGFLOW integration

[`MTGFLOW`](https://arxiv.org/abs/2208.02108) (Zhou et al., AAAI 2023) wrapped as
a first-class TSB-AD detector for the mtad-platform.

## Layout

```
extensions/models/mtgflow/
├── _upstream/
│   ├── MTGFLOW.py            # vendored from github.com/zqhang/MTGFLOW (read-only)
│   ├── NF.py                 # vendored from github.com/zqhang/MTGFLOW (read-only)
│   ├── ATTRIBUTION.md        # provenance + the two mechanical diffs applied
│   └── vendor.sh             # automated copy script
├── detector.py               # MTGFLOW_AD(BaseDetector) — the wrapper
├── runner.py                 # run_MTGFLOW(data_train, data_test, **hp)
├── smoke_test.py             # 5-check correctness gate
├── __init__.py
└── README.md                 # this file
```

## Design

The MTGFLOW math (LSTM + Graph Attention + Masked Autoregressive Flow) lives in
`_upstream/` and is **not modified**. The wrapper in `detector.py` adapts it to
TSB-AD's `BaseDetector` contract:

- **Input:** TSB-AD's `(n_samples, n_features)` ndarray (the format every other
  TSB-AD detector receives).
- **Inside:** the wrapper standardizes the features, builds sliding windows of
  shape `(K, L, D) = (n_features, window_size, 1)` matching MTGFLOW's native
  data contract, runs the upstream model unchanged, and folds per-window log
  probabilities back to per-timestep anomaly scores.
- **Output:** an `(n_samples,)` score array, scaled to `[0, 1]` by the runner —
  same post-processing as TSB-AD's other semi-supervised models.

The only "modifications" to MTGFLOW are scaffolding: the data loader (TSB-AD CSV
instead of SWaT-native CSV), the train/test split (TSB-AD's filename-encoded cut
instead of `0.6 * len(series)`), and the per-window-to-per-timestep score
upsampling that TSB-AD's metrics require. None of these touch the model itself.

## Setup (one-time)

1. Clone the MTGFLOW repository somewhere local:

   ```bash
   git clone https://github.com/zqhang/MTGFLOW.git /tmp/mtgflow-upstream
   ```

2. Vendor the model files:

   ```bash
   cd extensions/models/mtgflow/_upstream
   ./vendor.sh /tmp/mtgflow-upstream
   ```

   This copies `MTGFLOW.py` and `NF.py` into `_upstream/` and applies the two
   mechanical patches documented in `ATTRIBUTION.md` (relative-import rewrite
   plus a strip of a dead `turtle` import that breaks on Pythons without
   tkinter).

3. Verify the import works (TSB-AD venv must be active):

   ```bash
   python -c "from extensions.models.mtgflow import MTGFLOW_AD; print('ok')"
   ```

## Smoke test

Before running the full sweep, run the smoke test on one small dataset:

```bash
python -m extensions.models.mtgflow.smoke_test \
    --dataset_dir TSB-AD/Datasets/TSB-AD-M/ \
    --filename 057_SMD_id_1_Facility_tr_4529_1st_4629.csv \
    --epochs 5 \
    --seed 42
```

It checks five conditions: pool registration, score shape/dtype/NaN/Inf, scores
within `[0, 1]`, AUC-ROC above 0.5, AUC-PR above the anomaly-ratio baseline, and
training loss decreasing across epochs. If all five pass, the integration is
sound. If any fail, fix the wrapper before scaling up.

## Use in the full sweep

`run_baseline.py` only needs one new line at the top:

```python
from extensions.registry import register_all
register_all()
```

After that, `--model MTGFLOW` works exactly like `--model IForest` — same
runner, same JSON sidecars, same resume logic.

The 10-dataset × 5-seed sweep:

```bash
for seed in 13 17 42 1337 2024; do
  python extensions/runners/run_baseline.py \
      --model MTGFLOW \
      --file_list extensions/configs/dev_subset.csv \
      --dataset_dir TSB-AD/Datasets/TSB-AD-M/ \
      --output_dir results/ \
      --seed "$seed"
done
```

(or whatever invocation pattern `run_baseline.py` already supports — adjust the
flag names to match.)

## Hyperparameters

Defaults match MTGFLOW's `main.py` argparse defaults, with one platform-default
addition (`epochs=40`) since the original repo varies epoch count per dataset:

| HP                  | Default | Source                          |
|---------------------|---------|---------------------------------|
| `n_blocks`          | 1       | `main.py`                       |
| `hidden_size`       | 32      | `main.py`                       |
| `n_hidden`          | 1       | `main.py`                       |
| `input_size`        | 1       | `main.py` (per-sensor scalar)   |
| `window_size`       | 60      | `main.py`                       |
| `stride_size`       | 10      | `main.py`                       |
| `batch_size`        | 512     | `main.py`                       |
| `lr`                | 2e-3    | `main.py`                       |
| `weight_decay`      | 5e-4    | `main.py`                       |
| `dropout`           | 0.0     | `main.py` (test-time)           |
| `batch_norm`        | False   | `main.py`                       |
| `epochs`            | 40      | platform default (not in main.py) |
| `grad_clip_norm`    | 1.0     | standard for flow-based models  |
| `score_aggregation` | mean    | wrapper-specific (mean smoothes) |

Override any of these by editing `_MTGFLOW_DEFAULT_HP` in `extensions/registry.py`
or by passing `**kwargs` through TSB-AD's HP machinery.

## Why the numbers won't exactly match the MTGFLOW paper

Same reason TSB-AD's own leaderboard differs from each original paper:

1. **Data pipeline:** TSB-AD applies its own preprocessing (no normalization in
   the framework call; we apply StandardScaler inside the wrapper to keep the
   MAF numerically stable, matching MTGFLOW's own preprocessing).
2. **Train/test split:** filename-encoded `tr_NNNN` cutoff instead of
   `0.6 * len`.
3. **Seeds:** the platform uses `[13, 17, 42, 1337, 2024]`; the original paper
   uses 15–19.

This is intentional. The platform's job is fair cross-model comparison on a
common harness, not paper reproduction. The model itself is byte-identical to
the AAAI release — verify any time with `diff` against upstream.

## Citation

If you publish results that use this code, cite the MTGFLOW paper (BibTeX in
`_upstream/ATTRIBUTION.md`).