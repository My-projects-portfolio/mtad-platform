# NOTES.md — TSB-AD Investigation

Investigation of [TheDatumOrg/TSB-AD](https://github.com/TheDatumOrg/TSB-AD), the official repo for *"The Elephant in the Room: Towards A Reliable Time-Series Anomaly Detection Benchmark"* (Liu & Paparrizos, NeurIPS 2024 D&B Track). All paths below are relative to `TSB-AD/` unless noted; line numbers are pinned to commit at clone time.

---

## 1. Folder structure (2 levels)

```
TSB-AD/
├── TSB_AD/                       # Main installable package
│   ├── models/                   # 40+ detector implementations + base class
│   ├── evaluation/               # Metrics (basic, VUS, affiliation/, vus/)
│   ├── utils/                    # slidingWindows, torch_utility, utility
│   ├── HP_list.py                # HP search spaces + tuned optima (dicts)
│   ├── model_wrapper.py          # run_Unsupervise_AD / run_Semisupervise_AD dispatchers
│   ├── main.py                   # Single-file univariate runner (entry: `python -m TSB_AD.main`)
│   └── __init__.py
├── benchmark_exp/                # Sweep scripts
│   ├── Run_Detector_U.py         # Univariate sweep over a file list
│   ├── Run_Detector_M.py         # Multivariate sweep (ours of interest)
│   ├── HP_Tuning_U.py            # HP grid search, univariate
│   ├── HP_Tuning_M.py            # HP grid search, multivariate
│   └── Run_Custom_Detector.py    # Template for new detectors (PR target)
├── Datasets/
│   └── File_List/                # *.csv lists of dataset filenames per split
├── tutorials/                    # Notebook walkthroughs
├── docs/                         # Sphinx docs
├── assets/                       # Figures
├── README.md
├── requirements.txt
├── setup.py
└── pyproject.toml
```

The actual time-series data is **not in the repo** — `Datasets/TSB-AD-U/` and `Datasets/TSB-AD-M/` must be downloaded from `https://www.thedatum.org/datasets/TSB-AD-{U,M}.zip`.

---

## 2. Base detector interface

`TSB_AD/models/base.py:21` — `class BaseDetector(metaclass=abc.ABCMeta)`. Closely mirrors PyOD's `BaseDetector`, which is unsurprising (the README acknowledges PyOD).

```python
class BaseDetector(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, contamination=0.1): ...               # base.py:51
    @abc.abstractmethod
    def fit(self, X, y=None): ...                            # base.py:64  -> returns self
    @abc.abstractmethod
    def decision_function(self, X): ...                      # base.py:83  -> ndarray (n_samples,)
    def predict(self, X, return_confidence=False): ...       # base.py:133 -> binary labels via threshold_
```

After `fit`, three attributes are expected:
- `decision_scores_`: `np.ndarray` of shape `(n_samples,)`, higher = more anomalous
- `threshold_`: float, derived from `contamination` percentile of `decision_scores_`
- `labels_`: 0/1 array from thresholding

`_process_decision_scores()` (inherited helper) sets `threshold_` and `labels_` from `decision_scores_`. Custom detectors typically call it at the end of `fit`.

**Univariate vs multivariate distinction**: there is **no class-level flag**. The only signal is data shape passed to `fit` — `(n, 1)` for univariate, `(n, d>1)` for multivariate. Some models work on either (e.g. IForest); others have separate dispatch via `Sub_*` variants for sub-sequence extraction. The benchmark scripts split univariate vs multivariate by which file list and which HP dict they read.

---

## 3. Dataset loading interface

CSV format only. Each file has feature columns + a final `Label` column. Load idiom (used in both `TSB_AD/main.py:39-41` and `benchmark_exp/Run_Detector_M.py:55-57`):

```python
df = pd.read_csv(file_path).dropna()
data  = df.iloc[:, 0:-1].values.astype(float)   # (n, d), d>=1
label = df['Label'].astype(int).to_numpy()      # (n,)
```

**Train/test split is encoded in the filename**, not in the CSV. Files are named like `001_NAB_id_1_Facility_tr_1007_1st_2014.csv`, where `tr_1007` means rows `[:1007]` are training. Parsing (`Run_Detector_M.py:63`):

```python
train_index = filename.split('.')[0].split('_')[-3]
data_train = data[:int(train_index), :]
```

This is brittle — see Gaps section.

**File lists** live in `Datasets/File_List/`:
- `TSB-AD-U.csv` / `TSB-AD-M.csv` — full corpus
- `TSB-AD-U-Eva.csv` / `TSB-AD-M-Eva.csv` — held-out evaluation set
- `TSB-AD-U-Tuning.csv` / `TSB-AD-M-Tuning.csv` — HP tuning set

Each is a one-column CSV with header `file_name`. Multivariate eval set is roughly the half of the 1070-series corpus that is multivariate.

---

## 4. Models implemented in TSB-AD-M (multivariate)

44 files in `TSB_AD/models/`. The set is shared between univariate and multivariate runners — `Optimal_Multi_algo_HP_dict` (`HP_list.py:103-133`) defines which 28 are actually run for the multivariate benchmark:

**Statistical / classical**
`IForest`, `EIF`, `LOF`, `KNN`, `OCSVM`, `MCD`, `PCA`, `RobustPCA`, `HBOS`, `CBLOF`, `KMeansAD`, `KShapeAD`, `COPOD`, `MatrixProfile`, `SAND`, `Series2Graph`, `POLY`, `SR`, `FFT`, `Left_STAMPi`

**Deep / neural**
`AutoEncoder` (file: `AE.py`), `CNN`, `LSTMAD`, `xLSTMAD`, `Donut`, `OmniAnomaly`, `USAD`, `AnomalyTransformer`, `TranAD`, `TimesNet`, `FITS`, `PatchTST`, `M2N2`, `CHARM`, `MMPAD`, `Time_RCD`

**Foundation models**
`OFA` (frozen GPT-2), `Lag_Llama`, `Chronos`, `TimesFM`, `MOMENT`, `TSPulse`

**Important for our extension plan — these are already in TSB-AD:**
- USAD ✅ (`USAD.py`) — was on our "to add" list
- TranAD ✅ (`TranAD.py`) — was on our "to add" list
- OmniAnomaly ✅ (`OmniAnomaly.py`) — was on our "to add" list
- AnomalyTransformer ✅ (`AnomalyTransformer.py`) — was on our "to add" list
- **MTAD-GAT ❌** — only one from the user's list that is genuinely missing

---

## 5. Metrics implemented

`TSB_AD/evaluation/metrics.py:3` — `get_metrics(score, labels, slidingWindow=100, pred=None, version='opt', thre=250)` returns a dict of nine metrics. Implementations live in `TSB_AD/evaluation/basic_metrics.py` and the `vus/` and `affiliation/` subpackages.

| Returned key      | What it is                                                           | Notes                                                                                                |
|-------------------|----------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| `AUC-ROC`         | sklearn `roc_auc_score(labels, score)`                               | Threshold-independent                                                                                |
| `AUC-PR`          | sklearn `auc(recall, precision)`                                     | Threshold-independent                                                                                |
| `VUS-ROC`         | Volume Under Surface, ROC variant (Paparrizos et al., VLDB '22)      | Range-based, threshold-independent. Computed via `generate_curve()` from `vus/` subpackage           |
| `VUS-PR`          | VUS-PR variant — **the paper's headline reliable metric**            | Same                                                                                                 |
| `Standard-F1`     | Point F1 at oracle (best) threshold when `pred=None`                 | ⚠️ This is *oracle* F1, not raw operational F1 — see Gaps                                            |
| `PA-F1`           | Point-adjust F1 at oracle threshold                                  | Same caveat                                                                                          |
| `Event-based-F1`  | Event-level F1 (paper-specific definition)                           |                                                                                                      |
| `R-based-F1`      | Range-based F1 (Tatbul et al.)                                       |                                                                                                      |
| `Affiliation-F`   | Affiliation precision/recall F (Huet et al.)                         | Implementation in `evaluation/affiliation/metrics.py`                                                |

**Present:** point F1 (raw + PA), AUC-ROC, AUC-PR, VUS-ROC, VUS-PR, affiliation F, range-F1, event-F1.

**Absent / would need to add for our research:**
- **Detection delay** — not computed
- **Wall-clock training time** — only total `Time` is logged (a single number for fit + score together) at `Run_Detector_M.py:75-76`
- **Inference time per window** — not separated from training time
- **Peak GPU memory, parameter count, FLOPs** — not tracked at all
- **Operational point F1** — only oracle-threshold F1 is computed; there's no fixed-threshold or contamination-as-threshold variant

---

## 6. Datasets supported

The benchmark targets **TSB-AD-M** (multivariate, ~half of 1070 curated series across 40 source datasets). The README lists source datasets including SMD, MSL, SMAP, SWaT, WADI, GECCO, CreditCard, etc. — all flattened into the unified CSV format described in §3. Per-source domain is encoded in the filename.

The user wants to **also** add the original SMD / MSL / SMAP / SWaT in their native format alongside the TSB-AD-curated versions, so we can compare against papers that report on them.

---

## 7. Configuration system

**No YAML, no Hydra, no OmegaConf.** Configuration is two layers:

1. **Argparse** in each runner script (`TSB_AD/main.py:32`, `Run_Detector_M.py:31`) for paths and which detector to run.
2. **Hardcoded Python dicts** in `TSB_AD/HP_list.py` for hyperparameters:
   - `Multi_algo_HP_dict` — search grids
   - `Optimal_Multi_algo_HP_dict` — single best HP per model (used by `Run_Detector_M.py`)
   - Univariate equivalents at the bottom of the file

Adding YAML configs is one of our intended extensions — see CLAUDE.md.

---

## 8. Run commands

### Single experiment (one model, one file) — quoted from README and `main.py`

```bash
python -m TSB_AD.main --AD_Name IForest
```

Or, with explicit file:

```bash
python TSB_AD/main.py \
    --filename '001_NAB_id_1_Facility_tr_1007_1st_2014.csv' \
    --data_direc 'Datasets/TSB-AD-U/' \
    --AD_Name 'IForest'
```

### Full multivariate sweep — `benchmark_exp/Run_Detector_M.py`

```bash
cd benchmark_exp
python Run_Detector_M.py \
    --dataset_dir '../Datasets/TSB-AD-M/' \
    --file_lsit   '../Datasets/File_List/TSB-AD-M-Eva.csv' \
    --score_dir   'eval/score/multi/' \
    --save_dir    'eval/metrics/multi/' \
    --AD_Name     'IForest' \
    --save        True
```

⚠️ **`--file_lsit` is a real typo in the source** (`Run_Detector_M.py:33`). The flag must be spelled with the typo for argparse to accept it. Same typo appears in `Run_Detector_U.py`. Worth fixing upstream — but for now, scripts calling these need to use the misspelled flag.

### HP tuning sweep

```bash
cd benchmark_exp
python HP_Tuning_M.py \
    --dataset_dir '../Datasets/TSB-AD-M/' \
    --file_lsit   '../Datasets/File_List/TSB-AD-M-Tuning.csv' \
    --AD_Name     'IForest'
```

---

## 9. Results format

For one run of `Run_Detector_M.py --save True --AD_Name X`:

- **Per-file anomaly scores**: `eval/score/multi/X/<filename_without_ext>.npy` — 1D float array shape `(n_test_samples,)`
- **Per-model metrics CSV**: `eval/metrics/multi/X.csv`
  - Columns: `file, Time, AUC-PR, AUC-ROC, VUS-PR, VUS-ROC, Standard-F1, PA-F1, Event-based-F1, R-based-F1, Affiliation-F`
  - One row per dataset file
  - On exception, a row of zeros is written (`Run_Detector_M.py:91`) — silently masks failures
- **Log file**: `eval/score/multi/X/000_run_X.log` — per-file success/error lines

Everything is appended on the fly inside the file loop (the CSV is rewritten after each file), so partial runs leave usable output.

---

## 10. How to add a new model — minimal recipe

Four edits, no registry / decorator pattern:

1. **`TSB_AD/models/MyModel.py`** — implement `class MyModel(BaseDetector)` with `__init__(contamination=0.1, ...)`, `fit(X, y=None)`, `decision_function(X)`. Set `self.decision_scores_` in `fit` and call `self._process_decision_scores()`.

2. **`TSB_AD/model_wrapper.py`** — add a top-level function `run_MyModel_Unsupervised(data, **kwargs)` (or `_Semisupervised(data_train, data, **kwargs)`). The dispatcher uses `globals()[f"run_{name}_Unsupervised"]`-style lookup, so the function name **must** match the convention.

3. **`TSB_AD/model_wrapper.py:8-11`** — append `'MyModel'` to either `Unsupervise_AD_Pool` or `Semisupervise_AD_Pool`.

4. **`TSB_AD/HP_list.py`** — add an entry to `Multi_algo_HP_dict` (search grid) **and** `Optimal_Multi_algo_HP_dict` (single tuned setting). For univariate, add to the corresponding Uni dicts.

This is exactly the pattern we will follow from a parallel `extensions/` folder via monkey-patching — see CLAUDE.md.

---

## 11. Gaps, smells, and surprises

### Real bugs / smells
1. **`--file_lsit` typo** in `Run_Detector_{U,M}.py` and `HP_Tuning_{U,M}.py` argparse declarations. Live in the codebase.
2. **`Standard-F1` is oracle F1, not raw F1**: when `pred=None` (the default in `Run_Detector_M.py`), `metric_PointF1` sweeps thresholds and reports the best — this is an upper-bound metric, not what a deployed system would achieve. We need an "operational" F1 with a fixed/contamination-derived threshold for honest reporting.
3. **Silent failure path** — `Run_Detector_M.py:90-91` catches all exceptions and writes a zero row. Combined with the lack of tests, this hides real bugs in long sweeps.
4. **Sliding window from channel 0 only** — `slidingWindow = find_length_rank(data[:,0].reshape(-1, 1), rank=1)` (`Run_Detector_M.py:62`). For multivariate data, the window length used for VUS-{PR,ROC} ignores all but the first channel. This may bias results on series where the first column happens to be the boring one.
5. **Train/test split parsed from filename** — `filename.split('_')[-3]` (§3). One renamed file breaks everything.
6. **HP dict / search-grid drift** — `Optimal_Multi_algo_HP_dict` includes `KShapeAD`, `CHARM`, `PatchTST` (with empty `{}`!) that aren't in `Multi_algo_HP_dict`. `COPOD` has `{'HP': [None]}` in the search dict but `{'n_jobs': 1}` in the optimal dict — different keys.
7. **No tests, no CI** — no `tests/` directory, no GitHub Actions; correctness is benchmark-empirical.
8. **No type hints** anywhere in the runner scripts; minimal docstrings outside `base.py`.

### Surprises (worth flagging)
1. **4 of 5 "models to add" are already implemented.** The user's plan listed USAD, TranAD, OmniAnomaly, AnomalyTransformer, MTAD-GAT as additions — only MTAD-GAT is actually missing. Saves significant work; redirects effort to wrappers/configs and the genuine gap.
2. **No GPU memory / FLOPs / param count tracking.** Only wall-clock `Time` is logged, and that's combined fit + score. Anything reportable about computational cost in a paper has to be added by us.
3. **`base.py` is essentially a fork of PyOD's `BaseDetector`** — same docstrings, same attribute names. Means PyOD models drop in nearly trivially; also means our extensions stay compatible with the wider PyOD ecosystem if we keep the contract.

### What's strong
- Clean uniform `fit` / `decision_function` / `decision_scores_` contract across 40+ models
- Comprehensive metric coverage including the headline VUS-PR
- Datasets are well-curated and unified — same CSV format everywhere
- `eval/metrics/<MODEL>.csv` format is dead-simple to aggregate across runs

---

## 12. Dataset download (Session 5, 2026-05-11)

The full TSB-AD-M multivariate set was downloaded to EC2's EBS volume so subsequent sessions can run any TSB-AD model against the full benchmark.

- **Source URL**: `https://www.thedatum.org/datasets/TSB-AD-M.zip` (direct download from the TSB-AD authors' site; no auth required; served behind Cloudflare, origin LiteSpeed).
- **Zip size**: 540,383,983 bytes (~515 MiB / 540 MB). `last-modified: Sun, 10 May 2026 10:12 GMT`.
- **Method**: `wget --show-progress` inside a detached `tmux` session on EC2; sustained ~21.9 MB/s, full transfer in ~25 seconds. `unzip -t` integrity passed (every member CRC32 verified) before extraction.
- **Extract destination**: `~/mtad-platform/TSB-AD/Datasets/TSB-AD-M/`. The zip contains a single top-level `TSB-AD-M/` folder, so the extract command was `unzip -o /tmp/TSB-AD-M.zip -d ~/mtad-platform/TSB-AD/Datasets/`. The Session 4 bundled `057_SMD_id_1_Facility_tr_4529_1st_4629.csv` was overwritten by its byte-identical zip copy (net no-op).
- **Contents**: 200 CSVs, 2.5 GB extracted (compression ratio ~4.8×). All files dated 2024-10-23 (the canonical NeurIPS-D&B release timestamps).
- **Coverage**: 180/180 files listed in `Datasets/File_List/TSB-AD-M-Eva.csv` present on disk (0 missing, verified by a per-file `test -f` loop). The remaining ~20 files are the `TSB-AD-M-Tuning.csv` HP-tuning split.
- **Datasets represented**: MSL, SMAP, SMD, SVDB, MITDB, LTDB, GHL, Exathlon, CATSv2, Genesis (Sensor / Medical / Facility categories).
- **Format verified** on 3 random files (`080_LTDB_id_2_Medical`, `139_CATSv2_id_2_Sensor`, `002_MSL_id_1_Sensor`): canonical TSB-AD layout — header row + numeric feature columns + trailing `Label` column. Matches the contract `TSB_AD/utils/slidingWindows.py` and the runner scripts expect.
- **Gitignore**: covered by the bare `TSB-AD/` rule at `.gitignore:62`. `git check-ignore -v` confirms every dataset file is excluded; `git status` on EC2 stayed clean after extraction. No risk of accidental commit of the 2.5 GB payload.
- **Persistence**: the dataset lives only on EC2's EBS volume per the Phase 1 storage strategy. The backup cadence in CLAUDE.md (weekly tar to laptop, monthly EBS snapshot) applies. Re-download on a fresh machine is one `wget` + `unzip` away.
