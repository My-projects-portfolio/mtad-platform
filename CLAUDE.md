# CLAUDE.md — Project Memory for the MTAD Research Platform

This file is loaded into every future Claude Code session for this project. Keep it accurate; update it whenever a load-bearing decision changes.

---

## Project goal

Build a **long-term personal multivariate time-series anomaly detection (MTAD) research platform** on top of [TSB-AD](https://github.com/TheDatumOrg/TSB-AD) (Liu & Paparrizos, NeurIPS 2024). The platform is a *research tool*, not a product — it has to support iterating across multiple papers over multiple years, including the user's own proposed novel MTAD method.

## User's research context

- The user is a researcher building toward a paper that proposes a new MTAD method.
- The platform must serve **multiple papers**, not a single experiment, so reusability and extensibility are first-class concerns.
- Outputs are: (i) reproducible benchmark numbers for baselines, (ii) clean ablations of the proposed method, (iii) consistent figure/table generation across papers.

## Key design principle: do NOT modify TSB-AD core in place

`TSB-AD/` is treated as a **vendored upstream** — read-only, pull-able. All extensions live in a parallel top-level `extensions/` folder (to be created) with this layout:

```
extensions/
├── models/         # New detectors (MTAD-GAT, our proposed method, ...) — subclass TSB_AD.models.base.BaseDetector
├── metrics/        # Detection delay, operational F1, peak GPU memory, FLOPs, param count, inference time per window
├── datasets/       # Loaders for SMD, MSL, SMAP, SWaT in their native formats
├── configs/        # YAML configs (one per experiment) — Hydra/OmegaConf-style
├── runners/        # YAML-driven sweep scripts that import TSB-AD's wrappers without forking them
└── registry.py     # Monkey-patch hook: register new entries into TSB_AD.model_wrapper.Unsupervise_AD_Pool / HP_list dicts at import time
```

**Why this constraint**: TSB-AD is actively maintained upstream. We need to be able to `git pull` new models, fixes, and the leaderboard schema without merge conflicts in our research code.

**Future TSB-AD will be a git submodule.** For now (this initial commit) it is gitignored — we'll convert it to a proper submodule in a later session.

## Coding conventions

- **Python 3.10+** (use `match`, `|` union types, `pathlib.Path` everywhere)
- **Type hints everywhere** in `extensions/` code — `def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "MyDetector":`
- **Docstrings everywhere** in `extensions/` — short Google-style is fine; describe shapes (`(n, d)`) and units
- **No hardcoded paths** — all paths come from a YAML config or `pathlib.Path` derived from a project root sentinel
- **All configs in YAML** — argparse only for `--config path/to/config.yaml` and minor overrides; Hydra/OmegaConf preferred once we add it
- **Reproducibility** — every run logs git SHA, config YAML, seed, environment (CPU/GPU model, RAM, package versions) into the results directory
- **Preserve the `--file_lsit` typo** when calling TSB-AD's runner scripts. Do not "correct" it — argparse depends on the literal misspelling. Document this fact in any wrapper or runner that calls the upstream scripts.

## Models to add beyond TSB-AD's defaults

The user's original list was USAD, TranAD, OmniAnomaly, AnomalyTransformer, MTAD-GAT, plus a slot for the proposed method. **Investigation found that 4 of those 5 are already in TSB-AD** (`TSB_AD/models/{USAD,TranAD,OmniAnomaly,AnomalyTransformer}.py`). The actual addition list is therefore:

- **MTAD-GAT** — graph-attention-based MTAD (Zhao et al. 2020). The only genuinely missing baseline from the original list.
- **`<UserMethod>`** — placeholder slot for the user's own proposed method (file: `extensions/models/UserMethod.py`)
- (Optional, decide later) Newer baselines like DCdetector, CARLA, GenAD if the literature review surfaces them as relevant

For the four already-present models, our work is to **wrap and configure** them via YAML rather than reimplement.

## Metrics to ensure are included

TSB-AD ships: `AUC-ROC, AUC-PR, VUS-ROC, VUS-PR, Standard-F1 (oracle), PA-F1 (oracle), Event-based-F1, R-based-F1, Affiliation-F`. Our additions live in `extensions/metrics/`:

- **Point F1 — operational** (fixed threshold, e.g. contamination percentile) and **point F1 — oracle** (already exists as `Standard-F1`). Report both, never just one.
- **PA-F1** — operational and oracle, same reasoning
- **Detection delay** (median + p95 number of timesteps from anomaly start to first alert)
- **Training time** (fit only, separated from scoring)
- **Inference time per window** (per-sample throughput)
- **Peak GPU memory** (`torch.cuda.max_memory_allocated`)
- **Parameter count** (sum of `numel()` for trainable params)
- **FLOPs** (per inference forward pass; use `fvcore` or `thop`)

## Datasets to focus on

- **TSB-AD-M curated set** — primary benchmark (the eval split via `Datasets/File_List/TSB-AD-M-Eva.csv`)
- **SMD, MSL, SMAP, SWaT** in their **native formats** — for direct comparison against papers that don't report on TSB-AD. Loaders go in `extensions/datasets/` and convert to TSB-AD's CSV-with-trailing-Label-column format on the fly so they reuse TSB-AD's existing pipeline.

## Hardware context

- **Development**: Windows 11 laptop, CPU-only. Quick iteration, classical models, smoke tests.
- **Training**: AWS EC2 GPU (to be set up — instance type TBD, likely g5.xlarge or g6.xlarge). All deep models train there.
- Code must run on both. Detect device with `torch.cuda.is_available()`; never assume CUDA.

## Always present a plan before large multi-file changes

For any change that touches more than ~3 files or introduces a new abstraction, **write a short plan first** (use the `Plan` agent or `ExitPlanMode`) and get explicit user approval before editing. Bug fixes and one-file tweaks don't need this. The user has been burned by half-finished refactors and prefers the alignment cost.

---

## Quick orientation pointers (kept minimal — see NOTES.md for detail)

- TSB-AD base class: `TSB-AD/TSB_AD/models/base.py:21` — `BaseDetector`
- Multivariate runner: `TSB-AD/benchmark_exp/Run_Detector_M.py`
- HP dicts: `TSB-AD/TSB_AD/HP_list.py`
- Metrics entry point: `TSB-AD/TSB_AD/evaluation/metrics.py:3` — `get_metrics(score, labels, slidingWindow=...)`
- Argparse flag is `--file_lsit` (typo'd in the source) — preserve when calling, don't "fix" upstream

See `NOTES.md` for the full investigation report.
