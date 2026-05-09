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

## Storage & sync architecture

**Single source of truth** = the GitHub repository (https://github.com/My-projects-portfolio/mtad-platform). Every machine — dev laptop, EC2 GPU instance, future collaborator — must be reproducible from `git clone` + a `scripts/sync_all.sh` (to be created).

**Hard rule: any single file >5 MB does NOT go in git.** No exceptions.

What goes where:

| Artifact | Lives in | Notes |
|---|---|---|
| Code, configs (YAML), metadata YAMLs, leaderboard sources, website source | git | text, small |
| Per-run results JSONs (each <5 MB) | git | one file per run |
| Anomaly score arrays (`*.npy`) | S3 | per-run, sometimes large |
| Model checkpoints (`*.pt`, `*.ckpt`) | S3 | always large |
| Datasets (TSB-AD-M, SMD, MSL, SMAP, SWaT, ...) | S3 | downloaded by `scripts/fetch_data.py`, never committed |

**Caching invariant: experiments are immutable once computed.** Never re-run an experiment we already have. Only add new `(model × dataset × seed)` rows.

- Each experiment gets a deterministic `run_id = hash(model_name + model_version + dataset_id + seed + hyperparams + git_sha)`.
- Layout:
  - `results/runs/<run_id>.json` — small, in git: config, metrics, environment, timings
  - `results/scores/<run_id>.npy` — large, in S3: raw decision-function scores
- **Adding a new metric never requires re-training.** New metrics recompute from cached `scores/<run_id>.npy`.
- The S3 bucket name and region (`ap-southeast-2`) are set up in Session 2 — see "Hardware & AWS context" below.

## Coding conventions

- **Python 3.10+** (use `match`, `|` union types, `pathlib.Path` everywhere)
- **Type hints everywhere** in `extensions/` code — `def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "MyDetector":`
- **Docstrings everywhere** in `extensions/` — short Google-style is fine; describe shapes (`(n, d)`) and units
- **No hardcoded paths** — all paths come from a YAML config or `pathlib.Path` derived from a project root sentinel
- **All configs in YAML** — argparse only for `--config path/to/config.yaml` and minor overrides; Hydra/OmegaConf preferred once we add it
- **Reproducibility** — every run logs git SHA, config YAML, seed, environment (CPU/GPU model, RAM, package versions) into the results directory
- **Preserve the `--file_lsit` typo** when calling TSB-AD's runner scripts. Do not "correct" it — argparse depends on the literal misspelling. Document this fact in any wrapper or runner that calls the upstream scripts.

## Vertical-slicing rule when adding a model/dataset/metric

When adding a new **model** to the platform, the unit of work is one full vertical slice — completed and committed in one session, not deferred:

1. `extensions/models/<ModelName>.py` — the adapter (subclass `BaseDetector` + a `run_<ModelName>_(Un|Semi)supervised` wrapper).
2. `extensions/models/<ModelName>.meta.yaml` — metadata (see "Public website plan" below for required fields).
3. The model's website page is generated from the meta YAML and committed under `docs/`.
4. The leaderboard is regenerated and committed.
5. Everything pushed in one commit (or one short series).

**Same rule applies to datasets and metrics.** Never leave a model "partially added" — half-finished slices accumulate as silent rot and break the website's auto-generation.

## Always present a plan before large multi-file changes

For any change that touches more than ~3 files or introduces a new abstraction, **write a short plan first** (use the `Plan` agent or `ExitPlanMode`) and get explicit user approval before editing. Bug fixes and one-file tweaks don't need this. The user has been burned by half-finished refactors and prefers the alignment cost.

## Models to add beyond TSB-AD's defaults

The user's original list was USAD, TranAD, OmniAnomaly, AnomalyTransformer, MTAD-GAT, plus a slot for the proposed method. **Investigation found that 4 of those 5 are already in TSB-AD** (`TSB_AD/models/{USAD,TranAD,OmniAnomaly,AnomalyTransformer}.py`). For the four already-present models, our work is to **wrap and configure** them via YAML rather than reimplement.

**Priority order:**

1. **MTAD-GAT** — graph-attention-based MTAD (Zhao et al. 2020). The only genuinely missing baseline from the original list.
2. **`<UserMethod>`** — placeholder slot for the user's own proposed method (file: `extensions/models/UserMethod.py`).
3. **Newer baselines as relevance demands** — decided per session, not all up front. Candidate pool to pull from:
   - **DCdetector** — dual-attention contrastive method
   - **TimesNet (anomaly variant)** — period-decomposition deep model
   - **ModernTCN** — modern temporal convolutional network
   - **Foundation models** — Chronos, MOMENT, TimesFM (TSB-AD already has stubs for several; verify what works MV)

Add a model only when a specific paper or experiment needs it. Do not bulk-import — the vertical-slicing rule (above) applies to each.

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

### Metric philosophy: dual-reporting is a methodological contribution

TSB-AD's `Standard-F1` and `PA-F1` are **oracle** F1 — they sweep all thresholds and report the best. This is an upper bound, not what a deployed system actually achieves. Many MTAD papers report only oracle numbers, which inflates results.

**We always report BOTH operational AND oracle versions** of point-F1 and PA-F1 in every table and figure. Reviewers see the deployment-realistic number alongside the theoretical ceiling. This dual-reporting is itself worth highlighting as a methodological contribution in any paper produced from this platform — it directly addresses the "Elephant in the Room" critique that motivates TSB-AD itself.

## Datasets to focus on

- **TSB-AD-M curated set** — primary benchmark (the eval split via `Datasets/File_List/TSB-AD-M-Eva.csv`)
- **SMD, MSL, SMAP, SWaT** in their **native formats** — for direct comparison against papers that don't report on TSB-AD. Loaders go in `extensions/datasets/` and convert to TSB-AD's CSV-with-trailing-Label-column format on the fly so they reuse TSB-AD's existing pipeline.

## Hardware & AWS context

**Development laptop**: Windows 11 (corporate-managed), CPU-only. Quick iteration, classical models, smoke tests, code authoring. See "Application Control constraint" below for environment caveats.

**Training**: AWS EC2 GPU. Existing instance is **already provisioned and currently stopped** — only start when actively training to avoid idle cost. Instance type TBD/already-set; verify in Session 2.

**AWS account context:**

- **Account type**: institutional (university-managed). Policy restrictions are possible — assume any new resource may need IT approval.
- **Region**: `ap-southeast-2` (Sydney). **Keep all resources in this region** — cross-region data transfer is both billed and slow.
- **S3 bucket** (for scores, checkpoints, datasets): name TBD, to be created in Session 2.
- **IAM**: access keys configured locally on the dev laptop.
- **Cost discipline**: stop the GPU instance whenever it is not actively training. Estimated S3 storage cost is negligible (<$2/month at expected volumes); the dominant cost driver is GPU instance-hours.

**Portability rule**: code must run on both CPU laptop and GPU EC2. Detect device with `torch.cuda.is_available()`; never assume CUDA.

## Application Control constraint on dev laptop

The Windows 11 laptop is corporate-managed with **Application Control (deny-script-file-program policy)** enforced. Practical implications for any future Claude Code session:

- Some installers must be staged into `C:\elevate\` before they will run.
- PowerShell is restricted in some contexts; **PowerShell inside VS Code's integrated terminal works**, which is where development happens.
- Confirmed-working tools (no admin needed): `git`, `gh`, `ssh`, `claude`, `npm`, `python`, `node`, `winget` (for installs from official sources).
- Confirmed-blocked: enabling/starting Windows services that are administratively disabled (e.g., `ssh-agent`) — needs admin elevation, which we do not have routinely.
- **Future sessions must not propose fixes that require local admin elevation.** If a step truly needs admin, flag it and find an admin-free alternative (e.g., `IdentitiesOnly yes` in `~/.ssh/config` instead of running `ssh-agent`, as in Session 1).

## GitHub & web presence

- **Repository**: https://github.com/My-projects-portfolio/mtad-platform (public — flipped from private to enable Pages on the free tier)
- **Visibility implication**: the repo is public. Treat `CLAUDE.md` and `NOTES.md` as essentially **public documents** from this point on. Do **not** commit unpublished novel-method details — proposed-method ideas, ablation results, draft figures — until ready to publish. Drafts live outside the repo (or in a private branch we explicitly mark as such, with a future submodule rework if needed). If we later move to a paid plan, we can flip back to private; until then, every push is world-readable.
- **Pages site**: https://my-projects-portfolio.github.io/mtad-platform/ — built from `docs/` via `.github/workflows/pages.yml` (Jekyll on GitHub Actions, theme `jekyll-theme-cayman`). Workflow re-runs on push to `main` when `docs/**` changes.
- **Branch strategy**: only `main` for now. Introduce feature branches + PR-based workflow when collaborators or CI gates appear.
- **SSH access**: project-specific key at `~/.ssh/id_ed25519_github`, routed in `~/.ssh/config` for `github.com` with `IdentitiesOnly yes`. No `ssh-agent` — key is read directly from disk on each git operation (the Windows `ssh-agent` service is disabled on this laptop and enabling it requires admin).
- **Commit identity**: `Nafiseh Imanian <57588284+My-projects-portfolio@users.noreply.github.com>` (GitHub no-reply form, set globally).
- **`gh` CLI**: installed via `winget install --id GitHub.cli`, authenticated to `github.com` as `My-projects-portfolio` with scopes `repo, read:org, gist`.

## Public website plan

**Long-term goal**: an auto-generated learning portal at the GitHub Pages URL — classifying every model on the platform by method category, with paper links, BibTeX, complexity analysis, strengths/weaknesses, and a live leaderboard regenerated from `results/runs/`.

**Current state**: Jekyll placeholder using `jekyll-theme-cayman`, deployed by `.github/workflows/pages.yml`. One static `docs/index.md`. Just enough to prove the deploy pipeline works.

**Future migration**: once we have content (≥3 model pages, leaderboard, comparison plots), migrate from Jekyll to **Quarto** or **MkDocs Material**. Decision deferred until we know what we want from the site (academic-paper-style vs. doc-site-style). The metadata YAMLs will drive the site so the migration is mostly a renderer swap, not a content rewrite.

**Mandatory metadata YAML**: every model in `extensions/models/<Name>.py` MUST have a sibling `extensions/models/<Name>.meta.yaml`. Required fields:

- `paper_title`, `authors`, `year`, `venue`
- `paper_pdf_url`, `official_repo`
- `bibtex` (full entry, multi-line)
- `category` (e.g. `reconstruction`, `forecasting`, `density`, `graph`, `transformer`, `foundation`)
- `subcategory` (free-form refinement)
- `key_idea` (one paragraph)
- `complexity` (training and inference, big-O in n, d)
- `strengths` (list)
- `weaknesses` (list)
- `best_for` (list of dataset/scenario types)
- `not_recommended_for` (list)

**Same metadata-YAML pattern applies to `extensions/datasets/` and `extensions/metrics/`.** Without the meta YAML the website page cannot be generated, which means the model is not "added" by the vertical-slicing rule.

## Two-week supervisor demo target

**Date**: around 2026-05-20 (~11 days from session 1).

**Deliverables**:

- Working benchmark engine (config-driven runs end-to-end, results persisted per the caching invariant)
- Public GitHub repo (live)
- Website skeleton with at least the model index and leaderboard scaffolding
- 3–4 models compared on 2–3 datasets — actual numbers, not placeholders
- Auto-generated LaTeX comparison table (operational + oracle metrics, dual-reported per the metric philosophy)
- Pareto plot — accuracy vs. computation (FLOPs or inference time)

**Demo flow**:

1. Open the live Pages site → leaderboard
2. Click into one model page (paper, BibTeX, complexity, strengths/weaknesses)
3. Show the repo layout — `extensions/`, configs, results
4. Run the LaTeX-table generator on stage → paste-ready table
5. Walk through the roadmap (next 4–6 weeks)
6. Ask: scope cuts, paper venue, supervisor's preferred angle

## Next session preview — Session 2

**Goal**: AWS plumbing end-to-end so deep models can train on GPU and results sync back.

- Install AWS CLI on the dev laptop
- Create the S3 bucket in `ap-southeast-2` (name to be chosen)
- Smoke-test upload + download of a small file
- Verify connection to the existing EC2 GPU instance via VS Code Remote-SSH
- Install Claude Code on the EC2 instance
- Full round-trip data sync test: push from laptop → S3 → pull from EC2

**Explicitly deferred to Session 3 or later** (do not let scope creep pull these into Session 2):

- Converting `TSB-AD/` from gitignored vendor copy to a proper git submodule
- Scaffolding the `extensions/` directory skeleton
- `pyproject.toml` and `pip install -e .`
- First model adapter (MTAD-GAT)

---

## Quick orientation pointers (kept minimal — see NOTES.md for detail)

- TSB-AD base class: `TSB-AD/TSB_AD/models/base.py:21` — `BaseDetector`
- Multivariate runner: `TSB-AD/benchmark_exp/Run_Detector_M.py`
- HP dicts: `TSB-AD/TSB_AD/HP_list.py`
- Metrics entry point: `TSB-AD/TSB_AD/evaluation/metrics.py:3` — `get_metrics(score, labels, slidingWindow=...)`
- Argparse flag is `--file_lsit` (typo'd in the source) — preserve when calling, don't "fix" upstream

See `NOTES.md` for the full investigation report.
