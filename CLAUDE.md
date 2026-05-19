# CLAUDE.md — Project Memory for the MTAD Research Platform

This file is loaded into every future Claude Code session for this project. Keep it accurate; update it whenever a load-bearing decision changes.

---

## Project goal

Build a **long-term personal multivariate time-series anomaly detection (MTAD) research platform** on top of [TSB-AD](https://github.com/TheDatumOrg/TSB-AD) (Liu & Paparrizos, NeurIPS 2024). The platform is a *research tool*, not a product — it has to support iterating across multiple papers over multiple years, including the user's own proposed novel MTAD method.

## User's research context

- The user is a researcher building toward a paper that proposes a new MTAD method.
- The platform must serve **multiple papers**, not a single experiment, so reusability and extensibility are first-class concerns.
- Outputs are: (i) reproducible benchmark numbers for baselines, (ii) clean ablations of the proposed method, (iii) consistent figure/table generation across papers.

## Completed milestones

- **Session 1**: GitHub repository + Pages site live (`jekyll-theme-cayman` placeholder, deploy workflow on push to `main`).
- **Session 2**: AWS S3 bucket `mtad-platform-imanian-2026` provisioned in `ap-southeast-2` with versioning, public-access-block, SSE-S3, and prefix layout seeded. Activation deferred to Phase 2 — see "Storage & sync architecture".
- **Session 3**: EC2 wired up (`mtad-ec2`, g5.xlarge / A10G). `node` + Claude Code installed on EC2; repo cloned to `~/mtad-platform/` on EBS. IAM permission gap surfaced and documented — Phase 2 will require an RMIT IT request.
- **Session 4 (2026-05-11)**: Full TSB-AD venv on EC2 (Python 3.11, PyTorch 2.11 + CUDA 13.0; A10G detected). TSB-AD installed editable; all five core models import cleanly. First end-to-end smoke test: IForest on the bundled SMD 057 dataset. Results CSV + scores `.npy` committed to git; `.gitignore` updated to remove the blanket `results/` exclusion. `gh` CLI installed on EC2 and authenticated; commit pushed from EC2, pulled to laptop.
- **Session 5 (2026-05-11)**: Full TSB-AD-M multivariate dataset downloaded to EC2 EBS — 200 CSVs / 2.5 GB at `~/mtad-platform/TSB-AD/Datasets/TSB-AD-M/`. All 180 files referenced by `TSB-AD-M-Eva.csv` present. Data not committed (gitignored). Platform ready to run any TSB-AD model against the full benchmark.
- **Session 6 (2026-05-18)**: **Multi-seed runner with cost instrumentation; first three baseline models exercised.** See "Experimental setup" and "Session 6 — what landed" below for detail.
- **Session 7 (2026-05-19)**: **First detector integrated via the `extensions/` wrapper pattern — MTGFLOW (Zhou et al., AAAI 2023).** Wrapper at `extensions/models/mtgflow/` translates TSB-AD's `(n_samples, n_features)` contract to MTGFLOW's `(B, K, L, D)` windowed-tensor contract; registry hook attaches it to TSB-AD's `Semisupervise_AD_Pool` and `model_wrapper.run_MTGFLOW` at runtime without touching TSB-AD core. Upstream model code (LSTM + Graph Attention + MAF) vendored under `_upstream/` from `github.com/zqhang/MTGFLOW` commit `b89a6ae506d9d04094c60a03e310d8bb71dd9478`, pinned so benchmark numbers stay reproducible across upstream changes. `_upstream/vendor.sh` applies exactly two mechanical patches (relative-import rewrite + a dead-`turtle` import strip for tkinter-less Python builds); full provenance and diffs in `_upstream/ATTRIBUTION.md`. Smoke test on SMD 057 (5 epochs, seed 42) passes all five correctness gates with AUC-ROC 0.9472, AUC-PR 0.4609 (vs 0.023 baseline = 20× lift), training neg-log-lik 1.957 → 1.622, 2.0s wall time on the A10G. Ready for the 10-dataset × 5-seed production sweep. **This is the template pattern for all future model adapters that aren't already in TSB-AD** — vendor upstream into `_upstream/` via a `vendor.sh` script, write a `BaseDetector` wrapper in `detector.py`, expose a `run_<Model>` in `runner.py`, register via `extensions/registry.py`. No edits to TSB-AD core.

### Known artifacts (current)

- `extensions/runners/run_baseline.py` — the multi-seed runner. See "Runner schema" for the full I/O contract.
- `extensions/configs/dev_subset.csv` — the 10-dataset dev tier (pinned).
- `results/runs/<Model>/<file_stem>__seed<N>.json` — one JSON sidecar per `(model, dataset, seed)` cell.
- `results/scores/<Model>/<file_stem>__seed<N>.npy` — one anomaly-score array per cell.
- `results/runs/<Model>_summary.csv` — flat aggregate, regenerated from successful sidecars on every runner exit.
- Current cells on disk: IForest (10 × 1 seed), AutoEncoder (10 × 5 seeds = 50), LSTMAD (8 × 1 seed; MSL and SMD error sidecars — see "Known limitations").

### Known followups

- **MTGFLOW upstream issue**: file an issue at `github.com/zqhang/MTGFLOW` flagging the two dead imports at the top of `models/MTGFLOW.py` — `from cgitb import reset` (shadowed by a local on line 21) and `from turtle import forward, shape` (`forward` is shadowed by method definitions, `shape` is only used as `.shape` attribute access). The `turtle` import blocks module load on Python builds without `tkinter` (e.g. our Amazon Linux 2023 venv). We strip it during vendoring; upstream should clean it up. Non-blocking — file when convenient.

## Key design principle: do NOT modify TSB-AD core in place

`TSB-AD/` is treated as a **vendored upstream** — read-only, pull-able. All extensions live in `extensions/`:

```
extensions/
├── models/         # New detectors (MTAD-GAT, UserMethod, ...) — subclass TSB_AD.models.base.BaseDetector
├── metrics/        # Detection delay, operational F1, ... (deferred until needed)
├── datasets/       # Native loaders for SMD, MSL, SMAP, SWaT (deferred until needed)
├── configs/        # Pinned file lists, run configs
├── runners/        # run_baseline.py (current) + future analysis utilities
└── analysis/       # Aggregator, paired stats, plots, LaTeX tables (Session 7+)
```

**Why this constraint**: TSB-AD is actively maintained upstream. We need to be able to `git pull` new models and fixes without merge conflicts in our research code.

**Compatibility patches go in `run_baseline.py`, not in TSB-AD.** Two patches currently live there: (a) `__sklearn_tags__` shim on `BaseDetector` for sklearn 1.6+ compatibility; (b) defensive HP-dict filtering against the per-model wrapper signature, since some wrappers (e.g. `run_IForest`) silently fail when handed kwargs they don't accept. Both are documented inline.

**Future TSB-AD will be a git submodule.** For now it is gitignored — we'll convert it in a later session.

---

## Experimental setup

This section is load-bearing — it pins the scope of every sweep until a paper-submission phase says otherwise.

### Dev subset (10 datasets)

The runner operates on a curated 10-dataset development tier pinned at `extensions/configs/dev_subset.csv`. One file per source family, chosen to span domains and dimensionality:

| Dataset | Channels | Train idx | Role |
|---|---|---|---|
| 002_MSL  | 55  | 500   | Canonical anchor — NASA Mars Science Lab |
| 027_MITDB | 2  | 25000 | Diversity — ECG, low-dim |
| 078_SMD | 38   | 500   | Canonical anchor — server monitoring |
| 115_PSM | 25   | 50000 | Canonical anchor — eBay server metrics |
| 132_OPPORTUNITY | 248 | 895 | Diversity — high-dim activity recognition |
| 139_CATSv2 | 17 | 5592  | Diversity — simulated dynamics |
| 166_SMAP | 25  | 1113  | Canonical anchor — NASA satellite |
| 171_SWaT | 66  | 3749  | Canonical anchor — water-treatment ICS |
| 173_GECCO | 9  | 16165 | Diversity — water quality |
| 187_Exathlon | 16 | 6193 | Diversity — Spark cluster traces |

**Reporting tier (deferred until paper submission)**: the full 180-file `TSB-AD-M-Eva.csv` split, run only for the proposed method plus 3–5 key baselines near submission. The dev tier is for iteration and ablations; the reporting tier is for headline numbers.

### Baseline canonical list (10 models)

The platform reports on these as its default baseline set:

- **Classical (3)**: IForest, OCSVM, LOF — universal baselines, non-trivial to beat
- **Reconstruction (3)**: AutoEncoder, OmniAnomaly, USAD — most-cited MTAD paradigm
- **Forecasting (1)**: LSTMAD — predict-and-residual approach
- **Transformer (2)**: AnomalyTransformer, TranAD — self-attention based
- **Recent SOTA (1)**: TimesNet — periodicity decomposition (ICLR 2023)

Plus the user's own proposed method, slotted in `extensions/models/UserMethod.py` (off the public repo until publish).

DCdetector and MTAD-GAT are candidates for later additions but require new adapters (they're not in TSB-AD). Foundation models (Chronos, MOMENT, TimesFM) are deferred.

### Multi-seed protocol

- **Canonical seed list**: `[13, 17, 42, 1337, 2024]`
- **5 seeds** for stochastic deep models (AE, USAD, LSTMAD, OmniAnomaly, TranAD, AnomalyTransformer, TimesNet, UserMethod) — gives mean ± std and enables paired Wilcoxon comparisons.
- **1 seed** for deterministic classical models (IForest, LOF, OCSVM). TSB-AD's stock wrappers for these don't forward `random_state` to the underlying class, so additional seeds would produce identical numbers. Running 1 seed is honest, not wasteful.

### Runner schema

`extensions/runners/run_baseline.py` is the single entry point. For each `(model, file, seed)` cell it writes three artifacts:

1. **JSON sidecar** at `results/runs/<Model>/<file_stem>__seed<N>.json`. Schema:
   ```
   model, file, seed, timestamp_utc, git_sha
   env: {python, platform, torch, numpy, cuda_available, cuda_version, gpu_name}
   hp, hp_dropped_keys
   status: "success" | "error"
   traceback (only when status=error)
   On success, additionally:
     n_samples, n_features, sliding_window, train_index
     wall_time_seconds
     fit_time_seconds, score_time_seconds       # split via class-method wrapping
     infer_per_window_ms                        # derived: score_time / n_test_windows
                                                # null for unsupervised (TSB-AD reads
                                                # decision_scores_ rather than calling
                                                # decision_function)
     peak_gpu_memory_bytes                      # torch.cuda.max_memory_allocated
     param_count                                # sum of trainable params across
                                                # top-level nn.Module instances
                                                # created during the run; 0 for sklearn
     score_path                                 # relative path to the .npy
     metrics: {all 9 TSB-AD detection metrics}
   ```

2. **Score `.npy`** at `results/scores/<Model>/<file_stem>__seed<N>.npy` — the raw `(n_samples,)` anomaly-score array. This is the **backfill currency**: any future detection metric (operational F1, detection delay, ROC-K) can be recomputed from these arrays without re-running.

3. **Summary CSV** at `results/runs/<Model>_summary.csv` — regenerated on every runner exit from successful sidecars only.

**Resume behaviour**: success sidecars are skipped; error sidecars are retried; `--force` re-runs everything (must be passed from `main()` through to `run_one()` — this propagation is wired correctly as of the Session 6 fix).

**Instrumentation hooks** that make the schema work:
- `torch.nn.Module.__init__` is patched at import time to register every module instance into a per-run list; `param_count` is computed post-run as the sum of trainable params across "root" modules (those not contained as sub-modules of any other captured module). Handles multi-network architectures like USAD.
- Every class in every `TSB_AD.models.*` module that defines both `fit` and `decision_function` is method-wrapped at import time to accumulate fit / score timing into a per-run dict. This catches both `BaseDetector` subclasses **and** stand-alone classes (LSTMAD, xLSTMAD) that don't inherit from `BaseDetector`.

### Metrics captured per cell

**Detection (9, all from TSB-AD's `get_metrics`):** AUC-PR, AUC-ROC, VUS-PR, VUS-ROC, Standard-F1, PA-F1, Event-based-F1, R-based-F1, Affiliation-F.

**Cost (added by platform):** wall_time, fit_time, score_time, infer_per_window_ms, peak_gpu_memory_bytes, param_count.

**Reproducibility (added by platform):** git_sha, env, hp (with dropped keys), seed, timestamp_utc.

### Metric philosophy: dual-reporting is a methodological contribution

TSB-AD's `Standard-F1` and `PA-F1` are **oracle** F1 — they sweep all thresholds and report the best. This is an upper bound, not what a deployed system actually achieves. Many MTAD papers report only oracle numbers, which inflates results.

**We always report BOTH operational AND oracle versions** of point-F1 and PA-F1 in every table and figure. The dual-reporting is itself worth highlighting as a methodological contribution in any paper produced from this platform — it directly addresses the "Elephant in the Room" critique that motivates TSB-AD itself.

**Operational F1 is currently NOT in the captured metrics list — it's backfillable from saved score arrays**, to be added as a Session 7+ task (see "Analysis utilities, pending").

### Deferred / not in instrumentation

- **FLOPs**: deferred indefinitely. Decision rationale: most recent MTAD papers (TranAD, AnomalyTransformer, DCdetector, OmniAnomaly, TSB-AD itself) don't report FLOPs; the engineering cost (~3–4 hours plus per-model failure handling for thop edge cases) outweighs the marginal table-strengthening value for non-efficiency papers. If a future paper specifically needs it, re-run the deep models with `thop.profile` instrumentation — known cost (~1 day of engineering plus a sweep), not a catastrophe.
- **Operational F1, detection delay, ROC-K, P@K**: backfillable from saved scores. Build when a specific paper needs them.

---

## Known limitations

### Deep models on small-train datasets

LSTMAD fails on `002_MSL` and `078_SMD` (both with `train_index=500`) with `RuntimeError: stack expects a non-empty TensorList`. Root cause: LSTMAD's wrapper splits 20% of training data for validation, leaving 100 samples; with `window_size + pred_len ≈ 100`, the validation `ForecastDataset` produces zero windows, the DataLoader is empty, and `torch.stack([])` fires.

**Expected to affect other windowed deep models**: USAD, OmniAnomaly, TranAD, AnomalyTransformer, TimesNet probably hit the same wall on the same two datasets. Verify on each model's smoke test.

**Behaviour by design**: error sidecars are still written (with traceback); `<Model>_summary.csv` excludes error cells naturally. For paper tables, per-dataset bar charts handle the gap cleanly (no bar for affected `(model, dataset)` cells). Document the exclusion in the methodology section if a reviewer asks.

---

## Storage & sync architecture

**Single source of truth for code** = the GitHub repository (https://github.com/My-projects-portfolio/mtad-platform). Every machine must be reproducible from `git clone` plus the data-restore path.

**Hard rule: any single file >5 MB does NOT go in git.** No exceptions.

### Phase 1 (current): EC2 EBS is the primary data store, git carries small results

Datasets and large artifacts live on the EC2 instance's EBS volume. Small results — JSON sidecars, summary CSVs, small `.npy` score files — travel via git to GitHub and the laptop, so the leaderboard and analysis tooling have a live copy without an scp step. The S3 bucket exists but is dormant — activated in Phase 2.

| Artifact | Lives in | Notes |
|---|---|---|
| Code, configs (YAML), website source | git | text, small |
| JSON sidecars (`results/runs/<Model>/*.json`) | git | one per cell |
| Summary CSVs (`results/runs/<Model>_summary.csv`) | git | regenerated, small |
| Small anomaly score arrays (`results/scores/<Model>/*.npy`) | git | committed when <5 MB |
| Large score arrays (≥ ~5 MB per cell, expected on long datasets) | EC2 EBS only | exclude case-by-case in `.gitignore` |
| Model checkpoints (`*.pt`, `*.ckpt`) | EC2 EBS only — `~/mtad-platform/results/checkpoints/` | always large; never in git |
| Datasets (TSB-AD bundled, future native SMD/MSL/SMAP/SWaT) | EC2 EBS only | downloaded once per machine |

**Workflow (current):** SSH into EC2 via VS Code Remote-SSH and work there. Code + small results: `git commit`/`git push` from EC2, `git pull` on the laptop. Large artifacts stay on EBS; `tar` + `scp` to laptop at paper-writing milestones.

**EBS durability — load-bearing risk to manage:** EBS has no versioning. Because EBS is the *only* copy of large artifacts in Phase 1, the backup cadence is mandatory:

- **Weekly:** `tar` of `~/mtad-platform/results/` and `~/mtad-platform/data/` pulled to the laptop via `scp`.
- **Monthly:** EBS snapshot via `aws ec2 create-snapshot` (or AWS console).
- **Cost discipline:** stop the GPU instance whenever not actively training. Light dev ~$30–50/month; heavy training ~$200–400/month.

### Phase 2 (deferred): S3 activation

Trigger conditions: (a) a collaborator joins, (b) results need to flow between multiple machines, (c) artifacts approach paper submission and need offsite/versioned durability.

Migration steps: (1) email RMIT IT to provision an IAM instance profile scoped to bucket `mtad-platform-imanian-2026` — the role-creation actions are denied to my SSO user; (2) `scripts/sync_from_s3.sh` / `sync_to_s3.sh` for pull/push; (3) bootstrap on a fresh machine becomes `git clone` → `sync_from_s3.sh datasets`.

### Caching invariant (applies in both phases)

**Experiments are immutable once computed.** Never re-run an experiment we already have. Only add new `(model × dataset × seed)` rows.

- Each cell is uniquely identified by its `<file_stem>__seed<N>.json` sidecar path.
- Resume logic skips success sidecars and retries error sidecars; `--force` re-runs everything.
- **Adding a new detection metric never requires re-training.** New metrics recompute from cached score arrays.

---

## Coding conventions

- **Python 3.11+** on EC2 (3.11.14 in the venv). Laptop runs whatever's locally installed; no laptop runs are load-bearing.
- **Type hints** in `extensions/` code where they aid clarity. Pragmatism beats dogma — don't fight typing for inscrutable TSB-AD interop.
- **No hardcoded paths** — paths come from CLI args or `pathlib.Path` derived from the project root.
- **Reproducibility** — every run logs git SHA, env, seed, and HP into the JSON sidecar. This is automatic via `run_baseline.py`.
- **Preserve the `--file_lsit` typo** when calling TSB-AD's upstream runner scripts. Do not "correct" it. Our `run_baseline.py` uses `--file_list` (corrected) because it's our code, not TSB-AD's.

---

## Vertical-slicing rule when adding a model/dataset/metric

When adding a new **model**, the unit of work is one full vertical slice — completed and committed in one session:

1. `extensions/models/<ModelName>.py` — the adapter (if it needs one beyond TSB-AD's existing wrapper).
2. `extensions/models/<ModelName>.meta.yaml` — metadata for the website.
3. Smoke test: single seed × 10 datasets on the dev subset, verify the JSON sidecars populate cleanly (`fit_time > 0`, `param_count` sensible, etc.).
4. Full 5-seed sweep on the dev subset.
5. Commit results + sidecars + summary CSV in one commit.

**Same rule applies to datasets and metrics.** Never leave a model "partially added" — half-finished slices accumulate as silent rot.

## Always present a plan before large multi-file changes

For any change that touches more than ~3 files or introduces a new abstraction, **write a short plan first** and get explicit user approval before editing. Bug fixes and one-file tweaks don't need this. The user has been burned by half-finished refactors and prefers the alignment cost.

---

## Datasets to focus on

- **TSB-AD-M curated set** — primary benchmark. Dev tier = the 10 files pinned in `extensions/configs/dev_subset.csv`. Reporting tier = the full 180 in `Datasets/File_List/TSB-AD-M-Eva.csv`.
- **SMD, MSL, SMAP, SWaT** in their **native formats** — for direct comparison against papers that don't report on TSB-AD. Loaders will go in `extensions/datasets/` when needed; deferred until a specific paper requires it.

## Hardware & AWS context

**Development laptop**: Windows 11 (corporate-managed), CPU-only. Quick iteration, smoke tests, code authoring. See "Application Control constraint" below.

**Training**: AWS EC2 GPU. Instance `i-0a259946708f10614` ("Nafis-EC2-GPU", g5.xlarge, NVIDIA A10G, Amazon Linux 2023, user `ec2-user`) — kept stopped between sessions, started only when training. SSH alias `mtad-ec2`; VS Code Remote-SSH verified working.

**AWS account context:**

- **Account type**: RMIT-managed, account ID `430442692195`. SSO portal: `https://rmit-research.awsapps.com/start`. Assume any new resource may need IT approval.
- **Auth**: SSO only. Local profile `mtad` aliases the `RMIT-ResearchAdmin` role. Refresh creds with `aws sso login --profile mtad`.
- **Region**: `ap-southeast-2` (Sydney). Keep all resources in this region.
- **S3 bucket**: `mtad-platform-imanian-2026`, dormant in Phase 1.
- **IAM permission gap (verified 2026-05-09)**: the `RMIT-ResearchAdmin` SSO role cannot create EC2 instance profiles. Phase 2 requires emailing RMIT IT. **Do NOT fall back to copying SSO credentials onto EC2** as a workaround.
- **Bucket ownership caveat**: the bucket lives in RMIT's AWS account. If you leave RMIT, you lose access — keep a personal-archive copy of any artifact load-bearing for thesis/paper submission.
- **Cost discipline**: stop the GPU instance whenever not actively training.

**Portability rule**: code must run on both CPU laptop and GPU EC2. Detect device with `torch.cuda.is_available()`; never assume CUDA.

## Application Control constraint on dev laptop

The Windows 11 laptop has **Application Control (deny-script-file-program policy)** enforced. Practical implications:

- Some installers must be staged into `C:\elevate\` before they will run.
- PowerShell is restricted in some contexts; PowerShell inside VS Code's integrated terminal works.
- Confirmed-working tools (no admin needed): `git`, `gh`, `ssh`, `claude`, `npm`, `python`, `node`, `winget`.
- Confirmed-blocked: enabling Windows services that are administratively disabled (e.g., `ssh-agent`).
- **Future sessions must not propose fixes that require local admin elevation.**

## GitHub & web presence

- **Repository**: https://github.com/My-projects-portfolio/mtad-platform (public)
- **Visibility implication**: the repo is public. Treat `CLAUDE.md` and `NOTES.md` as essentially **public documents**. Do **not** commit unpublished novel-method details — UserMethod implementation, ablation results, draft figures — until ready to publish.
- **Pages site**: https://my-projects-portfolio.github.io/mtad-platform/ — built from `docs/` via `.github/workflows/pages.yml`. Currently a Jekyll placeholder; Session 7+ work will start populating it from the JSON sidecars + summary CSVs.
- **Branch strategy**: only `main` for now.
- **SSH access**: project-specific key at `~/.ssh/id_ed25519_github`, routed in `~/.ssh/config` with `IdentitiesOnly yes`.
- **Commit identity**: `Nafiseh Imanian <57588284+My-projects-portfolio@users.noreply.github.com>`.
- **`gh` CLI**: authenticated to `github.com` as `My-projects-portfolio` with scopes `repo, read:org, gist`.

## Public website plan

**Long-term goal**: an auto-generated learning portal at the GitHub Pages URL — classifying every model on the platform by method category, with paper links, BibTeX, complexity, strengths/weaknesses, and a live leaderboard regenerated from `results/runs/`.

**Current state**: Jekyll placeholder. Once analysis utilities land, the leaderboard becomes the first dynamic page driven by `results/runs/<Model>_summary.csv` files.

**Mandatory metadata YAML**: every model in `extensions/models/<Name>.py` MUST have a sibling `extensions/models/<Name>.meta.yaml`. Required fields: `paper_title`, `authors`, `year`, `venue`, `paper_pdf_url`, `official_repo`, `bibtex`, `category`, `subcategory`, `key_idea`, `complexity`, `strengths`, `weaknesses`, `best_for`, `not_recommended_for`. Without the meta YAML the model isn't "added" per the vertical-slicing rule.

## Two-week supervisor demo

**Date**: around 2026-05-20.

**Deliverable status:**
- ✅ Working benchmark engine (multi-seed runner with cost instrumentation)
- ✅ Public GitHub repo
- ⚠️ Website skeleton — placeholder only, leaderboard not yet auto-generated
- ✅ Three baselines compared on 10 datasets (IForest, AutoEncoder, LSTMAD) — actual numbers
- ⚠️ Auto-generated LaTeX comparison table — analysis utilities pending (Session 7)
- ⚠️ Pareto plot — analysis utilities pending (Session 7)
- ✅ Demo deck (8-slide pptx) — explains datasets, metrics, baselines, paper outputs, roadmap

**Demo flow**:
1. Show the deck — 5-minute overview
2. Open the live Pages site (placeholder) + GitHub repo
3. Walk through the runner schema, show a sample JSON sidecar
4. Show IForest vs AE vs LSTMAD on three contrasting datasets (where each wins)
5. Walk through the roadmap (Session 7+)
6. Ask: scope cuts, paper venue, supervisor's preferred angle

---

## Session 6 — what landed

**Built `extensions/runners/run_baseline.py`** with the full schema described in "Runner schema" above. Includes: sklearn 1.6+ shim, defensive HP-dict filtering, `param_count` via nn.Module init hook, fit/score time split via class-method wrapping (broadened to all model classes with fit + decision_function, not just BaseDetector subclasses, after LSTMAD revealed it doesn't inherit from BaseDetector), `infer_per_window_ms` derivation for semisupervised models, `peak_gpu_memory_bytes`, full reproducibility metadata.

**Pinned the 10-dataset dev subset** at `extensions/configs/dev_subset.csv`. Decision criteria: one file per source family, balance canonical comparison anchors (SMD/MSL/SMAP/SWaT/PSM — the datasets every recent SOTA paper reports) with diversity additions (MITDB/GECCO/CATSv2/OPPORTUNITY/Exathlon — different domains, channel counts from 2 to 248, training-set sizes from 500 to 50K).

**Pinned the 10-baseline canonical list** spanning classical, reconstruction, forecasting, transformer, and recent SOTA paradigms.

**Pinned multi-seed protocol**: 5 seeds `[13, 17, 42, 1337, 2024]` for stochastic models; 1 seed for deterministic classical models (TSB-AD's wrappers for IForest/LOF/OCSVM don't forward `random_state`).

**Ran three baselines**:
- **IForest**: 10 cells × 1 seed. Deterministic, single seed sufficient. Used to validate the runner end-to-end.
- **AutoEncoder**: 10 cells × 5 seeds = 50 cells. First deep model. Validated nn.Module hook for param_count (~17K–81K params, scales with input dim) and GPU memory tracking (~19 MB per cell).
- **LSTMAD**: 8 cells × 1 seed (MSL and SMD failed — see "Known limitations"). First non-BaseDetector model; revealed the need to broaden instrumentation from `BaseDetector.__subclasses__()` to all model classes with `fit + decision_function` methods.

**Methodological observations the data already surfaces:**
- AE doesn't dominate IForest — they trade wins across 10 datasets. Real instance of the "Elephant in the Room" pattern under reliable measures.
- PA-F1 inflation is reproducible: AE hits PA-F1=1.0 on MSL/SMAP/Exathlon across all 5 seeds, despite AUC-PR in the 0.28–0.99 range. Validates the dual-reporting argument empirically.
- LSTMAD wins big on three datasets where neither baseline could: SMAP (0.83 vs 0.46 / 0.61), GECCO (0.50 vs 0.05 / 0.19), OPPORTUNITY (0.25 vs 0.18 / 0.06) — but loses on SWaT (0.25 vs AE's 0.54). No single approach wins everywhere.
- Cost picture is non-monotone in dataset size: AE is 7× faster than IForest on small datasets but slower on small-channel-count high-volume ones (MITDB).

**Decisions explicitly made and documented**:
- Skip FLOPs instrumentation. Re-run if a future paper specifically requires it.
- Accept the deep-model-on-small-train-data gap as a documented limitation rather than working around it with per-model HP overrides.

**Demo deck built** at `/home/claude/build_deck.js` (pptxgenjs) → `mtad-platform.pptx`. 8 slides: title, overview, datasets, metrics, baselines, experimental rigor, paper outputs, roadmap.

---

## Next session preview (Session 7)

**Priority order:**

1. **Finish LSTMAD's 5-seed sweep** (~40 min compute) — the four new seeds. The MSL/SMD error sidecars will repopulate but won't enter the summary CSV.
2. **Smoke-test the remaining deep models** in order: USAD → OmniAnomaly → TranAD → AnomalyTransformer → TimesNet. One seed × 10 datasets each, verify sidecars populate cleanly, then run the 5-seed sweep. USAD is the multi-network test case for `param_count` (sum across G + D networks).
3. **Build `extensions/analysis/`** — the "commit 2" work, all operating on existing data (no re-runs):
   - `aggregate.py`: mean ± std per `(model, dataset)`, paired Wilcoxon across cells
   - `plots.py`: per-dataset bar charts with seed error bars, Standard-F1 vs PA-F1 scatter (F1-inflation visualization), Pareto plot (accuracy × cost), critical-difference diagram via `autorank`
   - `tables.py`: LaTeX-ready comparison table generator
4. **Operational F1 backfill** — recompute from saved score arrays, add to summary CSVs. ~30 min of code.
5. **Auto-generate the leaderboard page** on the Pages site from `results/runs/<Model>_summary.csv` files.
6. **UserMethod scaffold** — `extensions/models/UserMethod.py` + ablation harness. OFF the public repo until publish.
7. **Convert `TSB-AD/` to a git submodule** — deferred from Session 6 since the work was moving fast on the runner. Low-risk one-session task.

---

## Quick orientation pointers

- Multi-seed runner: `extensions/runners/run_baseline.py`
- Dev subset: `extensions/configs/dev_subset.csv`
- TSB-AD base class: `TSB-AD/TSB_AD/models/base.py:21` — `BaseDetector`
- TSB-AD multivariate runner (upstream, unused now): `TSB-AD/benchmark_exp/Run_Detector_M.py`
- HP dicts: `TSB-AD/TSB_AD/HP_list.py` — `Optimal_Multi_algo_HP_dict`
- Metrics entry point: `TSB-AD/TSB_AD/evaluation/metrics.py:3` — `get_metrics(score, labels, slidingWindow=...)`
- TSB-AD's argparse flag is `--file_lsit` (typo'd in upstream) — preserve when calling upstream; our runner uses `--file_list` (corrected)

See `NOTES.md` for the full investigation report.