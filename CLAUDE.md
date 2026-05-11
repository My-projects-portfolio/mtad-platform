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
- **Session 4 (2026-05-11)**: Full TSB-AD venv on EC2 (Python 3.11, PyTorch 2.11 + CUDA 13.0; A10G detected). TSB-AD installed editable; all five core models import cleanly (IForest, USAD, TranAD, AnomalyTransformer, OmniAnomaly). First end-to-end smoke test: IForest on the bundled SMD 057 dataset — AUC-ROC 0.80, AUC-PR 0.10, VUS-ROC 0.81, VUS-PR 0.10, Standard-F1 0.17, PA-F1 0.52, Affiliation-F 0.81. Results CSV + scores `.npy` committed to git; `.gitignore` updated to remove the blanket `results/` exclusion. `gh` CLI installed on EC2 and authenticated via device-code flow; commit pushed from EC2, pulled to laptop.
- **Session 5 (2026-05-11)**: Full TSB-AD-M multivariate dataset downloaded from `https://www.thedatum.org/datasets/TSB-AD-M.zip` to EC2 EBS — 200 CSVs / 2.5 GB at `~/mtad-platform/TSB-AD/Datasets/TSB-AD-M/`. All 180 files referenced by `TSB-AD-M-Eva.csv` present (0 missing); remaining ~20 are the HP-tuning split. Data not committed (gitignored via the bare `TSB-AD/` rule); lives only on EBS per Phase 1 storage strategy. Platform now ready to run any TSB-AD model against the full benchmark — Session 6+ work. See `NOTES.md` §12 for the download record.

### Known artifacts (post-Session 4)

- `data/smoke_test_list.csv` — file list pointing the runner at one dataset (the input to `--file_lsit`).
- `results/runs/IForest.csv` — first metrics CSV.
- `results/scores/IForest/057_SMD_id_1_Facility_tr_4529_1st_4629.npy` — first raw anomaly scores (190 KB).
- `057_SMD_id_1_Facility_tr_4529_1st_4629.csv` — 5.8 MB, 23,694 rows × 38 features + Label. **Bundled with TSB-AD** under `TSB-AD/Datasets/` (gitignored as part of the vendored upstream); not separately downloaded.

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

**Single source of truth for code** = the GitHub repository (https://github.com/My-projects-portfolio/mtad-platform). Every machine — dev laptop, EC2 GPU instance, future collaborator — must be reproducible from `git clone` plus the data-restore path described below.

**Hard rule: any single file >5 MB does NOT go in git.** No exceptions.

### Phase 1 (current, as of 2026-05-11): EC2 EBS is the primary data store, git carries small results

Datasets and large artifacts (checkpoints, large score arrays) live on the EC2 instance's EBS volume. **Small results — per-run CSVs and small `.npy` score files — travel via git** to GitHub and the laptop, so the leaderboard and analysis tooling have a live copy without an scp step. The S3 bucket exists but is dormant — activated in Phase 2 below. Reasons for this layout: (i) avoids the RMIT IT IAM-creation request that's currently blocked (see "Hardware & AWS context"), (ii) keeps the workflow simple while there's only one machine, (iii) lets the laptop render results from a plain `git pull`.

What goes where (updated post-Session 4):

| Artifact | Lives in | Notes |
|---|---|---|
| Code, configs (YAML), metadata YAMLs, leaderboard sources, website source | git | text, small |
| Per-run metrics CSVs (`results/runs/*.csv`) | git | one row per run |
| Small anomaly score arrays (`results/scores/<Model>/*.npy`, well under 5 MB) | git | committed since Session 4 — `.gitignore` no longer blanket-excludes `results/` |
| Large anomaly score arrays (≥ ~50 MB, expected once heavier models run) | EC2 EBS only — `~/mtad-platform/results/scores/` | exclude case-by-case in `.gitignore` when they appear; still bound by the 5 MB hard rule above for git |
| Model checkpoints (`*.pt`, `*.ckpt`) | EC2 EBS — `~/mtad-platform/results/checkpoints/` | always large; never in git |
| Datasets (TSB-AD-bundled CSVs, plus future SMD/MSL/SMAP/SWaT native) | EC2 EBS only — `TSB-AD/Datasets/` (gitignored as part of vendored upstream) and `~/mtad-platform/data/` | downloaded once per machine; never committed |

**Workflow (current, as exercised in Session 4):** SSH into EC2 via VS Code Remote-SSH and work there. Code + small results: `git commit`/`git push` from EC2, `git pull` on the laptop. Large artifacts (checkpoints, oversized scores): stay on EBS; `tar` + `scp` to laptop at paper-writing milestones. The laptop is a viewing window plus a local copy of code and lightweight results.

**EBS durability — load-bearing risk to manage:** EBS has no versioning, no cross-AZ durability, and dies with the instance if the volume is set to delete-on-termination. Because EBS is the *only* copy of large artifacts in Phase 1, the backup cadence is mandatory, not optional:

- **Weekly:** `tar` of `~/mtad-platform/results/` and `~/mtad-platform/data/` pulled to the laptop via `scp`.
- **Monthly:** EBS snapshot via `aws ec2 create-snapshot` (or AWS console).
- **Cost discipline:** stop the GPU instance whenever not actively training, even for short breaks. Light dev ~$30–50/month; heavy training ~$200–400/month. Track spending via the AWS Billing Dashboard weekly. EBS keeps charging while the instance is stopped, but at a much lower rate than GPU instance-hours.

### Phase 2 (deferred): S3 activation

Trigger conditions: (a) a collaborator joins, (b) results need to flow between multiple machines, (c) load-bearing artifacts approach paper submission and need offsite/versioned durability.

When triggered, the migration is:

1. Email RMIT IT to provision an IAM instance profile granting S3 access scoped to bucket `mtad-platform-imanian-2026` (the role-creation actions are denied to my SSO user — see "Hardware & AWS context"). Draft of the email lives in session notes.
2. `scripts/sync_from_s3.sh {datasets|scores|checkpoints|exports|all}` pulls; `scripts/sync_to_s3.sh ...` pushes. Both `--dry-run`-aware, idempotent (`aws s3 sync` only transfers changed files). Defaults: `MTAD_S3_BUCKET=mtad-platform-imanian-2026`, `AWS_PROFILE=mtad`. `*.sh` files are kept LF-only via `.gitattributes` so they run on Linux/EC2 even when authored on Windows.
3. Bootstrap on a fresh machine becomes `git clone` → `scripts/sync_from_s3.sh datasets`.

The bucket is already created with versioning on, all four public-access-block flags on, SSE-S3 encryption, and prefixes `datasets/ scores/ checkpoints/ exports/` seeded — so Phase 2 activation is purely about turning it on, not provisioning.

### Caching invariant (applies in both phases)

**Experiments are immutable once computed.** Never re-run an experiment we already have. Only add new `(model × dataset × seed)` rows.

- Each experiment gets a deterministic `run_id = hash(model_name + model_version + dataset_id + seed + hyperparams + git_sha)`.
- Layout (Phase 1):
  - `results/runs/<Model>.csv` (or `<run_id>.json` once the schema firms up) — small, committed to git: config, metrics, environment, timings.
  - `results/scores/<Model>/<dataset>.npy` — committed to git when small; EBS-only when large.
- **Adding a new metric never requires re-training.** New metrics recompute from cached score arrays (in git when small, on EBS when large).

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

**Training**: AWS EC2 GPU. Instance `i-0a259946708f10614` ("Nafis-EC2-GPU", g5.xlarge, NVIDIA A10G, Amazon Linux 2023, user `ec2-user`) — provisioned, kept stopped between sessions, only start when actively training. Pre-installed: `aws-cli` 2.33.27, `git` 2.50.1, `tmux` 3.2a. Missing (install in-session as needed): `node`, `claude`. SSH alias `mtad-ec2`; VS Code Remote-SSH verified working.

**AWS account context:**

- **Account type**: RMIT-managed (institutional), account ID `430442692195`. SSO portal: `https://rmit-research.awsapps.com/start`. Policy restrictions are possible — assume any new resource may need IT approval.
- **Auth**: SSO only (no long-lived access keys). Local profile `mtad` aliases the `RMIT-ResearchAdmin` role on the same account, sharing the `aws-rmit` SSO session. Refresh creds with `aws sso login --profile mtad` (loopback OAuth flow; opens the RMIT SSO page, redirects to `127.0.0.1`). Temp creds last ~hours; re-login when expired.
- **Region**: `ap-southeast-2` (Sydney). **Keep all resources in this region** — cross-region data transfer is both billed and slow.
- **S3 bucket**: `mtad-platform-imanian-2026` in `ap-southeast-2`. Versioning **enabled**, all four public-access-block flags **on**, server-side encryption SSE-S3 (AES-256, default). Bucket layout: `datasets/`, `scores/`, `checkpoints/`, `exports/` (each seeded with a `.keep` marker so the prefixes show up in the AWS console). **Status: dormant in Phase 1.** See "Storage & sync architecture" for activation criteria.
- **IAM permission gap (verified 2026-05-09)**: the `RMIT-ResearchAdmin` SSO role has `implicitDeny` on `iam:CreateRole`, `iam:PutRolePolicy`, `iam:CreateInstanceProfile`, `iam:AddRoleToInstanceProfile`, `iam:PassRole`, and `ec2:AssociateIamInstanceProfile` (all six checked via `simulate-principal-policy`). No permissions boundary on the role. Account is in AWS Organization `o-2vvrr6u1ue` (master `538238080661` = RMIT IT) with SCPs **enabled** but not introspectable from this account. Implication: I cannot create or attach an EC2 IAM instance profile myself — Phase 2 requires emailing RMIT IT to do it. **Do NOT fall back to copying SSO credentials onto EC2** as a workaround.
- **Bucket ownership caveat**: the bucket lives in RMIT's AWS account, not a personal account. RMIT pays the bill and ultimately controls deletion. **If you leave RMIT, you lose access** — keep a personal-archive copy of any artifact load-bearing for thesis/paper submission. The same caveat applies to the EC2 EBS volume in Phase 1, even more so since EBS has no versioning.
- **Cost discipline**: stop the GPU instance whenever it is not actively training. The dominant cost driver is GPU instance-hours; EBS storage is the next-largest line item in Phase 1. S3 storage cost is negligible (<$2/month at expected volumes) but currently zero since the bucket is dormant. Versioning will be on when activated, so deletes don't free space — old versions persist; revisit a lifecycle rule (expire non-current after N days) if storage grows.

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

## Session 3 (completed) — EC2 wiring

**Goal**: get the EC2 instance into a usable state for Phase 1 development. **All items closed.**

- ✅ SSH alias `mtad-ec2`; raw SSH and VS Code Remote-SSH both verified
- ✅ Instance identified (`i-0a259946708f10614`, g5.xlarge), pre-installed tooling inventoried
- ✅ IAM permission check completed — surfaced the gap above; storage strategy revised to Phase 1 (EBS-primary)
- ✅ `node` and `claude` installed on EC2
- ✅ Repo cloned to `~/mtad-platform/` on EBS
- ✅ EBS layout created: `~/mtad-platform/{data,results/runs,results/scores,results/checkpoints}/`
- ✅ Smoke test (no-op write under `results/`)
- ✅ Instance stopped at end of session

**Deferred to Phase 2 (S3 activation), not a Session 3 problem:**

- Email RMIT IT to provision `mtad-ec2-s3-role` and attach to the instance
- `scripts/sync_from_s3.sh` / `sync_to_s3.sh` plumbing
- Round-trip artifact test laptop ↔ S3 ↔ EC2

## Session 4 (completed) — TSB-AD venv + first end-to-end run

**Goal**: prove the benchmark engine works end-to-end on EC2 and that small results flow via git.

- ✅ Python 3.11 venv at `~/mtad-platform/.venv/`
- ✅ Installed TSB-AD's full `requirements.txt` (16 direct, ~80 transitive, ~5.5 GB venv)
- ✅ `pip install -e ./TSB-AD` — TSB-AD importable as editable package
- ✅ All five core models import cleanly: IForest, USAD, TranAD, AnomalyTransformer, OmniAnomaly
- ✅ PyTorch 2.11 + CUDA 13.0 confirmed; NVIDIA A10G detected
- ✅ Smoke test: IForest end-to-end via `TSB-AD/benchmark_exp/Run_Detector_M.py` on bundled SMD 057 (preserved the `--file_lsit` typo as documented). Results: AUC-PR 0.10, AUC-ROC 0.80, VUS-PR 0.10, VUS-ROC 0.81, Standard-F1 0.17, PA-F1 0.52, Affiliation-F 0.81
- ✅ `.gitignore` updated to remove the blanket `results/` exclusion; results CSV + 190 KB score `.npy` committed
- ✅ `gh` CLI installed on EC2, authenticated via device-code flow; commit pushed from EC2, pulled to laptop (after fixing `safe.directory` ownership)
- ✅ Instance stopped at end of session

## Next session preview

**Session 5**: Automated website generation from results — leaderboard page and per-model pages, generated from `results/runs/*.csv` and (future) `extensions/models/*.meta.yaml` files. The current Pages site is a static placeholder; Session 5 turns it into something driven by committed results.

**Session 6+** (order TBD per the vertical-slicing rule — one full slice per session, not bulk-import):

- Add metrics extensions in `extensions/metrics/`: operational F1 (fixed-threshold complement to oracle Standard-F1), FLOPs, parameter count, training/inference timing, peak GPU memory.
- Run additional baselines already in TSB-AD (USAD, TranAD, AnomalyTransformer) on the same SMD 057 starting point to populate the leaderboard.
- Convert `TSB-AD/` from gitignored vendor copy to a proper git submodule.
- Scaffold the `extensions/` directory skeleton.
- `pyproject.toml` and `pip install -e .` for the platform itself (separate from TSB-AD's editable install).
- First model adapter under `extensions/models/`: MTAD-GAT (the only baseline from the original list not already shipped in TSB-AD).

---

## Quick orientation pointers (kept minimal — see NOTES.md for detail)

- TSB-AD base class: `TSB-AD/TSB_AD/models/base.py:21` — `BaseDetector`
- Multivariate runner: `TSB-AD/benchmark_exp/Run_Detector_M.py`
- HP dicts: `TSB-AD/TSB_AD/HP_list.py`
- Metrics entry point: `TSB-AD/TSB_AD/evaluation/metrics.py:3` — `get_metrics(score, labels, slidingWindow=...)`
- Argparse flag is `--file_lsit` (typo'd in the source) — preserve when calling, don't "fix" upstream

See `NOTES.md` for the full investigation report.
