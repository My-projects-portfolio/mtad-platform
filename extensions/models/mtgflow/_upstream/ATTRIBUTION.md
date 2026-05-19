# MTGFLOW — Vendored Upstream Attribution

This directory contains the **MTGFLOW** model code, vendored from the original
authors' open-source release for use inside the mtad-platform.

## Source

- **Paper:** Zhou, Q., Chen, J., Liu, H., He, S., & Meng, W. (2023). *Detecting
  Multivariate Time Series Anomalies with Zero Known Label.* Proceedings of the
  AAAI Conference on Artificial Intelligence, 37(4), 4963–4971.
  https://arxiv.org/abs/2208.02108
- **Code repository:** https://github.com/zqhang/MTGFLOW
- **Files vendored from upstream:**
  - `models/MTGFLOW.py` → `MTGFLOW.py` here
  - `models/NF.py` → `NF.py` here

## Vendoring policy

These files are treated as **read-only upstream** — the same policy the
mtad-platform applies to the TSB-AD vendor copy:

- Do not modify the math (`LSTM`, `GNN`, `ScaleDotProductAttention`, `MAF`, `MADE`,
  `MaskedLinear`, `BatchNorm`, `FlowSequential`). These define MTGFLOW; changing
  them means we are no longer benchmarking MTGFLOW.
- Do not modify the forward pass, the log-probability computation, or the
  training/inference contracts.
- If you need a behavioural change for an experiment, do it in `detector.py`
  (the wrapper), not here.

## Diff applied during vendoring

**Two** mechanical changes are applied to `MTGFLOW.py` so it works as a submodule
of `extensions.models.mtgflow._upstream` and imports on Pythons without tkinter:

```diff
- from models.NF import MAF
+ from .NF import MAF
```

```diff
- from turtle import forward, shape
```

Justification for the second strip: this is dead code in upstream. `forward` is
only ever shadowed by `def forward(self, …)` method definitions inside the
classes below (e.g. lines 42, 106, 141, 201 of upstream `MTGFLOW.py`); `shape`
only appears in the file as an ndarray/tensor `.shape` attribute access. The
`turtle` module itself requires `tkinter`, which is not built into the Amazon
Linux 2023 Python 3.11 build the platform runs on, so leaving the import in
place produces a hard `ModuleNotFoundError: No module named 'turtle'` at module
load — even though the imported names are unused.

These are the only edits. Verify with:

```bash
diff -u <(curl -sL https://raw.githubusercontent.com/zqhang/MTGFLOW/main/models/MTGFLOW.py) MTGFLOW.py
```

You should see only the two changes above. Anything else is a vendoring bug
— re-run `vendor.sh` to repair.

`NF.py` is vendored byte-identical, no diff.

## License

The upstream MTGFLOW repository does not display a top-level LICENSE file at the
time of vendoring. We use this code under the standard academic-research norms
applicable to AAAI-published methods (research, comparison, reproduction). If you
intend to distribute this beyond academic research, contact the authors for an
explicit license grant.

## Citation requirement

Any paper, thesis, blog post, or talk that reports results produced by this code
**must cite** the MTGFLOW paper. Suggested BibTeX:

```bibtex
@inproceedings{zhou2023detecting,
  title     = {Detecting Multivariate Time Series Anomalies with Zero Known Label},
  author    = {Zhou, Qihang and Chen, Jiming and Liu, Haoyu and He, Shibo and Meng, Wenchao},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {37},
  number    = {4},
  pages     = {4963--4971},
  year      = {2023}
}
```

## Vendored snapshot

Vendored from https://github.com/zqhang/MTGFLOW commit `b89a6ae506d9d04094c60a03e310d8bb71dd9478` (2026-03-02) via `./vendor.sh /home/ec2-user/MTGFLOW`.
