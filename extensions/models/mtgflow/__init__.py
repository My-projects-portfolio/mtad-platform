"""MTGFLOW integration for the mtad-platform.

Wraps Zhou et al.'s MTGFLOW (AAAI 2023) inside TSB-AD's BaseDetector interface
so it can be benchmarked alongside the other multivariate detectors.

The vendored upstream code (the actual model — LSTM + Graph Attention + MAF) lives
under ``_upstream/`` and is treated as read-only. All adaptation to TSB-AD's data
contract lives in ``detector.py`` and ``runner.py``.

References:
    Zhou, Q., Chen, J., Liu, H., He, S., & Meng, W. (2023).
    Detecting Multivariate Time Series Anomalies with Zero Known Label.
    Proceedings of the AAAI Conference on Artificial Intelligence, 37(4), 4963–4971.
"""

from .detector import MTGFLOW_AD
from .runner import run_MTGFLOW

__all__ = ["MTGFLOW_AD", "run_MTGFLOW"]
