# -*- coding: utf-8 -*-
"""Registry: hook mtad-platform extensions into TSB-AD's dispatch.

TSB-AD's runner machinery is name-based. ``run_Semisupervise_AD(name, ...)`` does
``globals()[f'run_{name}']`` against ``TSB_AD.model_wrapper`` and checks
``Semisupervise_AD_Pool`` membership. To make MTGFLOW (and future extensions)
visible to the same machinery without modifying TSB-AD in place, we:

  1. Append the model name to the pool list.
  2. Attach the runner function to ``model_wrapper`` as a module-level attribute.
  3. Add a default HP entry to ``Optimal_Multi_algo_HP_dict``.

All operations are idempotent.

Usage at the top of a runner script::

    from extensions.registry import register_all
    register_all()

After this, ``run_Semisupervise_AD('MTGFLOW', data_train, data_test, **hp)``
works as if MTGFLOW shipped inside TSB-AD.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# Default HPs for the multivariate Eva sweep. Match MTGFLOW's main.py defaults,
# with epochs=40 (a sensible platform default; the original repo varies per
# dataset). Hyperparameter search would override these via HP_Tuning_M.py.
_MTGFLOW_DEFAULT_HP = {
    "n_blocks": 1,
    "input_size": 1,
    "hidden_size": 32,
    "n_hidden": 1,
    "window_size": 60,
    "stride_size": 10,
    "batch_size": 512,
    "epochs": 40,
    "lr": 2e-3,
    "weight_decay": 5e-4,
    "dropout": 0.0,
    "batch_norm": False,
    "grad_clip_norm": 1.0,
    "score_aggregation": "mean",
}


def register_mtgflow() -> None:
    """Register MTGFLOW with TSB-AD's dispatch. Idempotent."""
    from TSB_AD import model_wrapper, HP_list

    from extensions.models.mtgflow import run_MTGFLOW

    # 1. Add to the semi-supervised pool (it trains on data_train).
    if "MTGFLOW" not in model_wrapper.Semisupervise_AD_Pool:
        model_wrapper.Semisupervise_AD_Pool.append("MTGFLOW")
        logger.info("registered MTGFLOW in Semisupervise_AD_Pool")

    # 2. Attach the runner so model_wrapper.globals()['run_MTGFLOW'] resolves.
    if not hasattr(model_wrapper, "run_MTGFLOW"):
        model_wrapper.run_MTGFLOW = run_MTGFLOW
        logger.info("attached run_MTGFLOW to TSB_AD.model_wrapper")

    # 3. Default HPs for the optimal-config runner.
    if "MTGFLOW" not in HP_list.Optimal_Multi_algo_HP_dict:
        HP_list.Optimal_Multi_algo_HP_dict["MTGFLOW"] = dict(_MTGFLOW_DEFAULT_HP)
        logger.info("registered MTGFLOW defaults in Optimal_Multi_algo_HP_dict")


def register_all() -> None:
    """Register every mtad-platform extension. Add new ``register_xxx()`` calls here."""
    register_mtgflow()