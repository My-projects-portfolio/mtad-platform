# -*- coding: utf-8 -*-
"""HiRE-Flow integration for the mtad-platform."""

from .detector import HiREFlow_AD
from .runner import run_HiREFlow

__all__ = ["HiREFlow_AD", "run_HiREFlow"]
