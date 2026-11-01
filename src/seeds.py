"""Deterministic seeding across random, numpy and torch (Spec v1.2.1 §10, AC-7)."""

from __future__ import annotations

import os
import random

import numpy as np


def set_all_seeds(seed: int) -> None:
    """Seed every RNG the pipeline uses, in one call.

    Note: ``PYTHONHASHSEED`` cannot influence hash randomization of an already
    running interpreter, so it is recorded for subprocesses only; pipeline
    code must not depend on set/dict iteration order (guarded by tests).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch  # optional runtime: only the A3 ablation needs it
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - CPU-only environment
        torch.cuda.manual_seed_all(seed)
