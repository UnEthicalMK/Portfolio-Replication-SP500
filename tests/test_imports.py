"""S0 acceptance: import smoke tests (roadmap S0).

`main.py` is deliberately excluded from the sweep loop: it executes the
pipeline at import time (audit finding C1). The second test pins that
behavior so the S1+ rewriter cannot regress it silently.
"""

import importlib
import sys

import pytest

_MODULES = [
    "src.backtest",
    "src.config",
    "src.data_pipeline",
    "src.diagnostics",
    "src.features",
    "src.models.equal_weight",
    "src.models.simplex_optimizer",
    "src.models.sparse_autoencoder",
    "src.models_pipeline",
    "src.seeds",
]


def test_all_library_modules_import():
    for name in _MODULES:
        importlib.import_module(name)  # raises on any import failure


def test_main_module_executes_at_import_time(monkeypatch):
    """PINS audit finding C1: importing `main` must not run the pipeline.

    With `--help`, argparse exits before any stage executes. Under a bare
    pytest invocation (no CLI arguments) the legacy module would launch the
    entire pipeline over the network — which is why the sweep above skips
    `main` and this test controls argv explicitly.
    """
    monkeypatch.setattr(sys, "argv", [sys.executable, "--help"])
    with pytest.raises(SystemExit) as excinfo:
        importlib.import_module("main")
    assert excinfo.value.code == 0
