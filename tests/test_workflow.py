import numpy as np
import pandas as pd

from src.config import load_config
from src.folds import make_walk_forward_folds
from src.models import equal_weight, select_ridge
from src.reporting import generate_report_plots


def test_walk_forward_folds_are_strictly_chronological():
    index = pd.date_range("2010-01-01", periods=1000, freq="D")
    folds = make_walk_forward_folds(index, 100, 50, 50, 2012, 2012)
    assert folds
    fold = folds[0]
    assert fold.estimation[-1] < fold.validation[0] < fold.test[0]


def test_models_are_long_only_and_fully_invested():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=80)
    X = pd.DataFrame(rng.normal(size=(80, 5)), index=dates, columns=list("ABCDE"))
    y = pd.Series(rng.normal(size=80), index=dates)
    weights = equal_weight(X, y, 3)
    assert np.isclose(weights.sum(), 1)
    assert (weights >= 0).all()

    selected, _ = select_ridge(X.iloc[:50], y.iloc[:50], X.iloc[50:], y.iloc[50:], (3,), (0.01,), 0.5)
    assert np.isclose(selected.sum(), 1)
    assert (selected >= 0).all()
    assert (selected <= 0.5 + 1e-10).all()


def test_current_config_loads():
    cfg = load_config()
    assert cfg.benchmark.primary == "^SP500TR"


def test_report_generates_plot_files(tmp_path):
    results = pd.DataFrame(
        [
            {
                "test_year": 2020,
                "model": "ridge",
                "tracking_error": 0.08,
                "correlation": 0.9,
                "portfolio_return": 0.1,
                "benchmark_return": 0.12,
                "active_return": -0.02,
                "active_holdings": 5,
            },
            {
                "test_year": 2021,
                "model": "ridge",
                "tracking_error": 0.07,
                "correlation": 0.92,
                "portfolio_return": 0.15,
                "benchmark_return": 0.14,
                "active_return": 0.01,
                "active_holdings": 6,
            },
        ]
    )
    paths = generate_report_plots(results, tmp_path)
    assert len(paths) == 3
    assert all(path.exists() for path in paths)
