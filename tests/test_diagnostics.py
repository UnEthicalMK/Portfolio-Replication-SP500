import numpy as np
import pandas as pd

from src.diagnostics import calculate_max_drawdown, compute_metrics


def test_metrics_treat_returns_as_log_returns():
    portfolio = pd.DataFrame({"Model": [0.1, -0.1]})
    benchmark = pd.Series([0.0, 0.0], name="benchmark")
    weights = pd.DataFrame({"Model": [1.0]}, index=["ASSET"])

    metrics = compute_metrics(portfolio, benchmark, weights)

    expected = np.expm1(portfolio["Model"].sum())
    assert float(metrics.loc["Cumulative Return", "Model"]) == round(expected, 4)


def test_max_drawdown_uses_log_return_compounding():
    returns = pd.Series([0.1, -0.1])

    expected = (np.exp(0.1 - 0.1) - np.exp(0.1)) / np.exp(0.1)

    assert np.isclose(calculate_max_drawdown(returns), expected)
