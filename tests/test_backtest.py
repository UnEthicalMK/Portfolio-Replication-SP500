import pandas as pd
import pytest

from src.backtest import simulate_buy_and_hold, simulate_rebalanced


def test_backtest_rejects_weights_that_do_not_sum_to_one():
    prices = pd.DataFrame(
        {
            "^SP500TR": [100.0, 101.0],
            "ASSET": [50.0, 51.0],
        },
        index=pd.date_range("2023-01-03", periods=2),
    )
    weights = pd.DataFrame({"Model": [0.5]}, index=["ASSET"])

    with pytest.raises(ValueError, match="sum to 1.0"):
        simulate_buy_and_hold(prices, weights)


def test_monthly_rebalancing_records_turnover():
    dates = pd.to_datetime(["2023-01-03", "2023-01-31", "2023-02-01"])
    prices = pd.DataFrame(
        {
            "^SP500TR": [100.0, 101.0, 102.0],
            "ASSET_A": [50.0, 51.0, 49.0],
            "ASSET_B": [50.0, 49.0, 52.0],
        },
        index=dates,
    )
    weights = pd.DataFrame({"Model": [0.5, 0.5]}, index=["ASSET_A", "ASSET_B"])

    _, _, _, _, turnover = simulate_rebalanced(prices, weights, "monthly")

    assert turnover.loc["2023-02-01", "Model"] > 0
