import pandas as pd
import pytest

from src.backtest import simulate_buy_and_hold


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
