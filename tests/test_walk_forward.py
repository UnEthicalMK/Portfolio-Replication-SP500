import pandas as pd

from src.backtest import simulate_walk_forward


def test_walk_forward_rebalances_to_date_specific_targets():
    dates = pd.to_datetime(["2023-01-03", "2023-01-31", "2023-02-01"])
    prices = pd.DataFrame(
        {
            "^SP500TR": [100.0, 101.0, 102.0],
            "ASSET_A": [50.0, 60.0, 61.0],
            "ASSET_B": [50.0, 40.0, 39.0],
        },
        index=dates,
    )
    weights = {
        "Model": pd.DataFrame(
            [[1.0, 0.0], [0.0, 1.0]],
            index=pd.to_datetime(["2023-01-03", "2023-02-01"]),
            columns=["ASSET_A", "ASSET_B"],
        )
    }

    _, _, _, _, turnover = simulate_walk_forward(prices, weights)

    assert turnover.loc["2023-02-01", "Model"] > 0.5
