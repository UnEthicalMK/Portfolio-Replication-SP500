import numpy as np
import pandas as pd

from src.models.lasso_model import train_lasso


def test_simplex_optimizer_enforces_maximum_cardinality():
    rng = np.random.default_rng(7)
    index = pd.date_range("2020-01-01", periods=80)
    columns = [f"ASSET_{i}" for i in range(6)]
    X = pd.DataFrame(rng.normal(0, 0.01, size=(80, 6)), index=index, columns=columns)
    y = pd.Series(X["ASSET_0"] * 0.4 + rng.normal(0, 0.005, 80), index=index)

    weights, *_ = train_lasso(
        X.iloc[:60],
        y.iloc[:60],
        X.iloc[60:],
        y.iloc[60:],
        min_assets=2,
        max_assets=2,
    )

    assert (weights > 0).sum() == 2
    assert np.isclose(weights.sum(), 1.0)
