"""Leakage-safe expanding-window model retraining for the test period."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import TARGET
from src.models.equal_weight import train_equal_weight
from src.models.lasso_model import train_lasso
from src.models.sparse_autoencoder import train_sae


MODEL_NAMES = ("Equal_Weight", "Lasso", "SAE")


def fit_walk_forward_weights(
    returns: pd.DataFrame,
    rebalance_dates: list[pd.Timestamp],
    common_k: int,
    validation_days: int = 63,
) -> dict[str, pd.DataFrame]:
    """Fit each model using only observations strictly before each rebalance."""
    if TARGET not in returns.columns:
        raise ValueError(f"Returns are missing the benchmark column: {TARGET}")
    if validation_days < 1:
        raise ValueError("validation_days must be positive.")

    candidates = returns.drop(columns=[TARGET]).sort_index()
    benchmark = returns[TARGET].sort_index()
    result = {
        model: pd.DataFrame(index=rebalance_dates, columns=candidates.columns, dtype=float)
        for model in MODEL_NAMES
    }

    for rebalance_date in rebalance_dates:
        history_end = candidates.index[candidates.index < rebalance_date]
        if len(history_end) <= validation_days:
            raise ValueError(
                f"Insufficient pre-{rebalance_date.date()} history for walk-forward training."
            )
        history = candidates.loc[:history_end[-1]]
        target = benchmark.loc[history.index]
        X_train = history.iloc[:-validation_days]
        y_train = target.loc[X_train.index]
        X_val = history.iloc[-validation_days:]
        y_val = target.loc[X_val.index]

        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns
        )
        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val), index=X_val.index, columns=X_val.columns
        )

        lasso = train_lasso(
            X_train, y_train, X_val, y_val,
            min_assets=common_k, max_assets=common_k, verbose=False,
        )[0]
        equal = train_equal_weight(X_train, y_train, X_val, y_val, top_k=common_k)
        sae = train_sae(
            X_train_scaled, y_train, X_val_scaled, y_val,
            top_k=common_k, verbose=False,
        )
        for model, weights in zip(MODEL_NAMES, (equal, lasso, sae)):
            if not np.isclose(weights.sum(), 1.0, atol=1e-6):
                raise RuntimeError(
                    f"{model} produced invalid weights at {rebalance_date.date()}."
                )
            result[model].loc[rebalance_date] = weights

    return result
