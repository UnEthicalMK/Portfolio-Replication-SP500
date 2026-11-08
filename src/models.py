"""Portfolio construction models used by the research workflow."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def equal_weight(X: pd.DataFrame, y: pd.Series, k: int) -> pd.Series:
    correlations = X.corrwith(y).abs().sort_values(ascending=False)
    selected = correlations.head(min(k, len(correlations))).index
    weights = pd.Series(0.0, index=X.columns)
    weights.loc[selected] = 1.0 / len(selected)
    return weights


def _solve_simplex(
    X: pd.DataFrame,
    y: pd.Series,
    alpha: float,
    w_max: float,
) -> pd.Series:
    matrix = X.to_numpy()
    target = y.to_numpy()
    n_assets = matrix.shape[1]
    objective = lambda w: np.mean((matrix @ w - target) ** 2) + alpha * np.sum(w**2)
    result = minimize(
        objective,
        np.full(n_assets, 1.0 / n_assets),
        method="SLSQP",
        bounds=[(0.0, w_max)] * n_assets,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    if not result.success:
        raise RuntimeError(f"Simplex optimisation failed: {result.message}")
    if abs(result.x.sum() - 1.0) > 1e-8:
        raise RuntimeError("Simplex optimisation violated the full-investment constraint")
    return pd.Series(result.x, index=X.columns)


def ridge_candidate(
    X: pd.DataFrame,
    y: pd.Series,
    k: int,
    alpha: float,
    w_max: float,
) -> pd.Series:
    ranked = X.corrwith(y).abs().sort_values(ascending=False)
    selected = ranked.head(min(k, len(ranked))).index
    return _solve_simplex(X[selected], y, alpha, w_max).reindex(X.columns, fill_value=0.0)


def select_ridge(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    k_set: tuple[int, ...],
    alphas: tuple[float, ...],
    w_max: float,
) -> tuple[pd.Series, dict[str, float]]:
    candidates: list[tuple[float, pd.Series, int, float]] = []
    for k in k_set:
        if k > X_train.shape[1]:
            continue
        for alpha in alphas:
            weights = ridge_candidate(X_train, y_train, k, alpha, w_max)
            predicted = X_val[weights.index] @ weights
            score = float(np.mean((predicted - y_val) ** 2))
            candidates.append((score, weights, k, alpha))
    if not candidates:
        raise ValueError("No feasible ridge candidates")
    score, weights, k, alpha = min(candidates, key=lambda item: item[0])
    return weights, {"validation_mse": score, "k": k, "alpha": alpha}
