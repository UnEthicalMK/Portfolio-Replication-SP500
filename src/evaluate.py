"""Backtest and evaluation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def annualized_tracking_error(active_returns: pd.Series) -> float:
    return float(active_returns.std(ddof=1) * np.sqrt(252))


def evaluate_returns(
    portfolio: pd.Series,
    benchmark: pd.Series,
    weights: pd.Series,
) -> dict[str, float]:
    aligned = pd.concat([portfolio, benchmark], axis=1).dropna()
    aligned.columns = ["portfolio", "benchmark"]
    active = aligned["portfolio"] - aligned["benchmark"]
    return {
        "tracking_error": annualized_tracking_error(active),
        "correlation": float(aligned["portfolio"].corr(aligned["benchmark"])),
        "portfolio_return": float(np.expm1(aligned["portfolio"].sum())),
        "benchmark_return": float(np.expm1(aligned["benchmark"].sum())),
        "active_return": float(np.expm1(aligned["portfolio"].sum()) - np.expm1(aligned["benchmark"].sum())),
        "active_holdings": float((weights > 1e-10).sum()),
    }


def apply_initial_cost(weights: pd.Series, cost_bps: float) -> float:
    return float(weights.abs().sum() * cost_bps / 10_000)


def portfolio_returns(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    return returns.reindex(columns=weights.index).mul(weights, axis=1).sum(axis=1, min_count=1)
