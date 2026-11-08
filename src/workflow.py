"""End-to-end walk-forward research workflow."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import PipelineConfig
from src.evaluate import apply_initial_cost, evaluate_returns, portfolio_returns
from src.folds import make_walk_forward_folds
from src.models import equal_weight, select_ridge
from src.sae import select_sae


def _complete_columns(returns: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if returns[column].notna().all()]


def run_walk_forward(
    returns: pd.DataFrame,
    cfg: PipelineConfig,
) -> pd.DataFrame:
    benchmark = cfg.benchmark.primary
    folds = make_walk_forward_folds(
        returns.index,
        cfg.folds.estimation_days,
        cfg.folds.validation_days,
        cfg.folds.test_days,
        cfg.folds.first_test_year,
        cfg.folds.last_test_year,
    )
    rows: list[dict[str, object]] = []
    for fold in folds:
        estimation = returns.loc[fold.estimation]
        validation = returns.loc[fold.validation]
        test = returns.loc[fold.test]
        candidates = [column for column in returns.columns if column != benchmark]
        candidates = _complete_columns(estimation, candidates)
        if not candidates:
            continue
        X_train, y_train = estimation[candidates], estimation[benchmark]
        validation_frame = validation[[benchmark, *candidates]].dropna()
        if validation_frame.empty:
            continue
        X_val, y_val = validation_frame[candidates], validation_frame[benchmark]
        X_test, y_test = test[candidates], test[benchmark]

        model_weights = {
            "equal_weight": equal_weight(X_train, y_train, min(cfg.optimizer.k_set[-1], len(candidates))),
        }
        ridge_weights, metadata = select_ridge(
            X_train, y_train, X_val, y_val,
            cfg.optimizer.k_set, cfg.optimizer.alphas, cfg.optimizer.w_max,
        )
        model_weights["ridge"] = ridge_weights
        sae_weights, sae_metadata = select_sae(
            X_train, y_train, X_val, y_val,
            cfg.optimizer.k_set, cfg.optimizer.w_max, cfg.sae,
        )
        model_weights["sae"] = sae_weights

        for model_name, weights in model_weights.items():
            p_returns = portfolio_returns(X_test, weights)
            cost = apply_initial_cost(weights, cfg.costs.default_bps)
            if len(p_returns):
                p_returns = p_returns.copy()
                p_returns.iloc[0] -= cost
            metrics = evaluate_returns(p_returns, y_test, weights)
            rows.append(
                {
                    "test_year": fold.test_year,
                    "model": model_name,
                    "selected_k": (
                        metadata["k"] if model_name == "ridge"
                        else sae_metadata["k"] if model_name == "sae"
                        else float((weights > 0).sum())
                    ),
                    "alpha": metadata["alpha"] if model_name == "ridge" else float("nan"),
                    "seed": sae_metadata["seed"] if model_name == "sae" else float("nan"),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def save_results(results: pd.DataFrame, output_dir: str | Path) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "walk_forward_metrics.csv"
    results.to_csv(path, index=False)
    return path
