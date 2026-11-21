"""
tests/test_engine.py
--------------------
Automated Unit & Property Test Suite for the Cross-Market Cloning Engine.
Verifies purity, temporal integrity, weight constraints, and mathematical soundness.
"""

import pytest
import numpy as np
import pandas as pd
from src.data_pipeline import get_curated_adr_universe
from src.features import chronological_split, scale_features
from src.lasso_model import calculate_tracking_error, sweep_alpha, extract_portfolio_at_k
from src.autoencoder import MacroFactorAutoencoder, train_macro_autoencoder, select_ae_cluster_portfolio
from src.benchmarks import build_equal_weight_benchmark

# Domestic S&P 500 mega-caps that MUST NEVER appear in the proxy pool
FORBIDDEN_DOMESTIC_STOCKS = {
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "TSLA",
    "BRK-B", "JPM", "XOM", "UNH", "V", "PG", "JNJ", "HD", "MA", "LLY"
}

def test_universe_purity():
    """Confirms ZERO domestic S&P 500 companies are in the ADR candidate pool."""
    universe = get_curated_adr_universe()
    contaminated = set(universe).intersection(FORBIDDEN_DOMESTIC_STOCKS)
    assert len(contaminated) == 0, f"Purity violation! Found domestic S&P 500 stocks: {contaminated}"

def test_tracking_error_identity():
    """Confirms tracking error of identical series is exactly 0.0."""
    series = np.array([0.01, -0.02, 0.015, -0.005, 0.02])
    te = calculate_tracking_error(series, series)
    assert abs(te) < 1e-12

def test_temporal_split_integrity():
    """Verifies that temporal splits are strictly disjoint, chronological, and non-empty."""
    dates = pd.date_range("2014-01-01", "2024-12-31", freq="B")
    dummy_data = pd.DataFrame(
        np.random.randn(len(dates), 4),
        index=dates,
        columns=["TSM", "ASML", "SAP", "^SP500TR"]
    )
    X_train, y_train, X_val, y_val, X_test, y_test = chronological_split(dummy_data, target_col="^SP500TR")
    
    assert X_train.index[-1] < X_val.index[0], "Train overlaps with Validation!"
    assert X_val.index[-1] < X_test.index[0], "Validation overlaps with Test!"
    assert len(X_train) > 0 and len(X_val) > 0 and len(X_test) > 0

def test_feature_scaling_causality():
    """Verifies scaler parameters are fit ONLY on training data."""
    X_train = pd.DataFrame({"A": [10.0, 20.0, 30.0], "B": [1.0, 2.0, 3.0]})
    X_val = pd.DataFrame({"A": [100.0, 200.0], "B": [10.0, 20.0]})
    X_test = pd.DataFrame({"A": [1000.0], "B": [100.0]})
    
    X_tr_s, X_v_s, X_te_s, scaler = scale_features(X_train, X_val, X_test)
    
    # Train mean must be 0 and variance 1
    np.testing.assert_almost_equal(X_tr_s.mean().values, [0.0, 0.0])
    np.testing.assert_almost_equal(scaler.mean_, [20.0, 2.0])

def test_portfolio_weight_constraints():
    """Verifies portfolio weights are strictly non-negative and sum to exactly 1.0."""
    feature_names = pd.Index(["TSM", "ASML", "SAP", "NVO"])
    ew_port = build_equal_weight_benchmark(feature_names, n_assets=4)
    
    assert (ew_port['Weight'] >= 0.0).all(), "Negative weights detected!"
    np.testing.assert_almost_equal(ew_port['Weight'].sum(), 1.0)

def test_autoencoder_proportional_quota():
    """Verifies dynamic factor-proportional clustering satisfies weight and cardinality bounds."""
    np.random.seed(42)
    n_assets = 12
    n_days = 100
    n_factors = 4
    
    feature_names = pd.Index([f"ASSET_{i}" for i in range(n_assets)])
    loadings = np.random.randn(n_assets, n_factors)
    X_train_raw = pd.DataFrame(np.random.randn(n_days, n_assets) * 0.01, columns=feature_names)
    y_train_raw = pd.Series(np.random.randn(n_days) * 0.01)
    
    for target_k in [4, 6]:
        port = select_ae_cluster_portfolio(loadings, feature_names, X_train_raw, y_train_raw, target_k=target_k)
        assert len(port) == target_k, f"Expected {target_k} assets, got {len(port)}"
        assert (port['Weight'] >= 0.0).all(), "Weights must be non-negative"
        np.testing.assert_almost_equal(port['Weight'].sum(), 1.0, decimal=5)

