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
from src.stepwise import run_forward_stepwise
from src.benchmarks import build_equal_weight_benchmark
from src.pareto import compute_auc, compute_economic_loss_curve
from src.backtest import simulate_portfolio

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

def test_lasso_and_autoencoder_training():
    """Verifies Lasso sweep/extraction and Autoencoder forward/training shapes."""
    rng = np.random.RandomState(7)
    n_days, n_assets = 80, 6
    cols = [f"ADR_{i}" for i in range(n_assets)]
    X_train = pd.DataFrame(rng.randn(n_days, n_assets) * 0.01, columns=cols)
    X_val = pd.DataFrame(rng.randn(30, n_assets) * 0.01, columns=cols)
    X_test = pd.DataFrame(rng.randn(20, n_assets) * 0.01, columns=cols)
    y_train = pd.Series(0.5 * X_train["ADR_0"] + 0.5 * X_train["ADR_1"] + rng.randn(n_days) * 0.002)
    y_val = pd.Series(0.5 * X_val["ADR_0"] + 0.5 * X_val["ADR_1"] + rng.randn(30) * 0.002)

    X_tr_s, X_va_s, _, scaler = scale_features(X_train, X_val, X_test)
    lasso_res = sweep_alpha(X_tr_s, y_train, X_val, y_val, scaler, alphas=np.logspace(-5, -2, 12))
    port, _, val_te = extract_portfolio_at_k(lasso_res, target_k=3, feature_names=pd.Index(cols))
    assert 1 <= len(port) <= n_assets
    assert (port["Weight"] > 0.0).all()
    np.testing.assert_almost_equal(port["Weight"].sum(), 1.0, decimal=6)
    assert val_te >= 0.0

    model = train_macro_autoencoder(X_tr_s, X_va_s, n_factors=3, epochs=5, patience=3)
    assert isinstance(model, MacroFactorAutoencoder)

def test_stepwise_pareto_and_backtest_modes():
    """Verifies Stepwise sequential selection, Economic Loss K*, and Mode A vs Mode B accounting."""
    rng = np.random.RandomState(123)
    dates = pd.date_range("2022-01-03", periods=150, freq="B")
    cols = ["A", "B", "C", "D"]
    X_ret = pd.DataFrame(rng.randn(len(dates), len(cols)) * 0.01, index=dates, columns=cols)
    y_ret = pd.Series(0.6 * X_ret["A"] + 0.4 * X_ret["B"], index=dates, name="^SP500TR")

    step_ports = run_forward_stepwise(X_ret, y_ret, max_k=3)
    assert set(step_ports.keys()) == {1, 2, 3}
    for k, port in step_ports.items():
        assert len(port) == k
        np.testing.assert_almost_equal(port["Weight"].sum(), 1.0, decimal=6)

    frontier_df = pd.DataFrame({
        "K": [1, 2, 3],
        "Lasso_TE": [12.0, 9.0, 8.95],
        "AE_TE": [12.5, 9.2, 9.15],
        "Stepwise_TE": [11.8, 9.1, 9.05],
    })
    assert compute_auc([1, 2, 3], [12.0, 9.0, 8.95]) > 0
    loss_df, opt_k = compute_economic_loss_curve(frontier_df, lambda_friction=0.5, fixed_custody_bps_per_asset=0.2)
    assert opt_k["Lasso_K_star"] == 2
    assert len(loss_df) == 3

    combined = X_ret.join(y_ret)
    _, metrics_a, sub_a = simulate_portfolio(step_ports[2], combined, rebalance_freq="none")
    _, metrics_b, _ = simulate_portfolio(step_ports[2], combined, rebalance_freq="quarterly")

    assert metrics_a["Rebalance Events"] == 0
    assert metrics_a["Turnover Volume Traded ($)"] == 0.0
    assert metrics_a["Total Friction Paid ($)"] == pytest.approx(1000.0)
    assert metrics_b["Rebalance Events"] > 0
    assert metrics_b["Turnover Volume Traded ($)"] > 0.0
    assert metrics_b["Total Friction Paid ($)"] > metrics_a["Total Friction Paid ($)"]
    assert 2022 in sub_a

