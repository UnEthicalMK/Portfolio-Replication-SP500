"""
src/benchmarks.py
-----------------
Benchmark & Statistical Null Hypothesis Generation Suite.
Constructs:
1. Equal-Weighted Top Liquid ADR baseline (practitioner heuristic).
2. Monte Carlo Random Subset Null Model to establish
   statistical significance envelopes against random asset selection.
"""

import numpy as np
import pandas as pd
from scipy.optimize import nnls

def compute_portfolio_te(weights_df, X_eval, y_eval):
    """
    Computes annualized tracking error (%) for a portfolio on evaluation returns.
    """
    tickers = weights_df['Ticker'].tolist()
    weights = weights_df['Weight'].values
    port_returns = X_eval[tickers].values @ weights
    diff = y_eval.values - port_returns
    return np.std(diff, ddof=1) * np.sqrt(252) * 100.0

def build_equal_weight_benchmark(feature_names, n_assets=15, prioritized_tickers=None):
    """
    Constructs an equal-weight portfolio of top global mega-cap foreign ADRs.
    """
    if prioritized_tickers is None:
        # Recognized global tier-1 mega-caps spanning key sectors
        prioritized_tickers = [
            "TSM", "ASML", "SAP", "SONY", "NVO", "NVS", "AZN", "SHEL",
            "BP", "TTE", "RY", "TD", "HSBC", "BHP", "RIO", "BABA", "BUD", "DEO"
        ]
        
    available = [t for t in prioritized_tickers if t in feature_names]
    selected = available[:n_assets]
    if len(selected) < n_assets:
        remaining = [t for t in feature_names if t not in selected]
        selected.extend(remaining[:(n_assets - len(selected))])
        
    weights = np.ones(len(selected)) / len(selected)
    return pd.DataFrame({
        'Ticker': selected,
        'Weight': weights
    })

def simulate_monte_carlo_null(
    X_train_raw: pd.DataFrame,
    y_train_raw: pd.Series,
    X_eval_raw: pd.DataFrame,
    y_eval_raw: pd.Series,
    k: int,
    n_simulations: int = 500,
    seed: int = 42
):
    """
    Simulates N random draws of size k from the ADR pool.
    For each draw, fits NNLS weights on training data, then computes
    out-of-sample tracking error on evaluation data.
    
    Returns:
        dict: {'p5': 5th percentile, 'median': 50th percentile, 'p95': 95th percentile, 'all_te': list}
    """
    rng = np.random.RandomState(seed)
    candidates = list(X_train_raw.columns)
    all_te = []
    
    y_train = y_train_raw.values
    y_eval = y_eval_raw.values
    
    for _ in range(n_simulations):
        sampled = rng.choice(candidates, size=k, replace=False)
        X_sub_tr = X_train_raw[sampled].values
        w, _ = nnls(X_sub_tr, y_train)
        if w.sum() > 0:
            w_norm = w / w.sum()
        else:
            w_norm = np.ones(k) / k
            
        port_ret = X_eval_raw[sampled].values @ w_norm
        te = np.std(y_eval - port_ret, ddof=1) * np.sqrt(252) * 100.0
        all_te.append(te)
        
    all_te = np.array(all_te)
    return {
        'k': k,
        'p5': float(np.percentile(all_te, 5)),
        'median': float(np.percentile(all_te, 50)),
        'p95': float(np.percentile(all_te, 95)),
        'mean': float(np.mean(all_te)),
        'std': float(np.std(all_te))
    }
