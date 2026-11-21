"""
src/lasso_model.py
------------------
L1-regularized sparse cross-market tracking optimizer.
Fits Lasso paths on scaled features and computes tracking error
in raw, unscaled return space to ensure mathematical validity.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

def calculate_tracking_error(y_true, y_pred):
    """
    Computes annualized tracking error in percentage points:
    TE = std(y_true - y_pred, ddof=1) * sqrt(252) * 100
    """
    diff = np.asarray(y_true) - np.asarray(y_pred)
    return np.std(diff, ddof=1) * np.sqrt(252) * 100.0

def sweep_alpha(X_train_scaled, y_train, X_val_raw, y_val_raw, scaler, alphas=None):
    """
    Sweeps Lasso penalty across a wide spectrum (100 log-spaced points).
    Strict Mathematical Discipline:
    - Trains on X_train_scaled without intercept (fit_intercept=False).
    - Unscales coefficients: beta_raw = beta_scaled / feature_std.
    - Evaluates tracking error on UNTRANSFORMED X_val_raw and y_val_raw.
    """
    if alphas is None:
        alphas = np.logspace(-6, -1, 100)
        
    results = []
    feature_names = X_train_scaled.columns
    
    for alpha in alphas:
        # positive=True enforces long-only capital deployment
        model = Lasso(
            alpha=alpha,
            positive=True,
            fit_intercept=False,
            max_iter=15000,
            random_state=42
        )
        model.fit(X_train_scaled.values, y_train.values)
        
        # Unscale coefficients to physical price space
        raw_coefs = model.coef_ / scaler.scale_
        active_mask = raw_coefs > 1e-6
        active_count = int(np.sum(active_mask))
        
        # Validation prediction in RAW space
        val_preds_raw = X_val_raw.values @ raw_coefs
        val_te_raw = calculate_tracking_error(y_val_raw.values, val_preds_raw)
        
        # Normalized portfolio validation tracking error (sum weights = 1.0)
        if active_count > 0:
            norm_weights = raw_coefs.copy()
            norm_weights[active_mask] = norm_weights[active_mask] / np.sum(norm_weights[active_mask])
            val_preds_norm = X_val_raw.values @ norm_weights
            val_te_norm = calculate_tracking_error(y_val_raw.values, val_preds_norm)
        else:
            norm_weights = raw_coefs
            val_te_norm = calculate_tracking_error(y_val_raw.values, np.zeros_like(y_val_raw.values))
            
        results.append({
            'alpha': alpha,
            'active_assets': active_count,
            'val_te_raw': val_te_raw,
            'val_te_norm': val_te_norm,
            'raw_coefs': raw_coefs,
            'norm_weights': norm_weights
        })
        
    results_df = pd.DataFrame(results)
    print(f"[+] Lasso Alpha Sweep completed ({len(alphas)} points): Active assets range {results_df['active_assets'].min()} -> {results_df['active_assets'].max()}.")
    return results_df

def extract_portfolio_at_k(results_df, target_k, feature_names):
    """
    Extracts the portfolio from the alpha sweep whose active asset count
    matches or is closest to target_k, minimizing validation tracking error.
    """
    exact_matches = results_df[results_df['active_assets'] == target_k]
    if not exact_matches.empty:
        best_row = exact_matches.loc[exact_matches['val_te_norm'].idxmin()]
    else:
        # Nearest cardinality
        dist = (results_df['active_assets'] - target_k).abs()
        min_dist = dist.min()
        candidates = results_df[dist == min_dist]
        best_row = candidates.loc[candidates['val_te_norm'].idxmin()]
        
    weights = best_row['norm_weights']
    mask = weights > 1e-6
    
    portfolio = pd.DataFrame({
        'Ticker': feature_names[mask],
        'Weight': weights[mask]
    }).sort_values(by='Weight', ascending=False).reset_index(drop=True)
    
    # Re-normalize exactly to 1.0
    portfolio['Weight'] = portfolio['Weight'] / portfolio['Weight'].sum()
    return portfolio, best_row['alpha'], best_row['val_te_norm']

def plot_regularization_path(results_df, save_path="results/regularization_path.png"):
    """
    Plots the Lasso regularization path showing active asset count
    and validation tracking error as a function of regularization alpha.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    
    ax1.plot(results_df['alpha'], results_df['val_te_norm'], color='#1f77b4', linewidth=2, label='Val Tracking Error (%)')
    ax1.set_xscale('log')
    ax1.set_ylabel('Annualized TE (%)', fontsize=11)
    ax1.set_title('Lasso Regularization Path: Cross-Market ADR Replication', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    ax2.plot(results_df['alpha'], results_df['active_assets'], color='#ff7f0e', linewidth=2, label='Active Foreign ADRs (K)')
    ax2.set_xscale('log')
    ax2.set_xlabel('Regularization Penalty (Alpha)', fontsize=11)
    ax2.set_ylabel('Number of Stocks (K)', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[+] Saved Lasso regularization path plot to {save_path}")
