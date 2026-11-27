"""
src/pareto.py
-------------
Pareto Frontier Cardinality Engine & Economic Cost Optimization Suite.
Constructs the full empirical trade-off boundary:
Cardinality (K) vs. Out-of-Sample Tracking Error (TE).
Computes:
1. Model comparisons against Monte Carlo Null Envelope.
2. Integrated Area Under the Curve (AUC-Frontier).
3. Economic Total Loss Minimization L(K) determining optimal K*.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.benchmarks import simulate_monte_carlo_null, compute_portfolio_te
from src.lasso_model import extract_portfolio_at_k
from src.autoencoder import select_ae_cluster_portfolio

def compute_auc(k_values, te_values):
    """Computes integrated Area Under the Curve (lower is better)."""
    return float(np.trapezoid(te_values, k_values))

def evaluate_pareto_frontier(
    X_train_raw: pd.DataFrame,
    y_train_raw: pd.Series,
    X_val_raw: pd.DataFrame,
    y_val_raw: pd.Series,
    X_test_raw: pd.DataFrame,
    y_test_raw: pd.Series,
    lasso_results_df: pd.DataFrame,
    ae_loadings: np.ndarray,
    stepwise_portfolios: dict,
    k_grid: list = None,
    mc_sims: int = 300
):
    """
    Sweeps cardinality K across models on the out-of-sample Test Vault (2022-2024).
    
    Returns:
        pd.DataFrame: Pareto frontier summary with columns for all models.
    """
    if k_grid is None:
        k_grid = [4, 6, 8, 10, 12, 15, 18, 20, 25, 30, 35]
        
    feature_names = X_train_raw.columns
    frontier_rows = []
    
    print(f"\n[+] Mapping Pareto Frontier across Cardinality K in {k_grid}...")
    
    for k in k_grid:
        # 1. Lasso
        lasso_port, _, _ = extract_portfolio_at_k(lasso_results_df, target_k=k, feature_names=feature_names)
        lasso_te = compute_portfolio_te(lasso_port, X_test_raw, y_test_raw)
        
        # 2. Autoencoder Clustering
        ae_port = select_ae_cluster_portfolio(ae_loadings, feature_names, X_train_raw, y_train_raw, target_k=k)
        ae_te = compute_portfolio_te(ae_port, X_test_raw, y_test_raw)
        
        # 3. Stepwise
        if k in stepwise_portfolios:
            step_port = stepwise_portfolios[k]
        else:
            closest_k = min(stepwise_portfolios.keys(), key=lambda x: abs(x - k))
            step_port = stepwise_portfolios[closest_k]
        step_te = compute_portfolio_te(step_port, X_test_raw, y_test_raw)
        
        # 4. Monte Carlo Random Null
        mc_res = simulate_monte_carlo_null(
            X_train_raw, y_train_raw, X_test_raw, y_test_raw,
            k=k, n_simulations=mc_sims, seed=42 + k
        )
        
        frontier_rows.append({
            'K': k,
            'Lasso_TE': lasso_te,
            'AE_TE': ae_te,
            'Stepwise_TE': step_te,
            'MC_p5': mc_res['p5'],
            'MC_median': mc_res['median'],
            'MC_p95': mc_res['p95']
        })
        print(f"    K={k:>2} | Lasso: {lasso_te:5.2f}% | AE-Clust: {ae_te:5.2f}% | Stepwise: {step_te:5.2f}% | MC Null: {mc_res['median']:5.2f}%")
        
    frontier_df = pd.DataFrame(frontier_rows)
    return frontier_df

def compute_economic_loss_curve(frontier_df, lambda_friction: float = 0.15, fixed_custody_bps_per_asset: float = 0.08):
    """
    Computes Institutional Total Cost of Ownership:
    L(K) = Tracking_Error(K) + lambda_friction * (K * fixed_custody_bps_per_asset)
    Finds the optimal K* where marginal error reduction equals holding friction.
    """
    df = frontier_df.copy()
    friction_cost = df['K'] * fixed_custody_bps_per_asset * lambda_friction
    
    df['Lasso_Loss'] = df['Lasso_TE'] + friction_cost
    df['AE_Loss'] = df['AE_TE'] + friction_cost
    df['Stepwise_Loss'] = df['Stepwise_TE'] + friction_cost
    
    # Locate optimal K*
    opt_k = {
        'Lasso_K_star': int(df.loc[df['Lasso_Loss'].idxmin(), 'K']),
        'Lasso_Min_Loss': float(df['Lasso_Loss'].min()),
        'AE_K_star': int(df.loc[df['AE_Loss'].idxmin(), 'K']),
        'AE_Min_Loss': float(df['AE_Loss'].min()),
        'Stepwise_K_star': int(df.loc[df['Stepwise_Loss'].idxmin(), 'K']),
        'Stepwise_Min_Loss': float(df['Stepwise_Loss'].min())
    }
    return df, opt_k

def plot_pareto_frontier(frontier_df, mc_sims: int = 300, save_path="results/adr_pareto_frontier.png"):
    """
    Plots the empirical Pareto Frontier (K vs. Tracking Error) with the Monte Carlo null envelope.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(11, 6.5))
    
    k = frontier_df['K']
    
    # Monte Carlo Null Envelope (shaded region)
    plt.fill_between(k, frontier_df['MC_p5'], frontier_df['MC_p95'], color='gray', alpha=0.18, label=f'Monte Carlo Null Envelope (90% CI, N={mc_sims})')
    plt.plot(k, frontier_df['MC_median'], color='gray', linestyle='--', linewidth=1.5, label='Random Subset Median Null')
    
    # Model curves
    plt.plot(k, frontier_df['Lasso_TE'], marker='o', color='#1f77b4', linewidth=2.2, label='Lasso (L1 Sparse Optimizer)')
    plt.plot(k, frontier_df['AE_TE'], marker='s', color='#2ca02c', linewidth=2.0, label='Autoencoder (Factor-Proportional Quota)')
    plt.plot(k, frontier_df['Stepwise_TE'], marker='^', color='#ff7f0e', linewidth=2.0, label='Stepwise (Greedy Forward Selection)')
    
    plt.title('Cross-Market S&P 500 Replication: Cardinality Pareto Frontier\nOut-of-Sample Test Vault (2022-2024)', fontsize=13, fontweight='bold')
    plt.xlabel('Portfolio Cardinality (Number of Foreign ADRs, K)', fontsize=11)
    plt.ylabel('Annualized Tracking Error vs. ^SP500TR (%)', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[+] Saved Pareto frontier plot to {save_path}")

def plot_economic_cost_curve(loss_df, opt_k, save_path="results/economic_cost_curve.png"):
    """
    Visualizes the institutional trade-off: Tracking Error Risk vs. Holding Friction.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    k = loss_df['K']
    plt.plot(k, loss_df['Lasso_Loss'], marker='o', color='#1f77b4', linewidth=2.2, label=f'Lasso Total Loss (Optimal K*={opt_k["Lasso_K_star"]})')
    plt.plot(k, loss_df['AE_Loss'], marker='s', color='#2ca02c', linewidth=2.0, label=f'Autoencoder Total Loss (Optimal K*={opt_k["AE_K_star"]})')
    plt.plot(k, loss_df['Stepwise_Loss'], marker='^', color='#ff7f0e', linewidth=2.0, label=f'Stepwise Total Loss (Optimal K*={opt_k["Stepwise_K_star"]})')
    
    # Mark optimums
    plt.axvline(opt_k['Lasso_K_star'], color='#1f77b4', linestyle=':', alpha=0.7)
    plt.axvline(opt_k['AE_K_star'], color='#2ca02c', linestyle=':', alpha=0.7)
    
    plt.title('Institutional Total Cost of Ownership Curve L(K)\nBalancing Basis Risk (TE) vs. Foreign Custody & Friction Drag', fontsize=12, fontweight='bold')
    plt.xlabel('Portfolio Cardinality (K)', fontsize=11)
    plt.ylabel('Total Cost Objective Metric L(K)', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[+] Saved economic cost curve to {save_path}")
