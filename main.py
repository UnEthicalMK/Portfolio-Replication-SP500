"""
main.py
-------
The Master Orchestrator for the Cross-Market Synthetic S&P 500 Replication Engine.
Executes the data pipeline, trains Lasso, Autoencoder, and Stepwise models,
evaluates the Cardinality Pareto Frontier, and runs the 3-year multi-wave (2022-2024) shootout
comparing Buy-and-Hold vs. Quarterly Rebalancing with Friction across ALL models against S&P 500 TR.
"""

import os
import pandas as pd
import numpy as np

from src.data_pipeline import load_or_build_pipeline, BENCHMARK_TICKER
from src.features import chronological_split, scale_features
from src.lasso_model import sweep_alpha, extract_portfolio_at_k, plot_regularization_path
from src.autoencoder import train_macro_autoencoder, extract_factor_loadings, select_ae_cluster_portfolio, plot_latent_factors
from src.stepwise import run_forward_stepwise
from src.benchmarks import build_equal_weight_benchmark
from src.pareto import evaluate_pareto_frontier, compute_economic_loss_curve, compute_auc, plot_pareto_frontier, plot_economic_cost_curve
from src.backtest import simulate_portfolio, plot_multi_year_shootout, plot_rebalancing_impact

def print_section(title):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)

def main():
    print_section("CROSS-MARKET SYNTHETIC S&P 500 CLONING ENGINE")
    print("Research Question: Can we clone S&P 500 Total Return without domestic stocks,")
    print("                   using only liquid foreign ADRs traded on US exchanges?")
    
    # ---------------------------------------------------------
    # PHASE 1: Data Pipeline & Temporal Partitioning
    # ---------------------------------------------------------
    print_section("PHASE 1: Data Acquisition & Preprocessing")
    log_returns = load_or_build_pipeline(benchmark=BENCHMARK_TICKER)
    
    X_train, y_train, X_val, y_val, X_test, y_test = chronological_split(
        log_returns, target_col=BENCHMARK_TICKER
    )
    X_train_s, X_val_s, X_test_s, scaler = scale_features(X_train, X_val, X_test)
    feature_names = X_train.columns
    
    # ---------------------------------------------------------
    # PHASE 2: Model Training & Sparsity Sweeps
    # ---------------------------------------------------------
    print_section("PHASE 2: Model Fitting & Representation Learning")
    
    # 1. Lasso L1 Sparse Regularization Sweep
    print("\n>>> Fitting Lasso L1 Regularization Path (100 alphas)...")
    lasso_results = sweep_alpha(X_train_s, y_train, X_val, y_val, scaler)
    plot_regularization_path(lasso_results, save_path="results/regularization_path.png")
    
    # 2. Nonlinear Latent Factor Autoencoder
    print("\n>>> Training 8-Factor Bottleneck Autoencoder...")
    ae_model = train_macro_autoencoder(X_train_s, X_val_s, n_factors=8, epochs=120)
    ae_loadings, Z = extract_factor_loadings(ae_model, X_train_s)
    plot_latent_factors(Z, X_train.index, save_path="results/autoencoder_factors.png")
    
    # 3. Greedy Forward Stepwise Selection
    print("\n>>> Running Combinatorial Forward Stepwise Selection (k=1..35)...")
    stepwise_portfolios = run_forward_stepwise(X_train, y_train, max_k=35)
    
    # ---------------------------------------------------------
    # PHASE 3: Pareto Frontier & Economic Breakeven
    # ---------------------------------------------------------
    print_section("PHASE 3: Cardinality Pareto Frontier & Economic Loss Optimization")
    
    k_grid = [4, 6, 8, 10, 12, 15, 18, 20, 25, 30, 35]
    frontier_df = evaluate_pareto_frontier(
        X_train, y_train, X_val, y_val, X_test, y_test,
        lasso_results_df=lasso_results,
        ae_loadings=ae_loadings,
        stepwise_portfolios=stepwise_portfolios,
        k_grid=k_grid,
        mc_sims=300
    )
    
    # Compute Area Under the Curve (AUC)
    k_vals = frontier_df['K'].values
    lasso_auc = compute_auc(k_vals, frontier_df['Lasso_TE'].values)
    ae_auc = compute_auc(k_vals, frontier_df['AE_TE'].values)
    step_auc = compute_auc(k_vals, frontier_df['Stepwise_TE'].values)
    mc_auc = compute_auc(k_vals, frontier_df['MC_median'].values)
    
    print("\n[+] Global Frontier Dimensionality Compression (Integrated AUC - lower is better):")
    print(f"    Lasso AUC:            {lasso_auc:6.1f}")
    print(f"    Stepwise AUC:         {step_auc:6.1f}")
    print(f"    Autoencoder AUC:      {ae_auc:6.1f}")
    print(f"    Monte Carlo Null AUC: {mc_auc:6.1f}")
    
    # Economic Total Cost Function
    loss_df, opt_k = compute_economic_loss_curve(frontier_df, lambda_friction=0.15)
    print("\n[+] Economically Optimal Portfolio Cardinality K* (Balancing TE vs. Friction):")
    print(f"    Lasso Optimal K*:     {opt_k['Lasso_K_star']:>2} ADRs (Total Loss: {opt_k['Lasso_Min_Loss']:.2f})")
    print(f"    Stepwise Optimal K*:  {opt_k['Stepwise_K_star']:>2} ADRs (Total Loss: {opt_k['Stepwise_Min_Loss']:.2f})")
    print(f"    Autoencoder Opt K*:   {opt_k['AE_K_star']:>2} ADRs (Total Loss: {opt_k['AE_Min_Loss']:.2f})")
    
    plot_pareto_frontier(frontier_df, mc_sims=300, save_path="results/adr_pareto_frontier.png")
    plot_economic_cost_curve(loss_df, opt_k, save_path="results/economic_cost_curve.png")
    
    # ---------------------------------------------------------
    # PHASE 4: Multi-Wave Out-of-Sample Backtest (2022-2024)
    # Buy & Hold vs. Quarterly Rebalancing for ALL Models
    # ---------------------------------------------------------
    print_section("PHASE 4: 3-Year Test Vault Backtest (2022-2024 Stress Test)")
    
    k_test = 15
    print(f"\n[+] Extracting Portfolios at Standardized Balanced Tier (K={k_test}):")
    
    lasso_port_15, _, _ = extract_portfolio_at_k(lasso_results, target_k=k_test, feature_names=feature_names)
    step_port_15 = stepwise_portfolios[k_test]
    ae_port_15 = select_ae_cluster_portfolio(ae_loadings, feature_names, X_train, y_train, target_k=k_test)
    ew_port_15 = build_equal_weight_benchmark(feature_names, n_assets=k_test)
    
    models_dict = {
        "Lasso": lasso_port_15,
        "Stepwise": step_port_15,
        "Autoencoder": ae_port_15,
        "Equal-Weight": ew_port_15
    }
    
    test_combined = X_test.join(y_test)
    
    # Run physical simulations for BOTH Mode A (Buy & Hold) and Mode B (Quarterly) for ALL models
    audit_rows = []
    equity_curves_bnh = {}
    equity_curves_qtr = {}
    benchmark_equity_series = None
    sub_lasso_holder = None
    
    for name, port in models_dict.items():
        # Mode A: Buy & Hold Drift
        eq_bnh, m_bnh, sub_bnh = simulate_portfolio(port, test_combined, benchmark_col=BENCHMARK_TICKER, rebalance_freq="none")
        equity_curves_bnh[f"{name} (Drift)"] = eq_bnh['Portfolio']
        if benchmark_equity_series is None:
            benchmark_equity_series = eq_bnh['Benchmark']
        if name == "Lasso":
            sub_lasso_holder = sub_bnh
            
        audit_rows.append({
            "Model": name,
            "Mode": "Buy & Hold",
            "Return (%)": m_bnh["Total Return (%)"],
            "Tracking Error (%)": m_bnh["Tracking Error (%)"],
            "Information Ratio": m_bnh["Information Ratio"],
            "Max Drawdown (%)": m_bnh["Max Drawdown (%)"],
            "Correlation": m_bnh["Correlation"],
            "Beta": m_bnh["Beta"],
            "Rebalance Events": m_bnh["Rebalance Events"],
            "Turnover ($)": m_bnh["Turnover Volume Traded ($)"],
            "Total Friction ($)": m_bnh["Total Friction Paid ($)"]
        })
        
        # Mode B: Quarterly Rebalancing
        eq_qtr, m_qtr, _ = simulate_portfolio(port, test_combined, benchmark_col=BENCHMARK_TICKER, rebalance_freq="quarterly")
        equity_curves_qtr[f"{name} (Quarterly)"] = eq_qtr['Portfolio']
        
        audit_rows.append({
            "Model": name,
            "Mode": "Quarterly Rebalance",
            "Return (%)": m_qtr["Total Return (%)"],
            "Tracking Error (%)": m_qtr["Tracking Error (%)"],
            "Information Ratio": m_qtr["Information Ratio"],
            "Max Drawdown (%)": m_qtr["Max Drawdown (%)"],
            "Correlation": m_qtr["Correlation"],
            "Beta": m_qtr["Beta"],
            "Rebalance Events": m_qtr["Rebalance Events"],
            "Turnover ($)": m_qtr["Turnover Volume Traded ($)"],
            "Total Friction ($)": m_qtr["Total Friction Paid ($)"]
        })
        
    audit_df = pd.DataFrame(audit_rows)
    
    # Print Master Comparison Table
    print("\n" + "=" * 105)
    print(f" 3-YEAR MULTI-WAVE (2022-2024) AUDIT: BUY-AND-HOLD VS. QUARTERLY REBALANCING WITH COSTS (K={k_test})")
    print("=" * 105)
    print(f"{'Model':<13} {'Rebalance Mode':<20} {'Return':>9} {'TE (%)':>8} {'IR':>7} {'MDD (%)':>9} {'Corr':>7} {'Beta':>6} {'Turnover ($)':>13} {'Friction ($)':>12}")
    print("-" * 105)
    for _, r in audit_df.iterrows():
        print(f"{r['Model']:<13} {r['Mode']:<20} {r['Return (%)']:>8.2f}% {r['Tracking Error (%)']:>7.2f}% {r['Information Ratio']:>7.2f} {r['Max Drawdown (%)']:>8.2f}% {r['Correlation']:>7.4f} {r['Beta']:>6.2f} ${r['Turnover ($)']:>11,.0f} ${r['Total Friction ($)']:>10,.0f}")
    print("-" * 105)
    
    bench_cum = (benchmark_equity_series.iloc[-1] / 1_000_000.0 - 1.0) * 100.0
    bench_mdd = float(((benchmark_equity_series - benchmark_equity_series.cummax()) / benchmark_equity_series.cummax()).min() * 100.0)
    print(f"{'S&P 500 TR':<13} {'Benchmark Target':<20} {bench_cum:>8.2f}% {0.0:>7.2f}% {0.0:>7.2f} {bench_mdd:>8.2f}% {1.0000:>7.4f} {1.00:>6.2f} {'$0':>13} {'$0':>12}")
    print("=" * 105)
    
    # Sub-Period Breakdown for Lasso
    print("\n[+] Regime Sub-Period Tracking Error Breakdown (Lasso Mode A):")
    for yr in [2022, 2023, 2024]:
        lbl = {2022: "2022 (Rate Shock & Bear Market)", 2023: "2023 (Mega-Cap AI Rally)", 2024: "2024 (Broad-Market Expansion)"}[yr]
        print(f"    {lbl:<32}: Port: {sub_lasso_holder[yr]['Portfolio Return (%)']:>6.1f}% | S&P 500: {sub_lasso_holder[yr]['Benchmark Return (%)']:>6.1f}% | TE: {sub_lasso_holder[yr]['Tracking Error (%)']:.2f}%")
        
    print(f"\n[+] Top 5 Capital Allocations (Lasso K={k_test}):")
    for _, row in lasso_port_15.head(5).iterrows():
        print(f"    {row['Ticker']:<6} -> {row['Weight']*100:5.2f}%")
        
    # Generate Shootout Charts - Separate Image for Buy & Hold and Separate Image for Quarterly Rebalancing
    # 1. Buy & Hold Shootout (All 4 Models vs. S&P 500 TR)
    bnh_plot_map = {
        f"Lasso Clone (K={k_test}, Drift)": equity_curves_bnh["Lasso (Drift)"],
        f"Stepwise Selection (K={k_test}, Drift)": equity_curves_bnh["Stepwise (Drift)"],
        f"Autoencoder Quota (K={k_test}, Drift)": equity_curves_bnh["Autoencoder (Drift)"],
        f"Equal-Weight ADRs (K={k_test}, Drift)": equity_curves_bnh["Equal-Weight (Drift)"]
    }
    plot_multi_year_shootout(
        results_map=bnh_plot_map,
        benchmark_equity=benchmark_equity_series,
        title=f"3-Year Multi-Wave Performance: Mode A - Buy & Hold (Natural Price Drift, K={k_test})\n4 Models vs. S&P 500 Total Return Index (^SP500TR)",
        save_path="results/shootout_buy_and_hold.png"
    )
    
    # 2. Quarterly Rebalanced Shootout (All 4 Models vs. S&P 500 TR)
    qtr_plot_map = {
        f"Lasso Clone (K={k_test}, Quarterly Rebalanced)": equity_curves_qtr["Lasso (Quarterly)"],
        f"Stepwise Selection (K={k_test}, Quarterly Rebalanced)": equity_curves_qtr["Stepwise (Quarterly)"],
        f"Autoencoder Quota (K={k_test}, Quarterly Rebalanced)": equity_curves_qtr["Autoencoder (Quarterly)"],
        f"Equal-Weight ADRs (K={k_test}, Quarterly Rebalanced)": equity_curves_qtr["Equal-Weight (Quarterly)"]
    }
    plot_multi_year_shootout(
        results_map=qtr_plot_map,
        benchmark_equity=benchmark_equity_series,
        title=f"3-Year Multi-Wave Performance: Mode B - Quarterly Rebalanced with Fees (K={k_test})\n4 Models vs. S&P 500 Total Return Index (^SP500TR)",
        save_path="results/shootout_quarterly_rebalance.png"
    )
    
    # 3. Master Combined Shootout Overview
    all_equity_curves = {}
    for k_name, v_series in equity_curves_bnh.items():
        all_equity_curves[k_name] = v_series
    for k_name, v_series in equity_curves_qtr.items():
        if "Lasso" in k_name or "Stepwise" in k_name:
            all_equity_curves[k_name] = v_series
            
    plot_multi_year_shootout(
        results_map=all_equity_curves,
        benchmark_equity=benchmark_equity_series,
        title=f"3-Year Multi-Wave Performance: Drift vs. Quarterly Rebalancing Combined (K={k_test})\nTop Models vs. S&P 500 Total Return Index (^SP500TR)",
        save_path="results/multi_year_shootout.png"
    )
    
    # 4. Dedicated Bar Chart analyzing Rebalancing Impact and Friction Costs
    plot_rebalancing_impact(
        comparison_df=audit_df,
        save_path="results/rebalancing_impact.png"
    )
    
    print_section("EXECUTION COMPLETE - ARTIFACTS GENERATED IN results/")
    print("  • results/shootout_buy_and_hold.png          (Mode A: 4 Models vs. S&P 500)")
    print("  • results/shootout_quarterly_rebalance.png    (Mode B: 4 Models vs. S&P 500)")
    print("  • results/adr_pareto_frontier.png             (K vs. Tracking Error with MC Null)")
    print("  • results/rebalancing_impact.png              (Tracking Error vs. Friction Paid)")
    print("  • results/economic_cost_curve.png             (Total Cost of Ownership L(K))")
    print("  • results/multi_year_shootout.png             (Combined Multi-Wave Trajectory)")
    print("  • results/regularization_path.png             (Lasso Alpha Sparsity Sweep)")
    print("  • results/autoencoder_factors.png             (8 Latent Macro Factor Trajectories)")

if __name__ == "__main__":
    main()
