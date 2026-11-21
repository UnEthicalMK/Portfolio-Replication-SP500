"""
src/backtest.py
---------------
Institutional Multi-Wave Out-of-Sample Simulation Engine (2022-2024).
Simulates physical capital deployment under:
- Mode A: Multi-Year Buy-and-Hold with Daily Price Drift & Entry Friction
- Mode B: Institutional Quarterly Rebalancing with Turnover Drag & Execution Fees
Computes comprehensive risk metrics, turnover analytics, and regime breakdowns for ALL models.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_max_drawdown(equity_series: pd.Series) -> float:
    """Computes maximum peak-to-trough decline as a percentage."""
    running_max = equity_series.cummax()
    drawdowns = (equity_series - running_max) / running_max
    return float(drawdowns.min() * 100.0)

def simulate_portfolio(
    portfolio_weights: pd.DataFrame,
    test_log_returns: pd.DataFrame,
    benchmark_col: str = "^SP500TR",
    initial_capital: float = 1_000_000.0,
    fee_bps: float = 10.0,
    rebalance_freq: str = "none"  # "none" for Mode A (Buy & Hold), "quarterly" for Mode B
):
    """
    Simulates portfolio equity trajectory and computes institutional metrics.
    
    Parameters:
        portfolio_weights (pd.DataFrame): 'Ticker' and 'Weight'.
        test_log_returns (pd.DataFrame): Out-of-sample log returns including benchmark.
        benchmark_col (str): Target benchmark.
        initial_capital (float): Starting dollar portfolio value.
        fee_bps (float): Transaction cost in basis points (10 bps = 0.10%).
        rebalance_freq (str): "none" (Buy & Hold) or "quarterly".
        
    Returns:
        tuple: (equity_df, metrics_dict, sub_period_dict)
    """
    # 1. Convert log returns to simple returns for linear portfolio accounting
    simple_returns = np.exp(test_log_returns) - 1.0
    dates = test_log_returns.index
    
    benchmark_simple = simple_returns[benchmark_col]
    tickers = portfolio_weights['Ticker'].tolist()
    target_weights = dict(zip(tickers, portfolio_weights['Weight'].values))
    
    # Missing ticker defense
    for t in tickers:
        if t not in simple_returns.columns:
            raise KeyError(f"Selected ticker '{t}' missing from test dataset!")
            
    asset_returns = simple_returns[tickers]
    
    # 2. Capital deployment on Day 1 after entry transaction fee
    fee_pct = fee_bps / 10000.0
    entry_fee_dollars = initial_capital * fee_pct
    starting_capital = initial_capital - entry_fee_dollars
    
    # Initialize dollar positions
    current_positions = {t: starting_capital * target_weights[t] for t in tickers}
    portfolio_values = []
    daily_returns = []
    
    # Track turnover metrics for Mode B
    total_rebalance_turnover_dollars = 0.0
    total_rebalance_fees = 0.0
    rebalance_events = 0
    
    for i, date in enumerate(dates):
        # Grow positions by daily simple return
        daily_asset_ret = asset_returns.loc[date]
        day_end_positions = {t: current_positions[t] * (1.0 + daily_asset_ret[t]) for t in tickers}
        day_end_total = sum(day_end_positions.values())
        
        # Mode B: Quarterly Rebalancing event check (last trading day of calendar quarter)
        is_quarter_end = (rebalance_freq == "quarterly") and (
            i < len(dates) - 1 and dates[i+1].quarter != date.quarter
        )
        
        if is_quarter_end:
            rebalance_events += 1
            # Current drifted weights
            drift_weights = {t: day_end_positions[t] / day_end_total for t in tickers}
            # Turnover fraction = 0.5 * sum(|w_target - w_current|)
            turnover_fraction = sum(abs(target_weights[t] - drift_weights[t]) for t in tickers) * 0.5
            turnover_traded_dollars = day_end_total * turnover_fraction
            turnover_fee = turnover_traded_dollars * fee_pct
            
            total_rebalance_turnover_dollars += turnover_traded_dollars
            total_rebalance_fees += turnover_fee
            day_end_total -= turnover_fee
            
            # Reset positions to exact target weights
            current_positions = {t: day_end_total * target_weights[t] for t in tickers}
        else:
            current_positions = day_end_positions
            
        portfolio_values.append(day_end_total)
        
        # Daily portfolio return
        if i == 0:
            daily_returns.append((day_end_total / initial_capital) - 1.0)
        else:
            daily_returns.append((day_end_total / portfolio_values[i-1]) - 1.0)
            
    # Series of portfolio and benchmark cumulative equity
    port_equity = pd.Series(portfolio_values, index=dates)
    port_cum_ret = (port_equity / initial_capital) - 1.0
    benchmark_cum_ret = (1.0 + benchmark_simple).cumprod() - 1.0
    
    port_ret_series = pd.Series(daily_returns, index=dates)
    bench_ret_series = benchmark_simple
    
    # 3. Overall Institutional Metrics
    return_diff = port_ret_series - bench_ret_series
    tracking_error = float(np.std(return_diff, ddof=1) * np.sqrt(252) * 100.0)
    
    mean_excess = float(return_diff.mean() * 252 * 100.0)
    info_ratio = float(mean_excess / tracking_error) if tracking_error > 0 else 0.0
    
    port_mdd = compute_max_drawdown(port_equity)
    bench_equity = (1.0 + benchmark_simple).cumprod() * initial_capital
    bench_mdd = compute_max_drawdown(bench_equity)
    
    corr = float(port_ret_series.corr(bench_ret_series))
    cov_pm = float(np.cov(port_ret_series, bench_ret_series)[0, 1])
    var_m = float(np.var(bench_ret_series, ddof=1))
    beta = float(cov_pm / var_m) if var_m > 0 else 1.0
    alpha = float(((port_ret_series.mean() - beta * bench_ret_series.mean()) * 252) * 100.0)
    
    total_all_friction_dollars = entry_fee_dollars + total_rebalance_fees
    years_simulated = len(dates) / 252.0
    annual_turnover_rate = float((total_rebalance_turnover_dollars / initial_capital) / years_simulated * 100.0) if years_simulated > 0 else 0.0
    
    metrics = {
        "Total Return (%)": float(port_cum_ret.iloc[-1] * 100.0),
        "Benchmark Return (%)": float(benchmark_cum_ret.iloc[-1] * 100.0),
        "Tracking Error (%)": tracking_error,
        "Information Ratio": info_ratio,
        "Max Drawdown (%)": port_mdd,
        "Benchmark MDD (%)": bench_mdd,
        "Correlation": corr,
        "Beta": beta,
        "Jensen Alpha (%)": alpha,
        "Rebalance Events": rebalance_events,
        "Turnover Volume Traded ($)": float(total_rebalance_turnover_dollars),
        "Rebalance Fees Paid ($)": float(total_rebalance_fees),
        "Total Friction Paid ($)": float(total_all_friction_dollars),
        "Annual Turnover Rate (%)": annual_turnover_rate
    }
    
    # 4. Regime Sub-Period Breakdowns: 2022 (Bear), 2023 (Tech Surge), 2024 (Expansion)
    sub_periods = {}
    for year in [2022, 2023, 2024]:
        yr_dates = dates[dates.year == year]
        if len(yr_dates) > 0:
            yr_port_ret = port_ret_series.loc[yr_dates]
            yr_bench_ret = bench_ret_series.loc[yr_dates]
            yr_diff = yr_port_ret - yr_bench_ret
            yr_te = float(np.std(yr_diff, ddof=1) * np.sqrt(252) * 100.0)
            yr_port_tot = float(((1.0 + yr_port_ret).cumprod().iloc[-1] - 1.0) * 100.0)
            yr_bench_tot = float(((1.0 + yr_bench_ret).cumprod().iloc[-1] - 1.0) * 100.0)
            sub_periods[year] = {
                "Portfolio Return (%)": yr_port_tot,
                "Benchmark Return (%)": yr_bench_tot,
                "Tracking Error (%)": yr_te
            }
            
    equity_df = pd.DataFrame({
        "Portfolio": port_equity,
        "Benchmark": bench_equity,
        "Portfolio_Return_Cum": port_cum_ret,
        "Benchmark_Return_Cum": benchmark_cum_ret
    }, index=dates)
    
    return equity_df, metrics, sub_periods

def plot_multi_year_shootout(
    results_map: dict,
    benchmark_equity: pd.Series,
    initial_capital: float = 1_000_000.0,
    title: str = "3-Year Multi-Wave Performance Shootout (2022-2024)\nCross-Market Foreign ADR Replication vs. S&P 500 Total Return",
    save_path: str = "results/multi_year_shootout.png"
):
    """
    Plots cumulative wealth curves comparing multiple models against S&P 500 Total Return.
    Can be used for Buy-and-Hold Shootout or Quarterly Rebalancing Shootout.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(14, 7.5))
    
    # Benchmark in bold black
    bench_cum = (benchmark_equity / initial_capital) - 1.0
    plt.plot(bench_cum.index, bench_cum * 100, label='S&P 500 TR (^SP500TR)', color='black', linewidth=3.0, zorder=10)
    
    # Dedicated color palette for the 4 models
    styles = {
        'Lasso': {'color': '#1f77b4'},
        'Stepwise': {'color': '#ff7f0e'},
        'Autoencoder': {'color': '#2ca02c'},
        'Equal-Weight': {'color': '#9467bd'}
    }
    
    for label, eq_series in results_map.items():
        base_color = '#7f7f7f'
        for k, v in styles.items():
            if k in label:
                base_color = v['color']
                break
                
        cum_ret = (eq_series / initial_capital) - 1.0
        plt.plot(cum_ret.index, cum_ret * 100, label=label, color=base_color, linewidth=2.2, alpha=0.9)
        
    plt.title(title, fontsize=13, fontweight='bold')
    plt.ylabel('Cumulative Return (%)', fontsize=11)
    plt.xlabel('Date', fontsize=11)
    
    # Highlight sub-periods with shaded spans
    plt.axvspan(pd.to_datetime('2022-01-01'), pd.to_datetime('2022-12-31'), color='gray', alpha=0.10, label='2022 Bear Market')
    plt.axvspan(pd.to_datetime('2023-01-01'), pd.to_datetime('2023-12-31'), color='blue', alpha=0.06, label='2023 AI Surge')
    plt.axvspan(pd.to_datetime('2024-01-01'), pd.to_datetime('2024-12-31'), color='green', alpha=0.06, label='2024 Expansion')
    
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', framealpha=0.9, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[+] Saved shootout chart to {save_path}")

def plot_rebalancing_impact(
    comparison_df: pd.DataFrame,
    save_path: str = "results/rebalancing_impact.png"
):
    """
    Bar chart comparing Buy-and-Hold vs. Quarterly Rebalancing across all models:
    Tracking Error reduction vs. Total Friction cost paid.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    models = comparison_df['Model'].unique()
    x = np.arange(len(models))
    width = 0.35
    
    bnh_te = [comparison_df[(comparison_df['Model'] == m) & (comparison_df['Mode'] == 'Buy & Hold')]['Tracking Error (%)'].values[0] for m in models]
    qtr_te = [comparison_df[(comparison_df['Model'] == m) & (comparison_df['Mode'] == 'Quarterly Rebalance')]['Tracking Error (%)'].values[0] for m in models]
    
    ax1.bar(x - width/2, bnh_te, width, label='Buy & Hold (Drift)', color='#1f77b4', alpha=0.85)
    ax1.bar(x + width/2, qtr_te, width, label='Quarterly Rebalanced', color='#2ca02c', alpha=0.85)
    ax1.set_ylabel('Annualized Tracking Error (%)', fontsize=11)
    ax1.set_title('Tracking Error: Drift vs. Quarterly Rebalance', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend()
    
    # Friction cost in dollars
    fees_paid = [comparison_df[(comparison_df['Model'] == m) & (comparison_df['Mode'] == 'Quarterly Rebalance')]['Total Friction ($)'].values[0] for m in models]
    bars = ax2.bar(models, fees_paid, color='#d62728', alpha=0.8, width=0.45)
    ax2.set_ylabel('Total Transaction & Rebalance Costs ($)', fontsize=11)
    ax2.set_title('Cumulative Friction Paid Over 3 Years ($1M Account)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 100, f"${yval:,.0f}", ha='center', va='bottom', fontsize=9.5, fontweight='bold')
        
    plt.suptitle('Economic Friction Audit: Buy & Hold vs. Quarterly Rebalancing Across All Models', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[+] Saved rebalancing impact plot to {save_path}")
