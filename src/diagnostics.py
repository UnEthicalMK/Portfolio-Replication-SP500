import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

plt.style.use('seaborn-v0_8-whitegrid')

TARGET = '^SP500TR'

# =========================================================
# Quantitative Math Helpers
# =========================================================

def calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculates the maximum peak-to-trough drop in a return series."""
    cum_returns = np.exp(returns.cumsum())
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    return drawdown.min()

def compute_metrics(port_returns: pd.DataFrame, bench_returns: pd.Series, weights_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates institutional reporting metrics and exports to CSV."""
    print("Computing Final Summary Metrics...")
    
    metrics = {}
    
    for model in port_returns.columns:
        p_ret = port_returns[model]
        residuals = p_ret - bench_returns
        
        # 1. Annualised Tracking Error
        te = residuals.std() * np.sqrt(252)
        
        # 2. Cumulative Return 2023
        cum_ret = (1 + p_ret).prod() - 1
        
        # 3. Max Single-Day Deviation (absolute)
        max_dev = residuals.abs().max()
        
        # 4. Active Instrument Count (Weights > 0)
        active_count = (weights_df[model] > 1e-4).sum()
        
        # 5. Sharpe Ratio of Residuals
        resid_sharpe = (residuals.mean() / residuals.std()) * np.sqrt(252)
        
        metrics[model] = {
            "Tracking Error (Ann.)": f"{te:.4f}",
            "Cumulative Return": f"{cum_ret:.4f}",
            "Max Daily Deviation": f"{max_dev:.4f}",
            "Active Instruments": int(active_count),
            "Residual Sharpe": f"{resid_sharpe:.2f}"
        }
        
    bench_cum = (1 + bench_returns).prod() - 1
    
    df = pd.DataFrame(metrics)
    print("\n---------------------------------------------------------")
    print(f"Benchmark (^SP500TR) Cumulative Return: {bench_cum:.4f}")
    print("---------------------------------------------------------")
    print(df.to_string())
    print("---------------------------------------------------------\n")
    
    return df

# =========================================================
# Visualizations
# =========================================================

def plot_residual_distribution(port_returns: pd.DataFrame, bench_returns: pd.Series):
    """Plot 8: Visualises the tracking bias and tail risk."""
    print("Generating Plot 8: Residual Distribution...")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['steelblue', 'darkorange', 'seagreen']
    
    for (model, color) in zip(port_returns.columns, colors):
        residuals = port_returns[model] - bench_returns
        
        # Plot histogram
        ax.hist(residuals, bins=50, alpha=0.4, color=color, label=model, density=True)
        
        # Fit and plot a normal distribution over it
        mu, std = residuals.mean(), residuals.std()
        x = np.linspace(residuals.min(), residuals.max(), 200)
        ax.plot(x, norm.pdf(x, mu, std), color=color, linewidth=1.5)
        
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title("Residual Distribution — Portfolio minus S&P 500 (2023) (Plot 8)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Daily Residual")
    ax.set_ylabel("Density")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig("plots/08_residual_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()

# =========================================================
# Terminal Tear Sheet
# =========================================================

def print_performance_metrics(metrics_dict: dict):
    """Generates a strictly formatted, terminal-agnostic performance tear sheet."""
    col_1 = 18  # Metric Name
    col_2 = 14  # Lasso Clone
    col_3 = 14  # Autoencoder
    col_4 = 18  # Naive Benchmark
    col_5 = 12  # S&P 500
    
    print("\n" + "=" * 80)
    header = (f"{'Metric':<{col_1}}"
              f"{'Lasso (Simplex)':<{col_2}}"
              f"{'Autoencoder':<{col_3}}"
              f"{'Equal Weight':<{col_4}}"
              f"{'S&P 500':<{col_5}}")
    print(header)
    print("-" * 80)
    
    row_te = (f"{'Tracking Error':<{col_1}}"
              f"{metrics_dict['Lasso']['te'] * 100:<{col_2}.3f}%"
              f"{metrics_dict['SAE']['te'] * 100:<{col_3}.3f}%"
              f"{metrics_dict['Equal_Weight']['te'] * 100:<{col_4}.3f}%"
              f"{0.0:<{col_5}.2f}%")
    print(row_te)
    
    row_corr = (f"{'Correlation':<{col_1}}"
                f"{metrics_dict['Lasso']['corr']:<{col_2}.4f}"
                f"{metrics_dict['SAE']['corr']:<{col_3}.4f}"
                f"{metrics_dict['Equal_Weight']['corr']:<{col_4}.4f}"
                f"{1.0:<{col_5}.4f}")
    print(row_corr)
    
    row_mdd = (f"{'Max Drawdown':<{col_1}}"
               f"{metrics_dict['Lasso']['mdd'] * 100:<{col_2}.2f}%"
               f"{metrics_dict['SAE']['mdd'] * 100:<{col_3}.2f}%"
               f"{metrics_dict['Equal_Weight']['mdd'] * 100:<{col_4}.2f}%"
               f"{metrics_dict['S&P 500']['mdd'] * 100:<{col_5}.2f}%")
    print(row_mdd)
    
    row_stocks = (f"{'Stocks Used':<{col_1}}"
                  f"{metrics_dict['Lasso']['stocks']:<{col_2}}"
                  f"{metrics_dict['SAE']['stocks']:<{col_3}}"
                  f"{metrics_dict['Equal_Weight']['stocks']:<{col_4}}"
                  f"{500:<{col_5}}")
    print(row_stocks)
    print("=" * 80 + "\n")

# =========================================================
# Main Execution
# =========================================================

def run_diagnostics():
    print("--- Starting Milestone 5: Final Diagnostics ---")
    
    # 1. Load the unified Backtest results and Frozen Weights
    returns_df = pd.read_csv('data/oos_portfolio_returns.csv', index_col=0, parse_dates=True)
    weights_df = pd.read_csv('models/frozen_weights_2022.csv', index_col=0)
    
    bench_returns = returns_df['S&P 500 (Target)']
    port_returns = returns_df.drop(columns=['S&P 500 (Target)'])
    
    # 2. Execute original summary metrics and save to CSV
    metrics_df = compute_metrics(port_returns, bench_returns, weights_df)
    metrics_df.to_csv("data/final_summary_metrics.csv")
    
    # 3. Build the exact metrics dictionary required for the tear sheet
    metrics_dict = {}
    for model in ['Equal_Weight', 'Lasso', 'SAE']:
        p_ret = port_returns[model]
        
        te = np.std(p_ret - bench_returns) * np.sqrt(252)
        corr = p_ret.corr(bench_returns)
        mdd = calculate_max_drawdown(p_ret)
        stocks_used = int((weights_df[model] > 1e-4).sum())
        
        metrics_dict[model] = {'te': te, 'corr': corr, 'mdd': mdd, 'stocks': stocks_used}
        
    # Add S&P 500 baseline stats
    metrics_dict['S&P 500'] = {'mdd': calculate_max_drawdown(bench_returns)}
    
    # 4. Output Graphics and Formatted Tear Sheet
    plot_residual_distribution(port_returns, bench_returns)
    print_performance_metrics(metrics_dict)
    
    print("Milestone 5 Complete. Tear sheet fully generated and plots saved.")