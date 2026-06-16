import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

plt.style.use('seaborn-v0_8-whitegrid')

def calculate_max_drawdown(returns: pd.Series) -> float:
    # Compute peak-to-trough decline from cumulative returns.
    cumulative = np.exp(returns.cumsum())
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()

def compute_metrics(port_returns: pd.DataFrame,
                    bench_returns: pd.Series,
                    weights_df: pd.DataFrame) -> pd.DataFrame:
    print("[INFO] Computing performance metrics...")

    metrics = {}

    for model in ['Equal_Weight', 'Ridge_Tracker', 'SAE']:
        p_ret = port_returns[model]
        residuals = p_ret - bench_returns

        te = residuals.std() * np.sqrt(252)
        cum_ret = np.exp(p_ret.sum()) - 1
        max_dev = residuals.abs().max()
        active_count = (weights_df[model] > 1e-4).sum()
        resid_sharpe = (residuals.mean() / residuals.std()) * np.sqrt(252)

        metrics[model] = {
            "Tracking Error (Ann.)": te,
            "Cumulative Return": cum_ret,
            "Max Daily Deviation": max_dev,
            "Active Instruments": int(active_count),
            "Residual Sharpe": resid_sharpe
        }

    bench_cum = np.exp(bench_returns.sum()) - 1

    df = pd.DataFrame(metrics)

    print("\nPerformance Summary")
    print("-" * 60)
    print(f"Benchmark Return : {bench_cum:.4f}")
    print("-" * 60)
    print(df.to_string())
    print("-" * 60)

    return df

def plot_residual_distribution(port_returns: pd.DataFrame,
                               bench_returns: pd.Series):
    print("[INFO] Generating residual distribution plot...")

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['steelblue', 'darkorange', 'seagreen']

    for model, color in zip(port_returns.columns, colors):
        residuals = port_returns[model] - bench_returns

        ax.hist(residuals, bins=50, alpha=0.4,
                color=color, label=model, density=True)

        mu, std = residuals.mean(), residuals.std()
        x = np.linspace(residuals.min(), residuals.max(), 200)

        ax.plot(x, norm.pdf(x, mu, std),
                color=color, linewidth=1.5)

    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title("Residual Distribution — Portfolio minus S&P 500 (2023)",
                 fontsize=12, fontweight='bold')
    ax.set_xlabel("Daily Residual")
    ax.set_ylabel("Density")
    ax.legend()

    plt.tight_layout()
    plt.savefig("plots/08_residual_distribution.png",
                dpi=150, bbox_inches='tight')
    plt.close()

def print_performance_metrics(metrics_dict: dict):
    col_1 = 18
    col_2 = 14
    col_3 = 14
    col_4 = 18
    col_5 = 12

    print("\n" + "=" * 80)

    header = (
        f"{'Metric':<{col_1}}"
        f"{'Ridge Tracker':<{col_2}}"
        f"{'Autoencoder':<{col_3}}"
        f"{'Equal Weight':<{col_4}}"
        f"{'S&P 500':<{col_5}}"
    )

    print(header)
    print("-" * 80)

    print(
        f"{'Tracking Error':<{col_1}}"
        f"{metrics_dict['Ridge_Tracker']['te'] * 100:<{col_2}.3f}%"
        f"{metrics_dict['SAE']['te'] * 100:<{col_3}.3f}%"
        f"{metrics_dict['Equal_Weight']['te'] * 100:<{col_4}.3f}%"
        f"{0.0:<{col_5}.2f}%"
    )

    print(
        f"{'Correlation':<{col_1}}"
        f"{metrics_dict['Ridge_Tracker']['corr']:<{col_2}.4f}"
        f"{metrics_dict['SAE']['corr']:<{col_3}.4f}"
        f"{metrics_dict['Equal_Weight']['corr']:<{col_4}.4f}"
        f"{1.0:<{col_5}.4f}"
    )

    print(
        f"{'Max Drawdown':<{col_1}}"
        f"{metrics_dict['Ridge_Tracker']['mdd'] * 100:<{col_2}.2f}%"
        f"{metrics_dict['SAE']['mdd'] * 100:<{col_3}.2f}%"
        f"{metrics_dict['Equal_Weight']['mdd'] * 100:<{col_4}.2f}%"
        f"{metrics_dict['S&P 500']['mdd'] * 100:<{col_5}.2f}%"
    )

    print(
        f"{'Stocks Used':<{col_1}}"
        f"{metrics_dict['Ridge_Tracker']['stocks']:<{col_2}}"
        f"{metrics_dict['SAE']['stocks']:<{col_3}}"
        f"{metrics_dict['Equal_Weight']['stocks']:<{col_4}}"
        f"{500:<{col_5}}"
    )

    print("=" * 80 + "\n")

def run_diagnostics():
    print("\n" + "=" * 60)
    print("MILESTONE 5 | PERFORMANCE DIAGNOSTICS")
    print("=" * 60)

    print("[INFO] Loading backtest results...")

    returns_df = pd.read_csv(
        'data/oos_portfolio_returns.csv',
        index_col=0,
        parse_dates=True
    )

    weights_df = pd.read_csv(
        'models/frozen_weights_2022.csv',
        index_col=0
    )

    bench_returns = returns_df['S&P 500 (Target)']
    port_returns = returns_df.drop(columns=['S&P 500 (Target)'])

    metrics_df = compute_metrics(
        port_returns,
        bench_returns,
        weights_df
    )

    metrics_df.to_csv("data/final_summary_metrics.csv")

    metrics_dict = {}

    for model in port_returns.columns:
        p_ret = port_returns[model]

        te = np.std(p_ret - bench_returns) * np.sqrt(252)
        corr = p_ret.corr(bench_returns)
        mdd = calculate_max_drawdown(p_ret)
        stocks_used = int((weights_df[model] > 1e-4).sum())

        metrics_dict[model] = {
            'te': te,
            'corr': corr,
            'mdd': mdd,
            'stocks': stocks_used
        }

    # Benchmark statistics
    metrics_dict['S&P 500'] = {
        'mdd': calculate_max_drawdown(bench_returns)
    }

    plot_residual_distribution(port_returns, bench_returns)
    print_performance_metrics(metrics_dict)

    print("[SUCCESS] Diagnostics completed.\n")