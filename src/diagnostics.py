import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from src.config import DATA_DIR, MODELS_DIR, PLOTS_DIR, TARGET

plt.style.use('seaborn-v0_8-whitegrid')


def write_metrics_markdown(metrics: pd.DataFrame, benchmark_returns: pd.Series) -> None:
    """Write a GitHub-friendly strategy report while retaining the CSV export."""
    benchmark_return = np.expm1(benchmark_returns.sum())
    report_lines = [
        "# Final Out-of-Sample Metrics",
        "",
        f"- Benchmark: `{TARGET}`",
        f"- Benchmark cumulative return: **{benchmark_return:.2%}**",
        f"- Common portfolio size: **K={int(metrics['K'].iloc[0])}**",
        "- Returns are net of the configured transaction-cost assumptions.",
        "",
    ]

    columns = [
        "Model",
        "K",
        "Tracking Error (Ann.)",
        "Cumulative Return",
        "Correlation",
        "Max Drawdown",
        "Max Daily Deviation",
        "Active Instruments",
        "Residual Sharpe",
    ]
    headers = [
        "Model", "K", "Tracking error", "Cumulative return", "Correlation",
        "Max drawdown", "Max daily deviation", "Active instruments",
        "Residual Sharpe",
    ]
    for strategy, group in metrics.groupby("Strategy", sort=False):
        report_lines.extend([
            f"## {strategy.replace('_', ' ').title()}",
            "",
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ])
        for _, row in group.iterrows():
            values = [
                row["Model"],
                str(int(row["K"])),
                f"{float(row['Tracking Error (Ann.)']):.2%}",
                f"{float(row['Cumulative Return']):.2%}",
                f"{float(row['Correlation']):.4f}",
                f"{float(row['Max Drawdown']):.2%}",
                f"{float(row['Max Daily Deviation']):.2%}",
                str(int(row["Active Instruments"])),
                row["Residual Sharpe"],
            ]
            report_lines.append("| " + " | ".join(values) + " |")
        report_lines.append("")

    (DATA_DIR / "final_summary_metrics.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

# =========================================================
# Quantitative Math Helpers
# =========================================================

def calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculates the maximum peak-to-trough drop in a return series."""
    cum_returns = np.exp(returns.cumsum())
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    return drawdown.min()

def compute_metrics(
    port_returns: pd.DataFrame,
    bench_returns: pd.Series,
    active_counts: dict[str, int] | pd.DataFrame,
    strategy: str = "unknown",
    common_k: int | None = None,
) -> pd.DataFrame:
    """Calculate strategy metrics for the saved research report."""
    
    metrics = {}
    
    for model in port_returns.columns:
        p_ret = port_returns[model]
        residuals = p_ret - bench_returns
        
        # 1. Annualised Tracking Error
        te = residuals.std() * np.sqrt(252)
        
        # 2. Cumulative Return 2023
        cum_ret = np.expm1(p_ret.sum())
        
        # 3. Max Single-Day Deviation (absolute)
        max_dev = residuals.abs().max()
        
        # 4. Active Instrument Count (Weights > 0)
        if isinstance(active_counts, pd.DataFrame):
            active_count = int((active_counts[model] > 1e-4).sum())
        else:
            active_count = active_counts[model]
        
        # 5. Sharpe Ratio of Residuals
        residual_std = residuals.std()
        resid_sharpe = (
            (residuals.mean() / residual_std) * np.sqrt(252)
            if residual_std > 0 else np.nan
        )
        
        metrics[model] = {
            "Strategy": strategy,
            "K": common_k if common_k is not None else active_count,
            "Tracking Error (Ann.)": f"{te:.4f}",
            "Cumulative Return": f"{cum_ret:.4f}",
            "Correlation": f"{p_ret.corr(bench_returns):.4f}",
            "Max Drawdown": f"{calculate_max_drawdown(p_ret):.4f}",
            "Max Daily Deviation": f"{max_dev:.4f}",
            "Active Instruments": int(active_count),
            "Residual Sharpe": f"{resid_sharpe:.2f}"
        }
        
    df = pd.DataFrame(metrics)
    return df

# =========================================================
# Visualizations
# =========================================================

def plot_residual_distribution(port_returns: pd.DataFrame, bench_returns: pd.Series):
    """Plot 8: Visualise tracking bias, tails, and normal-fit comparison."""
    print("Generating Plot 8: Residual distributions and normal fits...")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['steelblue', 'darkorange', 'seagreen']
    
    for (model, color) in zip(port_returns.columns, colors):
        residuals = port_returns[model] - bench_returns
        
        # Plot histogram
        ax.hist(residuals, bins=50, alpha=0.4, color=color, label=model, density=True)
        
        # Fit and plot a normal distribution over it
        mu, std = residuals.mean(), residuals.std()
        x = np.linspace(residuals.min(), residuals.max(), 200)
        ax.plot(
            x,
            norm.pdf(x, mu, std),
            color=color,
            linewidth=1.5,
            linestyle="--",
            label=f"{model} normal fit",
        )
        
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title("2023 Tracking Residuals with Normal-Fit Overlays", fontsize=12, fontweight='bold')
    ax.set_xlabel("Daily Residual")
    ax.set_ylabel("Density")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "08_residual_distribution.png", dpi=150, bbox_inches='tight')
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
              f"{metrics_dict['Lasso']['te'] * 100:>{col_2 - 1}.3f}%"
              f"{metrics_dict['SAE']['te'] * 100:>{col_3 - 1}.3f}%"
              f"{metrics_dict['Equal_Weight']['te'] * 100:>{col_4 - 1}.3f}%"
              f"{0.0:>{col_5 - 1}.2f}%")
    print(row_te)
    
    row_corr = (f"{'Correlation':<{col_1}}"
                f"{metrics_dict['Lasso']['corr']:<{col_2}.4f}"
                f"{metrics_dict['SAE']['corr']:<{col_3}.4f}"
                f"{metrics_dict['Equal_Weight']['corr']:<{col_4}.4f}"
                f"{1.0:<{col_5}.4f}")
    print(row_corr)
    
    row_mdd = (f"{'Max Drawdown':<{col_1}}"
               f"{metrics_dict['Lasso']['mdd'] * 100:>{col_2 - 1}.2f}%"
               f"{metrics_dict['SAE']['mdd'] * 100:>{col_3 - 1}.2f}%"
               f"{metrics_dict['Equal_Weight']['mdd'] * 100:>{col_4 - 1}.2f}%"
               f"{metrics_dict['S&P 500']['mdd'] * 100:>{col_5 - 1}.2f}%")
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
    returns_df = pd.read_csv(DATA_DIR / 'oos_portfolio_returns.csv', index_col=0, parse_dates=True)
    weights_df = pd.read_csv(MODELS_DIR / 'frozen_weights_2022.csv', index_col=0)
    common_k = int(
        pd.read_csv(DATA_DIR / "k_selection_validation.csv")
        .sort_values(["Mean_TE", "Worst_TE", "K"])
        .iloc[0]["K"]
    )
    adaptive_weights = pd.read_csv(DATA_DIR / "walk_forward_weights.csv")
    adaptive_counts = (
        adaptive_weights[adaptive_weights["Weight"] > 1e-4]
        .groupby(["Model", "RebalanceDate"])["Ticker"]
        .nunique()
        .groupby(level=0)
        .max()
        .to_dict()
    )
    
    if 'S&P 500 (Target)' not in returns_df.columns:
        raise ValueError("Backtest output is missing the S&P 500 benchmark returns.")

    required_models = {'Equal_Weight', 'Lasso', 'SAE'}
    available_models = {
        column.split("__", 1)[0]
        for column in returns_df.columns
        if "__" in column
    }
    missing_models = required_models.difference(available_models)
    if missing_models:
        raise ValueError(
            f"Backtest output is missing model returns: {sorted(missing_models)}"
        )

    missing_weights = required_models.difference(weights_df.columns)
    if missing_weights:
        raise ValueError(
            f"Frozen weights are missing model columns: {sorted(missing_weights)}"
        )

    bench_returns = returns_df['S&P 500 (Target)']
    strategy_columns = [
        column for column in returns_df.columns if column != 'S&P 500 (Target)'
    ]
    strategies = sorted({column.split("__", 1)[1] for column in strategy_columns})
    summaries = []
    for strategy in strategies:
        selected = [
            column for column in strategy_columns
            if column.endswith(f"__{strategy}")
        ]
        port_returns = returns_df[selected].rename(
            columns=lambda column: column.split("__", 1)[0]
        )
        if strategy.startswith("walk_forward"):
            active_counts = {model: int(adaptive_counts[model]) for model in port_returns}
        else:
            active_counts = {
                model: int((weights_df[model] > 1e-4).sum())
                for model in port_returns
            }
        metrics_df = compute_metrics(
            port_returns,
            bench_returns,
            active_counts,
            strategy,
            common_k,
        )
        summary = metrics_df.T.reset_index(names="Model")
        summaries.append(summary)

    all_metrics = pd.concat(summaries, ignore_index=True)
    all_metrics.to_csv(DATA_DIR / "final_summary_metrics.csv", index=False)
    write_metrics_markdown(all_metrics, bench_returns)

    port_returns = returns_df[
        [column for column in strategy_columns if column.endswith("__buy_hold")]
    ].rename(columns=lambda column: column.split("__", 1)[0])
    if set(port_returns.columns) != required_models:
        raise ValueError("Backtest output must include buy-and-hold returns for all models.")
    
    print("  Saved readable metrics report to data/final_summary_metrics.md")
    print("  Saved machine-readable metrics to data/final_summary_metrics.csv")
    plot_residual_distribution(port_returns, bench_returns)
    print("  Saved residual diagnostic plot to plots/08_residual_distribution.png")
    print("Milestone 5 Complete. Metrics and plots saved.\n")