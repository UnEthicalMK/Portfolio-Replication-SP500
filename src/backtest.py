import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.config import (
    DATA_DIR,
    MODELS_DIR,
    PLOTS_DIR,
    REBALANCE_FREQUENCIES,
    TARGET,
    TEST_START,
    WALK_FORWARD_VALIDATION_DAYS,
)
from src.walk_forward import fit_walk_forward_weights

plt.style.use('seaborn-v0_8-whitegrid')

INITIAL_CAPITAL = 1_000_000.0

# ---------------------------------------------------------
# Friction Matrix: Global ADR Liquidity Tiers
# ---------------------------------------------------------
FRICTION_BPS = {
    # Emerging Market ADRs (Wider bid-ask spreads)
    'EM_ADR': [
        'BABA', 'JD', 'BIDU', 'TME', 'NTES', 'PDD', # China/HK
        'INFY', 'WIT', 'HDB', 'IBN', 'RDY', 'MMYT', # India
        'MELI', 'VALE', 'PBR', 'BBD',               # LatAm
        'TSM', 'UMC',                               # Taiwan
        'SKM', 'KB'                                 # South Korea
    ],
    'EM_FEE': 0.0015,  # 15 bps
    
    # Developed Market ADRs (Tighter bid-ask spreads)
    'DEV_ADR': [
        'ASML', 'NVO', 'SAP', 'SHEL', 'BCS',        # Europe
        'TM', 'SONY', 'HMC'                         # Japan
    ],
    'DEV_FEE': 0.0010  # 10 bps
}

def get_friction(ticker: str) -> float:
    """Assigns institutional execution slippage based on MSCI market classification."""
    if ticker in FRICTION_BPS['EM_ADR']:
        return FRICTION_BPS['EM_FEE']
    return FRICTION_BPS['DEV_FEE']

def ensure_directories():
    os.makedirs('plots', exist_ok=True)
    os.makedirs('data', exist_ok=True)

def load_backtest_data() -> tuple:
    print("Loading 2023 Out-of-Sample data and frozen weights...")
    raw_prices = pd.read_csv(DATA_DIR / 'raw_prices.csv', index_col=0, parse_dates=True)
    prices_2023 = raw_prices.loc[TEST_START:'2023-12-31']
    weights = pd.read_csv(MODELS_DIR / 'frozen_weights_2022.csv', index_col=0)

    if prices_2023.empty:
        raise ValueError("No price data is available for the 2023 backtest period.")
    if TARGET not in prices_2023.columns:
        raise ValueError(f"Backtest prices are missing the benchmark column: {TARGET}")
    if weights.empty:
        raise ValueError("Frozen portfolio weights are empty.")

    return prices_2023, weights

def _validate_weights(prices: pd.DataFrame, weights_df: pd.DataFrame) -> None:
    models = weights_df.columns
    proxy_prices = prices.drop(columns=[TARGET])

    missing_assets = weights_df.index.difference(proxy_prices.columns)
    if not missing_assets.empty:
        raise ValueError(
            f"Frozen weights contain assets missing from prices: {missing_assets.tolist()}"
        )

    for model in models:
        model_weights = weights_df[model].fillna(0.0)
        if not np.isfinite(model_weights).all():
            raise ValueError(f"Model {model} contains non-finite portfolio weights.")
        if (model_weights < 0).any():
            raise ValueError(f"Model {model} contains negative portfolio weights.")
        if not np.isclose(model_weights.sum(), 1.0, atol=1e-6):
            raise ValueError(
                f"Model {model} weights must sum to 1.0; got {model_weights.sum():.8f}."
            )


def _rebalance_dates(index: pd.DatetimeIndex, frequency: str) -> set:
    if frequency == "buy_hold":
        return {index[0]}
    if frequency not in {"monthly", "quarterly"}:
        raise ValueError(f"Unsupported rebalance frequency: {frequency}")
    periods = pd.Series(index.to_period("M" if frequency == "monthly" else "Q"), index=index)
    return set(index[periods.ne(periods.shift(1))])


def simulate_rebalanced(
    prices: pd.DataFrame, weights_df: pd.DataFrame, frequency: str,
    include_costs: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _validate_weights(prices, weights_df)
    models = weights_df.columns
    proxy_prices = prices.drop(columns=[TARGET])
    portfolio_values = pd.DataFrame(
        index=prices.index, columns=models, dtype=float
    )
    gross_values = portfolio_values.copy()
    fees_paid = pd.DataFrame(0.0, index=prices.index, columns=models)
    turnovers = pd.DataFrame(0.0, index=prices.index, columns=models)
    rebalance_dates = _rebalance_dates(prices.index, frequency)
    
    for model in models:
        target_weights = weights_df[model].fillna(0.0)
        shares = pd.Series(0.0, index=proxy_prices.columns)
        gross_shares = shares.copy()
        portfolio_value = INITIAL_CAPITAL
        for date in prices.index:
            current_prices = proxy_prices.loc[date]
            marked_value = float((shares * current_prices).sum())
            gross_marked_value = float((gross_shares * current_prices).sum())
            if date == prices.index[0]:
                marked_value = INITIAL_CAPITAL
                gross_marked_value = INITIAL_CAPITAL
            if date in rebalance_dates:
                target_value = marked_value * target_weights
                current_value = shares * current_prices
                traded_value = (target_value - current_value).abs()
                fees = sum(
                    traded_value[ticker] * get_friction(ticker)
                    for ticker in target_value.index
                )
                portfolio_value = max(marked_value - fees, 0.0)
                fees_paid.loc[date, model] = fees
                gross_shares = (marked_value * target_weights) / current_prices
                target_value = portfolio_value * target_weights
                shares = target_value / current_prices
                turnovers.loc[date, model] = traded_value.sum() / max(marked_value, 1.0)
            else:
                portfolio_value = marked_value
            portfolio_values.loc[date, model] = portfolio_value
            gross_values.loc[date, model] = gross_marked_value
    portfolio_returns = np.log(portfolio_values).diff().dropna()
    benchmark_val = prices[TARGET]
    benchmark_val_rebased = benchmark_val / benchmark_val.iloc[0] * INITIAL_CAPITAL
    benchmark_returns = np.log(benchmark_val).diff().dropna()
    result = (
        portfolio_values, portfolio_returns, benchmark_val_rebased,
        benchmark_returns, turnovers,
    )
    if include_costs:
        return result + (fees_paid, np.log(gross_values).diff().dropna())
    return result


def simulate_buy_and_hold(prices: pd.DataFrame, weights_df: pd.DataFrame) -> tuple:
    result = simulate_rebalanced(prices, weights_df, "buy_hold")
    return result[:4]


def simulate_walk_forward(
    prices: pd.DataFrame,
    weights_by_date: dict[str, pd.DataFrame],
    frequency: str = "monthly",
    include_costs: bool = False,
) -> tuple:
    """Simulate portfolios whose target weights are refreshed at each rebalance."""
    if frequency not in {"monthly", "quarterly"}:
        raise ValueError("Walk-forward simulation supports monthly or quarterly rebalancing.")
    models = list(weights_by_date)
    proxy_prices = prices.drop(columns=[TARGET])
    target_dates = set(next(iter(weights_by_date.values())).index)
    values = pd.DataFrame(index=prices.index, columns=models, dtype=float)
    gross_values = values.copy()
    fees_paid = pd.DataFrame(0.0, index=prices.index, columns=models)
    turnovers = pd.DataFrame(0.0, index=prices.index, columns=models)

    for model in models:
        weights = weights_by_date[model].reindex(columns=proxy_prices.columns).fillna(0.0)
        shares = pd.Series(0.0, index=proxy_prices.columns)
        gross_shares = shares.copy()
        for date in prices.index:
            current_prices = proxy_prices.loc[date]
            marked_value = float((shares * current_prices).sum())
            gross_marked_value = float((gross_shares * current_prices).sum())
            if date == prices.index[0]:
                marked_value = INITIAL_CAPITAL
                gross_marked_value = INITIAL_CAPITAL
            if date in target_dates:
                target_weights = weights.loc[date]
                current_value = shares * current_prices
                target_value = marked_value * target_weights
                traded_value = (target_value - current_value).abs()
                fees = sum(
                    traded_value[ticker] * get_friction(ticker)
                    for ticker in target_value.index
                )
                net_value = max(marked_value - fees, 0.0)
                fees_paid.loc[date, model] = fees
                gross_shares = (marked_value * target_weights) / current_prices
                shares = (net_value * target_weights) / current_prices
                turnovers.loc[date, model] = (
                    traded_value.sum() / max(marked_value, 1.0)
                )
                marked_value = net_value
            values.loc[date, model] = marked_value
            gross_values.loc[date, model] = gross_marked_value

    returns = np.log(values).diff().dropna()
    benchmark_value = prices[TARGET] / prices[TARGET].iloc[0] * INITIAL_CAPITAL
    benchmark_returns = np.log(prices[TARGET]).diff().dropna()
    result = values, returns, benchmark_value, benchmark_returns, turnovers
    if include_costs:
        return result + (fees_paid, np.log(gross_values).diff().dropna())
    return result


def _record_cost_attribution(
    rows: list[dict],
    strategy: str,
    net_returns: pd.DataFrame,
    gross_returns: pd.DataFrame,
    fees: pd.DataFrame,
) -> None:
    for model in net_returns:
        total_fees = float(fees[model].sum())
        rows.append(
            {
                "Strategy": strategy,
                "Model": model,
                "Gross Return": np.expm1(gross_returns[model].sum()),
                "Net Return": (
                    (1.0 - fees[model].iloc[0] / INITIAL_CAPITAL)
                    * np.exp(net_returns[model].sum())
                    - 1.0
                ),
                "Transaction Costs": total_fees,
                "Transaction Cost (bps)": total_fees / INITIAL_CAPITAL * 10000,
            }
        )

def plot_cumulative_returns(port_values: pd.DataFrame, bench_val: pd.Series):
    print("Generating Plot 6: Out-of-sample performance...")
    panels = (
        ("buy_hold", "Buy and hold"),
        ("walk_forward_monthly", "Walk-forward — monthly"),
        ("walk_forward_quarterly", "Walk-forward — quarterly"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    
    bench_cum = bench_val / bench_val.iloc[0]
    colors = {"Equal_Weight": "steelblue", "Lasso": "darkorange", "SAE": "seagreen"}
    for ax, (frequency, title) in zip(axes, panels):
        ax.plot(bench_cum, color="black", linewidth=2, label="Benchmark")
        for model, color in colors.items():
            column = model if frequency == "buy_hold" else f"{model}__{frequency}"
            if column in port_values:
                port_cum = port_values[column] / port_values[column].iloc[0]
                ax.plot(port_cum, color=color, linewidth=1.4, label=model)
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("Growth of $1")
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "06_oos_performance.png", dpi=150, bbox_inches='tight')
    plt.close()

def plot_rolling_tracking_error(port_returns: pd.DataFrame, bench_returns: pd.Series):
    print("Generating Plot 7: Rolling tracking error...")
    panels = (
        ("buy_hold", "Buy and hold"),
        ("walk_forward_monthly", "Walk-forward — monthly"),
        ("walk_forward_quarterly", "Walk-forward — quarterly"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    colors = {"Equal_Weight": "steelblue", "Lasso": "darkorange", "SAE": "seagreen"}
    for ax, (frequency, title) in zip(axes, panels):
        for model, color in colors.items():
            column = f"{model}__{frequency}"
            if column in port_returns:
                rolling_te = (port_returns[column] - bench_returns).rolling(21).std() * np.sqrt(252)
                ax.plot(rolling_te, label=model, color=color, linewidth=1.4)
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("21-day annualized tracking error")
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "07_oos_tracking_error.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_turnover(turnover: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    turnover.T.plot.bar(ax=ax, color=["steelblue", "darkorange", "seagreen"])
    ax.set_title("Cumulative traded notional by policy")
    ax.set_ylabel("Turnover / initial portfolio value")
    ax.set_xlabel("Execution policy")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Model")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "09_turnover.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

def run_backtest():
    print("\n--- Starting Milestone 4: Out-of-Sample Backtest ---")
    ensure_directories()
    prices, weights = load_backtest_data()
    all_returns = {}
    cost_rows = []
    (
        port_vals, port_rets, bench_val, bench_rets, _, buyhold_fees, buyhold_gross
    ) = simulate_rebalanced(prices, weights, "buy_hold", include_costs=True)
    all_returns.update({f"{model}__buy_hold": port_rets[model] for model in port_rets})
    _record_cost_attribution(cost_rows, "buy_hold", port_rets, buyhold_gross, buyhold_fees)
    total_turnover = {}

    monthly_dates = sorted(_rebalance_dates(prices.index, "monthly"))
    quarterly_dates = sorted(_rebalance_dates(prices.index, "quarterly"))
    raw_returns = pd.read_csv(
        DATA_DIR / "log_returns.csv", index_col=0, parse_dates=True
    ).loc[:prices.index[-1]]
    common_k = int(
        pd.read_csv(DATA_DIR / "k_selection_validation.csv")
        .sort_values(["Mean_TE", "Worst_TE", "K"])
        .iloc[0]["K"]
    )
    print(f"  Running monthly walk-forward retraining with K={common_k}...")
    monthly_weights = fit_walk_forward_weights(
        raw_returns,
        monthly_dates,
        common_k,
        validation_days=WALK_FORWARD_VALIDATION_DAYS,
    )
    weight_frames = []
    (
        monthly_values, monthly_returns, _, _, monthly_turnover,
        monthly_fees, monthly_gross,
    ) = simulate_walk_forward(
        prices, monthly_weights, "monthly", include_costs=True
    )
    print(f"  Running quarterly walk-forward retraining with K={common_k}...")
    quarterly_weights = fit_walk_forward_weights(
        raw_returns, quarterly_dates, common_k,
        validation_days=WALK_FORWARD_VALIDATION_DAYS,
    )
    (
        quarterly_values, quarterly_returns, _, _, quarterly_turnover,
        quarterly_fees, quarterly_gross,
    ) = simulate_walk_forward(
        prices, quarterly_weights, "quarterly", include_costs=True
    )
    _record_cost_attribution(
        cost_rows, "walk_forward_monthly", monthly_returns, monthly_gross, monthly_fees
    )
    _record_cost_attribution(
        cost_rows, "walk_forward_quarterly",
        quarterly_returns, quarterly_gross, quarterly_fees,
    )
    for frequency, frames in (
        ("monthly", monthly_weights),
        ("quarterly", quarterly_weights),
    ):
        weight_frames.extend(
            frame.rename_axis("RebalanceDate")
            .stack()
            .rename("Weight")
            .reset_index()
            .rename(columns={"level_1": "Ticker"})
            .assign(Model=model, Frequency=frequency)
            for model, frame in frames.items()
        )
    pd.concat(weight_frames, ignore_index=True).to_csv(
        DATA_DIR / "walk_forward_weights.csv", index=False
    )
    all_returns.update({
        f"{model}__walk_forward_monthly": monthly_returns[model]
        for model in monthly_returns
    })
    all_returns.update({
        f"{model}__walk_forward_quarterly": quarterly_returns[model]
        for model in quarterly_returns
    })
    total_turnover["walk_forward_monthly"] = monthly_turnover.sum().to_dict()
    total_turnover["walk_forward_quarterly"] = quarterly_turnover.sum().to_dict()

    all_values = {
        "Equal_Weight": port_vals["Equal_Weight"],
        "Lasso": port_vals["Lasso"],
        "SAE": port_vals["SAE"],
    }
    for model in monthly_values:
        all_values[f"{model}__walk_forward_monthly"] = monthly_values[model]
    for model in quarterly_values:
        all_values[f"{model}__walk_forward_quarterly"] = quarterly_values[model]
    all_values_df = pd.DataFrame(all_values)
    plot_cumulative_returns(all_values_df, bench_val)
    plot_rolling_tracking_error(pd.DataFrame(all_returns), bench_rets)
    plot_turnover(pd.DataFrame(total_turnover))
    
    # FIX: Merge the returns into a single DataFrame for seamless diagnostics integration
    combined_returns = pd.DataFrame(all_returns)
    combined_returns['S&P 500 (Target)'] = bench_rets
    combined_returns.to_csv(DATA_DIR / 'oos_portfolio_returns.csv')
    pd.DataFrame(total_turnover).to_csv(DATA_DIR / "oos_turnover.csv")
    pd.DataFrame(cost_rows).to_csv(DATA_DIR / "oos_cost_attribution.csv", index=False)
    
    print("Milestone 4 Complete. Backtest PnL and Returns saved.\n")