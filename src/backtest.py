import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')

TARGET = '^GSPC'
INITIAL_CAPITAL = 1_000_000.0

# Transaction cost assumptions by market liquidity
FRICTION_BPS = {
    'EM_ADR': [
        'BABA', 'JD', 'BIDU', 'TME', 'NTES', 'PDD',
        'INFY', 'WIT', 'HDB', 'IBN', 'RDY', 'MMYT',
        'MELI', 'VALE', 'PBR', 'BBD',
        'TSM', 'UMC',
        'SKM', 'KB'
    ],
    'EM_FEE': 0.0015,
    'DEV_ADR': [
        'ASML', 'NVO', 'SAP', 'SHEL', 'BCS',
        'TM', 'SONY', 'HMC'
    ],
    'DEV_FEE': 0.0010
}

def get_friction(ticker: str) -> float:
    if ticker in FRICTION_BPS['EM_ADR']:
        return FRICTION_BPS['EM_FEE']
    return FRICTION_BPS['DEV_FEE']

def ensure_directories():
    os.makedirs('plots', exist_ok=True)
    os.makedirs('data', exist_ok=True)

def load_backtest_data():
    print("[INFO] Loading out-of-sample data and frozen portfolios...")

    raw_prices = pd.read_csv(
        'data/raw_prices.csv',
        index_col=0,
        parse_dates=True
    )

    prices_2023 = raw_prices.loc['2023-01-01':'2023-12-31']

    weights = pd.read_csv(
        'models/frozen_weights_2022.csv',
        index_col=0
    )

    return prices_2023, weights

def simulate_buy_and_hold(
    prices: pd.DataFrame,
    weights_df: pd.DataFrame
):
    print("[INFO] Running buy-and-hold backtest with transaction costs...")

    models = weights_df.columns
    portfolio_values = pd.DataFrame(index=prices.index, columns=models)

    p0 = prices.iloc[0]
    proxy_prices = prices.drop(columns=[TARGET])

    for model in models:
        weights = weights_df[model]
        allocated_capital = weights * INITIAL_CAPITAL

        # Apply one-time entry costs
        capital_after_fees = allocated_capital.copy()

        for ticker in weights.index:
            if weights[ticker] > 0:
                capital_after_fees[ticker] *= (
                    1.0 - get_friction(ticker)
                )

        shares = capital_after_fees / p0.drop(TARGET, errors='ignore')
        shares = shares.fillna(0)

        portfolio_values[model] = proxy_prices.dot(shares)

    portfolio_returns = np.log(portfolio_values).diff().dropna()

    benchmark_value = prices[TARGET]
    benchmark_rebased = (
        benchmark_value /
        benchmark_value.iloc[0] *
        INITIAL_CAPITAL
    )

    benchmark_returns = np.log(benchmark_value).diff().dropna()

    return (
        portfolio_values,
        portfolio_returns,
        benchmark_rebased,
        benchmark_returns
    )

def plot_cumulative_returns(
    portfolio_values: pd.DataFrame,
    benchmark_value: pd.Series
):
    print("[INFO] Generating Plot 6: Cumulative Returns")

    fig, ax = plt.subplots(figsize=(13, 6))

    benchmark_cum = benchmark_value / benchmark_value.iloc[0]

    ax.plot(
        benchmark_cum,
        color='black',
        linewidth=2.5,
        label='S&P 500 (Price Index)'
    )

    colors = ['steelblue', 'darkorange', 'seagreen']

    for model, color in zip(portfolio_values.columns, colors):
        cumulative = (
            portfolio_values[model] /
            portfolio_values[model].iloc[0]
        )

        ax.plot(
            cumulative,
            color=color,
            linewidth=1.5,
            linestyle='--',
            label=model
        )

    ax.set_title(
        "Cumulative Return — 2023 Out-of-Sample Backtest (Plot 6)",
        fontsize=14,
        fontweight='bold'
    )

    ax.set_ylabel("Growth of $1")
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "plots/06_cumulative_returns.png",
        dpi=150,
        bbox_inches='tight'
    )
    plt.close()

def plot_rolling_tracking_error(
    portfolio_returns: pd.DataFrame,
    benchmark_returns: pd.Series
):
    print("[INFO] Generating Plot 7: Rolling Tracking Error")

    fig, ax = plt.subplots(figsize=(13, 5))

    colors = ['steelblue', 'darkorange', 'seagreen']

    for model, color in zip(portfolio_returns.columns, colors):
        active_return = (
            portfolio_returns[model]
            - benchmark_returns
        )

        rolling_te = (
            active_return
            .rolling(window=21)
            .std()
            * np.sqrt(252)
        )

        ax.plot(
            rolling_te,
            label=model,
            color=color,
            linewidth=1.5
        )

    ax.set_title(
        "Rolling 21-Day Annualised Tracking Error (2023) (Plot 7)",
        fontsize=14,
        fontweight='bold'
    )

    ax.set_ylabel("Annualised Tracking Error")
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "plots/07_rolling_tracking_error.png",
        dpi=150,
        bbox_inches='tight'
    )
    plt.close()

def run_backtest():
    print("\nMilestone 4 | Out-of-Sample Backtest")

    ensure_directories()

    prices, weights = load_backtest_data()

    (
        portfolio_values,
        portfolio_returns,
        benchmark_value,
        benchmark_returns
    ) = simulate_buy_and_hold(prices, weights)

    plot_cumulative_returns(
        portfolio_values,
        benchmark_value
    )

    plot_rolling_tracking_error(
        portfolio_returns,
        benchmark_returns
    )

    # Save a unified return matrix for diagnostics
    combined_returns = portfolio_returns.copy()
    combined_returns['S&P 500 (Target)'] = benchmark_returns

    combined_returns.to_csv(
        'data/oos_portfolio_returns.csv'
    )

    print("[SUCCESS] Backtest completed. Results saved.")