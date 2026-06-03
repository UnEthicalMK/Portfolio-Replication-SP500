import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')

TARGET = '^GSPC'
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
    raw_prices = pd.read_csv('data/raw_prices.csv', index_col=0, parse_dates=True)
    prices_2023 = raw_prices.loc['2023-01-01':'2023-12-31']
    weights = pd.read_csv('models/frozen_weights_2022.csv', index_col=0)
    return prices_2023, weights

def simulate_buy_and_hold(prices: pd.DataFrame, weights_df: pd.DataFrame) -> tuple:
    print("Executing buy-and-hold backtest with Global ADR transaction costs...")
    models = weights_df.columns
    portfolio_values = pd.DataFrame(index=prices.index, columns=models)
    p0 = prices.iloc[0]
    
    for model in models:
        model_weights = weights_df[model]
        allocated_capital = model_weights * INITIAL_CAPITAL
        
        capital_after_fees = allocated_capital.copy()
        for ticker in model_weights.index:
            if model_weights[ticker] > 0:
                fee = get_friction(ticker)
                capital_after_fees[ticker] *= (1.0 - fee)
        
        shares = capital_after_fees / p0.drop(TARGET, errors='ignore')
        shares = shares.fillna(0)
        
        proxy_prices = prices.drop(columns=[TARGET])
        daily_value = proxy_prices.dot(shares)
        portfolio_values[model] = daily_value
        
    portfolio_returns = np.log(portfolio_values).diff().dropna()
    
    benchmark_val = prices[TARGET]
    benchmark_val_rebased = benchmark_val / benchmark_val.iloc[0] * INITIAL_CAPITAL
    benchmark_returns = np.log(benchmark_val).diff().dropna()
    
    return portfolio_values, portfolio_returns, benchmark_val_rebased, benchmark_returns

def plot_cumulative_returns(port_values: pd.DataFrame, bench_val: pd.Series):
    print("Generating Plot 6: Cumulative Returns (2023)...")
    fig, ax = plt.subplots(figsize=(13, 6))
    
    bench_cum = bench_val / bench_val.iloc[0]
    ax.plot(bench_cum, color='black', linewidth=2.5, label='S&P 500 (Total Return)')
    
    colors = ['steelblue', 'darkorange', 'seagreen']
    for model, color in zip(port_values.columns, colors):
        port_cum = port_values[model] / port_values[model].iloc[0]
        ax.plot(port_cum, color=color, linewidth=1.5, linestyle='--', label=model)
        
    ax.set_title("Cumulative Return — 2023 Out-of-Sample Backtest (Plot 6)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Growth of $1")
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/06_cumulative_returns.png", dpi=150, bbox_inches='tight')
    plt.close()

def plot_rolling_tracking_error(port_returns: pd.DataFrame, bench_returns: pd.Series):
    print("Generating Plot 7: Rolling Tracking Error...")
    fig, ax = plt.subplots(figsize=(13, 5))
    colors = ['steelblue', 'darkorange', 'seagreen']
    
    for model, color in zip(port_returns.columns, colors):
        active_return = port_returns[model] - bench_returns
        rolling_te = active_return.rolling(window=21).std() * np.sqrt(252)
        ax.plot(rolling_te, label=model, color=color, linewidth=1.5)
        
    ax.set_title("Rolling 21-Day Annualised Tracking Error (2023) (Plot 7)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Annualised Tracking Error")
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/07_rolling_tracking_error.png", dpi=150, bbox_inches='tight')
    plt.close()

def run_backtest():
    print("\n--- Starting Milestone 4: Out-of-Sample Backtest ---")
    ensure_directories()
    prices, weights = load_backtest_data()
    port_vals, port_rets, bench_val, bench_rets = simulate_buy_and_hold(prices, weights)
    
    plot_cumulative_returns(port_vals, bench_val)
    plot_rolling_tracking_error(port_rets, bench_rets)
    
    # FIX: Merge the returns into a single DataFrame for seamless diagnostics integration
    combined_returns = port_rets.copy()
    combined_returns['S&P 500 (Target)'] = bench_rets
    combined_returns.to_csv('data/oos_portfolio_returns.csv')
    
    print("Milestone 4 Complete. Backtest PnL and Returns saved.\n")