import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')

# ---------------------------------------------------------
# Configuration: The Global ADR Universe (10-Year Regime)
# ---------------------------------------------------------
START_DATE = '2014-01-01'
END_DATE = '2023-12-31'
TARGET = '^GSPC'

CHINA_HK    = ['BABA', 'JD', 'BIDU', 'TME', 'NTES', 'PDD']
TAIWAN      = ['TSM', 'UMC']
INDIA       = ['INFY', 'WIT', 'HDB', 'IBN', 'RDY', 'MMYT']
LATAM       = ['MELI', 'VALE', 'PBR', 'BBD']
EUROPE      = ['ASML', 'NVO', 'SAP', 'SHEL', 'BCS']
JAPAN       = ['TM', 'SONY', 'HMC']
SOUTH_KOREA = ['SKM', 'KB']

PROXIES = CHINA_HK + TAIWAN + INDIA + LATAM + EUROPE + JAPAN + SOUTH_KOREA
UNIVERSE = [TARGET] + PROXIES

def ensure_directories():
    os.makedirs('data', exist_ok=True)
    os.makedirs('plots', exist_ok=True)

def fetch_data() -> pd.DataFrame:
    print(f"Downloading data for {len(UNIVERSE)} tickers (Target + Global ADRs)...")
    data = yf.download(UNIVERSE, start=START_DATE, end=END_DATE, progress=False)
    return data['Close']

def plot_price_history(prices: pd.DataFrame):
    print("Generating Plot 1: Price History...")
    n_cols = 4
    n_rows = int(np.ceil(prices.shape[1] / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 2.5 * n_rows), sharex=True)
    for ax, col in zip(axes.flat, prices.columns):
        valid_start = prices[col].first_valid_index()
        if valid_start is not None:
            rebased = prices[col] / prices[col].loc[valid_start] * 100
            ax.plot(rebased, linewidth=0.8, color='steelblue')
            
        ax.set_title(col, fontsize=10, fontweight='bold')
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
    
    for i in range(prices.shape[1], len(axes.flat)):
        axes.flat[i].set_visible(False)
        
    plt.suptitle("Normalised Price Levels (Base = 100, 2014–2023)", fontsize=14, y=0.99)
    plt.tight_layout()
    plt.savefig("plots/01_price_history.png", dpi=150, bbox_inches='tight')
    plt.close()

def plot_missing_data(prices: pd.DataFrame, title_suffix: str, filename: str):
    print(f"Generating Missing Data Heatmap ({title_suffix})...")
    missing = prices.isnull().astype(int)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(missing.T, cbar=False, cmap=['white', 'crimson'], ax=ax, xticklabels=False)
    ax.set_title(f"Missing Data Map — Red = Missing {title_suffix}", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"plots/{filename}", dpi=150, bbox_inches='tight')
    plt.close()

def clean_and_compute_returns(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Executing Date Alignment (Handling Holidays)...")
    
    # 1. Forward-fill up to 5 days to handle disparate international market holidays
    prices_filled = prices.ffill(limit=5)
    
    # 2. Drop any assets that still have NaNs (e.g., late IPOs like BABA, PDD)
    prices_cleaned = prices_filled.dropna(axis=1)
    dropped_tickers = set(prices_filled.columns) - set(prices_cleaned.columns)
    
    if dropped_tickers:
        print(f"WARNING: Dropped {len(dropped_tickers)} tickers due to short trading history: {dropped_tickers}")
    
    if TARGET not in prices_cleaned.columns:
        raise ValueError(f"CRITICAL ERROR: Target benchmark {TARGET} was dropped.")
    
    # 3. Calculate continuous log returns
    log_returns = np.log(prices_cleaned).diff().dropna()
    return prices_cleaned, log_returns

def run_data_pipeline():
    print("\n--- Starting Milestone 1: Data Pipeline (Global ADR Universe) ---")
    ensure_directories()
    raw_prices = fetch_data()
    plot_price_history(raw_prices)
    plot_missing_data(raw_prices, "(Pre-fill)", "02_missing_data.png")
    
    clean_prices, log_returns = clean_and_compute_returns(raw_prices)
    clean_prices.to_csv("data/raw_prices.csv")
    log_returns.to_csv("data/log_returns.csv")
    
    print(f"Milestone 1 Complete. Final Return Matrix Shape: {log_returns.shape[0]} days, {log_returns.shape[1]} assets.\n")