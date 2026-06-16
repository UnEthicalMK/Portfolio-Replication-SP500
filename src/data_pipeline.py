import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')

# Configuration
START_DATE = '2014-01-01'
END_DATE = '2023-12-31'
TARGET = '^GSPC'

# Global ADR universe
CHINA_HK = ['BABA', 'JD', 'BIDU', 'TME', 'NTES', 'PDD']
TAIWAN = ['TSM', 'UMC']
INDIA = ['INFY', 'WIT', 'HDB', 'IBN', 'RDY', 'MMYT']
LATAM = ['MELI', 'VALE', 'PBR', 'BBD']
EUROPE = ['ASML', 'NVO', 'SAP', 'SHEL', 'BCS']
JAPAN = ['TM', 'SONY', 'HMC']
SOUTH_KOREA = ['SKM', 'KB']

PROXIES = CHINA_HK + TAIWAN + INDIA + LATAM + EUROPE + JAPAN + SOUTH_KOREA
UNIVERSE = [TARGET] + PROXIES

def ensure_directories():
    os.makedirs('data', exist_ok=True)
    os.makedirs('plots', exist_ok=True)

def fetch_data() -> pd.DataFrame:
    print(f"[INFO] Downloading price data for {len(UNIVERSE)} instruments...")

    data = yf.download(
        UNIVERSE,
        start=START_DATE,
        end=END_DATE,
        progress=False
    )

    if data.empty:
        raise ValueError("No data returned from Yahoo Finance.")

    return data['Close']

def plot_price_history(prices: pd.DataFrame):
    print("[INFO] Generating Plot 1: Normalised Price History")

    n_cols = 4
    n_rows = int(np.ceil(prices.shape[1] / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(16, 2.5 * n_rows),
        sharex=True
    )

    for ax, ticker in zip(axes.flat, prices.columns):
        valid_start = prices[ticker].first_valid_index()

        if valid_start is not None:
            # Rebase each instrument to a common starting value
            rebased = prices[ticker] / prices[ticker].loc[valid_start] * 100
            ax.plot(rebased, linewidth=0.8, color='steelblue')

        ax.set_title(ticker, fontsize=10, fontweight='bold')
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)

    # Hide unused subplot slots
    for i in range(prices.shape[1], len(axes.flat)):
        axes.flat[i].set_visible(False)

    plt.suptitle(
        "Normalised Price Levels (Base = 100, 2014–2023)",
        fontsize=14,
        y=0.99
    )

    plt.tight_layout()
    plt.savefig(
        "plots/01_price_history.png",
        dpi=150,
        bbox_inches='tight'
    )
    plt.close()

def plot_missing_data(
    prices: pd.DataFrame,
    title_suffix: str,
    filename: str
):
    print("[INFO] Generating Plot 2: Missing Data Heatmap")

    missing = prices.isnull().astype(int)

    fig, ax = plt.subplots(figsize=(14, 8))

    sns.heatmap(
        missing.T,
        cbar=False,
        cmap=['white', 'crimson'],
        xticklabels=False,
        ax=ax
    )

    ax.set_title(
        f"Missing Data Map — Red = Missing {title_suffix}",
        fontsize=12,
        fontweight='bold'
    )

    plt.tight_layout()
    plt.savefig(
        f"plots/{filename}",
        dpi=150,
        bbox_inches='tight'
    )
    plt.close()

def clean_and_compute_returns(
    prices: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:

    print("[INFO] Aligning international trading calendars...")

    # Fill short holiday gaps across markets
    prices_filled = prices.ffill(limit=5)

    # Remove instruments without a complete history
    prices_cleaned = prices_filled.dropna(axis=1)

    dropped_tickers = (
        set(prices_filled.columns)
        - set(prices_cleaned.columns)
    )

    if dropped_tickers:
        print(
            f"[WARNING] Removed {len(dropped_tickers)} instruments "
            f"with incomplete histories: {sorted(dropped_tickers)}"
        )

    if TARGET not in prices_cleaned.columns:
        raise ValueError(
            f"Benchmark {TARGET} was removed during cleaning."
        )

    # Compute continuously compounded daily returns
    log_returns = np.log(prices_cleaned).diff().dropna()

    return prices_cleaned, log_returns

def run_data_pipeline():
    print("\nMilestone 1 | Data Ingestion & Preprocessing")

    ensure_directories()

    raw_prices = fetch_data()

    plot_price_history(raw_prices)
    plot_missing_data(
        raw_prices,
        "(Pre-Fill)",
        "02_missing_data.png"
    )

    clean_prices, log_returns = clean_and_compute_returns(raw_prices)

    clean_prices.to_csv("data/raw_prices.csv")
    log_returns.to_csv("data/log_returns.csv")

    print(
        f"[SUCCESS] Final return matrix: "
        f"{log_returns.shape[0]} days × "
        f"{log_returns.shape[1]} instruments"
    )