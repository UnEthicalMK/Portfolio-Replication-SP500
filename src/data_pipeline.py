"""Market-data ingestion and point-in-time universe preparation."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf

from src.config import load_config
from src.universe import eligible_at, load_manifest, screen_liquidity

plt.style.use("seaborn-v0_8-whitegrid")


def ensure_directories() -> None:
    os.makedirs("data", exist_ok=True)
    os.makedirs("plots", exist_ok=True)


def _adjusted_close(downloaded: pd.DataFrame) -> pd.DataFrame:
    """Return adjusted closes explicitly, with a clear fallback for indices."""
    if isinstance(downloaded.columns, pd.MultiIndex):
        if "Adj Close" in downloaded.columns.get_level_values(0):
            return downloaded["Adj Close"]
        if "Close" in downloaded.columns.get_level_values(0):
            return downloaded["Close"]
    if "Adj Close" in downloaded:
        return downloaded[["Adj Close"]]
    return downloaded[["Close"]]


def fetch_data(
    symbols: list[str],
    start: str,
    end: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download prices and volumes without fabricating missing observations."""
    downloaded = yf.download(
        symbols,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="column",
    )
    if downloaded.empty:
        raise ValueError("No data returned from Yahoo Finance.")

    prices = _adjusted_close(downloaded)
    volumes = downloaded["Volume"] if "Volume" in downloaded else pd.DataFrame(index=prices.index)
    prices = prices.reindex(columns=symbols)
    volumes = volumes.reindex(columns=symbols)
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    volumes.index = pd.to_datetime(volumes.index).tz_localize(None)
    return prices, volumes


def clean_and_compute_returns(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute returns from observed prices; do not forward-fill securities."""
    prices = prices.sort_index()
    if prices.index.has_duplicates:
        raise ValueError("Price data contains duplicate dates.")
    if (prices <= 0).any().any():
        raise ValueError("Price data contains non-positive values.")
    returns = np.log(prices).diff()
    return prices, returns


def _plot_missing_data(prices: pd.DataFrame) -> None:
    missing = prices.isnull().astype(int)
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(missing.T, cbar=False, cmap=["white", "crimson"], xticklabels=False, ax=ax)
    ax.set_title("Missing Data Map — Red = Missing")
    plt.tight_layout()
    plt.savefig("plots/02_missing_data.png", dpi=150, bbox_inches="tight")
    plt.close()


def run_data_pipeline() -> None:
    """Download the configured universe and apply the initial prior-only screen."""
    print("\nMilestone 1 | Data Ingestion & Preprocessing")
    ensure_directories()
    cfg = load_config()
    manifest_path = Path(cfg.universe.manifest_path)
    manifest = load_manifest(manifest_path)
    formation_date = pd.Timestamp(f"{cfg.folds.first_test_year}-01-01") - pd.Timedelta(days=1)
    eligible = eligible_at(
        manifest,
        formation_date,
        cfg.universe.exchanges,
        cfg.universe.allowed_security_types,
    )
    symbols = [cfg.benchmark.primary, *eligible.index.tolist()]
    prices, volumes = fetch_data(symbols, cfg.download.start, cfg.download.end)
    proxy_prices = prices.drop(columns=[cfg.benchmark.primary], errors="ignore")
    proxy_volumes = volumes.drop(columns=[cfg.benchmark.primary], errors="ignore")
    screened = screen_liquidity(
        proxy_prices,
        proxy_volumes,
        formation_date,
        eligible,
        cfg.universe.liquidity_lookback_days,
        cfg.universe.min_history_days,
        cfg.universe.min_median_dollar_volume,
        cfg.universe.min_valid_volume_fraction,
    )
    selected_symbols = [cfg.benchmark.primary, *screened.index.tolist()]
    clean_prices, log_returns = clean_and_compute_returns(prices[selected_symbols])
    clean_prices.to_csv("data/raw_prices.csv")
    log_returns.to_csv("data/log_returns.csv")
    screened.to_csv("data/eligible_universe.csv")
    _plot_missing_data(prices[selected_symbols])
    print(
        f"[SUCCESS] Retained {len(screened)} proxy securities; "
        f"return matrix: {log_returns.shape[0]} days × {log_returns.shape[1]} instruments"
    )
