import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf

from src.config import (
    DATA_DIR,
    END_DATE,
    MAX_FORWARD_FILL_DAYS,
    MIN_TRAIN_COVERAGE,
    PLOTS_DIR,
    START_DATE,
    TARGET,
    TRAIN_END,
    UNIVERSE_FILE,
)

plt.style.use("seaborn-v0_8-whitegrid")


def _ensure_directories() -> None:
    for directory in (DATA_DIR, PLOTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def load_universe() -> pd.DataFrame:
    if not UNIVERSE_FILE.exists():
        raise FileNotFoundError(f"Universe manifest not found: {UNIVERSE_FILE}")
    universe = pd.read_csv(UNIVERSE_FILE)
    required = {"ticker", "asset_type", "listing_region"}
    missing = required.difference(universe.columns)
    if missing:
        raise ValueError(f"Universe manifest is missing columns: {sorted(missing)}")
    if universe["ticker"].duplicated().any():
        duplicates = universe.loc[universe["ticker"].duplicated(), "ticker"].tolist()
        raise ValueError(f"Universe manifest contains duplicate tickers: {duplicates}")
    if TARGET in set(universe["ticker"]):
        raise ValueError(f"Benchmark {TARGET} must not appear in the candidate universe.")
    return universe


def _extract_close(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if raw.empty:
        raise RuntimeError("Market-data provider returned no rows.")
    if not isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns:
            raise RuntimeError("Market-data response has no Close field.")
        return raw[["Close"]].rename(columns={"Close": tickers[0]})

    level_zero = set(raw.columns.get_level_values(0))
    level_one = set(raw.columns.get_level_values(1))
    if "Close" in level_zero:
        close = raw["Close"]
    elif "Close" in level_one:
        close = raw.xs("Close", level=1, axis=1)
    else:
        raise RuntimeError("Market-data response has no Close field.")
    return close.reindex(columns=tickers)


def fetch_data(universe: pd.DataFrame) -> pd.DataFrame:
    tickers = [TARGET, *universe["ticker"].tolist()]
    print(f"  Downloading {len(tickers)} instruments ({START_DATE} to {END_DATE})")
    raw = yf.download(
        tickers,
        start=START_DATE,
        end=pd.Timestamp(END_DATE) + pd.Timedelta(days=1),
        auto_adjust=True,
        group_by="column",
        progress=False,
        threads=False,
    )
    prices = _extract_close(raw, tickers)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    if TARGET not in prices.columns or prices[TARGET].dropna().empty:
        raise RuntimeError(f"Benchmark {TARGET} was not returned by the data provider.")
    return prices


def _write_universe_manifest(prices: pd.DataFrame, universe: pd.DataFrame) -> None:
    train_prices = prices.loc[:TRAIN_END]
    rows = []
    for row in universe.itertuples(index=False):
        ticker = row.ticker
        series = prices[ticker] if ticker in prices else pd.Series(dtype=float)
        coverage = float(series.loc[:TRAIN_END].notna().mean()) if not series.empty else 0.0
        available = int(series.notna().sum()) if not series.empty else 0
        eligible = coverage >= MIN_TRAIN_COVERAGE
        rows.append(
            {
                "ticker": ticker,
                "asset_type": row.asset_type,
                "listing_region": row.listing_region,
                "observations": available,
                "train_coverage": round(coverage, 6),
                "eligible": eligible,
                "reason": "coverage threshold" if eligible else "insufficient pre-validation history",
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(DATA_DIR / "universe_manifest.csv", index=False)
    eligible_count = int(manifest["eligible"].sum())
    print(f"  Candidate universe: {len(manifest)} | eligible after coverage: {eligible_count}")


def clean_and_compute_returns(
    prices: pd.DataFrame, universe: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _write_universe_manifest(prices, universe)
    filled = prices.ffill(limit=MAX_FORWARD_FILL_DAYS)
    train = filled.loc[:TRAIN_END]
    eligible = [
        ticker
        for ticker in universe["ticker"]
        if ticker in filled
        and train[ticker].notna().mean() >= MIN_TRAIN_COVERAGE
    ]
    selected = [TARGET, *eligible]
    cleaned = filled[selected].dropna()
    if len(eligible) < 10:
        raise ValueError(
            f"Only {len(eligible)} candidates passed coverage checks; at least 10 are required."
        )
    if cleaned.empty or cleaned[TARGET].isna().any():
        raise ValueError("Cleaning produced no complete benchmark-aligned price history.")
    log_returns = np.log(cleaned).diff().dropna()
    if not np.isfinite(log_returns.to_numpy()).all():
        raise ValueError("Log-return matrix contains non-finite values.")
    return cleaned, log_returns


def plot_price_history(prices: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    rebased = prices / prices.iloc[0] * 100
    candidates = rebased.drop(columns=TARGET)
    quantiles = candidates.quantile([0.1, 0.25, 0.5, 0.75, 0.9], axis=1).T
    ax.fill_between(
        quantiles.index, quantiles[0.1], quantiles[0.9],
        color="steelblue", alpha=0.18, label="Candidate 10–90th percentile"
    )
    ax.fill_between(
        quantiles.index, quantiles[0.25], quantiles[0.75],
        color="steelblue", alpha=0.28, label="Candidate 25–75th percentile"
    )
    benchmark_line, = ax.plot(
        rebased.index,
        rebased[TARGET],
        linewidth=2.5,
        color="black",
        label="S&P 500 Total Return",
    )
    median_path = quantiles[0.5]
    median_line, = ax.plot(
        median_path.index,
        median_path,
        linewidth=2,
        color="darkorange",
        label="Candidate median",
    )
    ax.set_title("Rebased Candidate Price Distribution vs. Benchmark")
    ax.set_ylabel("Index (first common observation = 100)")
    ax.set_xlabel("Date")
    ax.set_yscale("log")
    ax.set_ylim(bottom=25)
    ax.text(
        0.99,
        0.02,
        "Bands show cross-sectional dispersion; benchmark shown in black",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="dimgray",
    )
    ax.legend(
        handles=[benchmark_line, median_line],
        loc="upper left",
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "01_price_history.png", dpi=150)
    plt.close(fig)


def plot_missing_data(prices: pd.DataFrame) -> None:
    coverage = prices.notna().mean().sort_values(ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1.2, 1]})
    colors = ["crimson" if value < 0.95 else "steelblue" for value in coverage]
    axes[0].barh(coverage.index, coverage, color=colors)
    axes[0].axvline(0.95, color="black", linestyle="--", label="95% eligibility threshold")
    axes[0].set_title("Raw price coverage by ticker")
    axes[0].set_xlabel("Fraction of observations present")
    axes[0].legend()
    daily_missing = prices.isna().sum(axis=1)
    axes[1].plot(daily_missing.index, daily_missing, color="crimson", linewidth=1)
    axes[1].set_title("Missing tickers by trading day")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Number of missing tickers")
    axes[1].set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "02_missing_data.png", dpi=150)
    plt.close(fig)


def plot_quant_eda(log_returns: pd.DataFrame) -> None:
    candidates = log_returns.drop(columns=TARGET)
    annualized_vol = candidates.std() * np.sqrt(252)
    benchmark_corr = candidates.corrwith(log_returns[TARGET])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].hist(
        log_returns[TARGET], bins=60, density=True, alpha=0.8, color="black"
    )
    axes[0].set_title("Benchmark Daily Log-Return Distribution")
    axes[0].set_xlabel("Log return")
    axes[0].set_ylabel("Density")
    axes[1].scatter(annualized_vol, benchmark_corr, color="steelblue")
    for ticker in candidates.columns:
        axes[1].annotate(ticker, (annualized_vol[ticker], benchmark_corr[ticker]), fontsize=7)
    axes[1].set_title("Risk vs. Benchmark Correlation")
    axes[1].set_xlabel("Annualized volatility")
    axes[1].set_ylabel("Correlation with S&P 500 Total Return")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "10_quant_eda_overview.png", dpi=150)
    plt.close(fig)

    rolling_vol = log_returns[TARGET].rolling(63).std() * np.sqrt(252)
    fig, ax = plt.subplots(figsize=(14, 5))
    rolling_vol.plot(ax=ax, color="darkorange")
    ax.set_title("63-Day Rolling Benchmark Volatility")
    ax.set_ylabel("Annualized volatility")
    ax.set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "11_rolling_volatility.png", dpi=150)
    plt.close(fig)

    corr = candidates.corr()
    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title("Candidate Return Correlation Matrix")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "12_candidate_correlation.png", dpi=150)
    plt.close(fig)


def run_data_pipeline() -> None:
    print("\n[1/5] Data acquisition and quant EDA")
    _ensure_directories()
    universe = load_universe()
    raw_prices = fetch_data(universe)
    plot_missing_data(raw_prices)
    clean_prices, log_returns = clean_and_compute_returns(raw_prices, universe)
    plot_price_history(clean_prices)
    plot_quant_eda(log_returns)
    clean_prices.to_csv(DATA_DIR / "raw_prices.csv")
    log_returns.to_csv(DATA_DIR / "log_returns.csv")
    metadata = {
        "benchmark": TARGET,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "candidate_count": len(universe),
        "final_asset_count": len(clean_prices.columns) - 1,
        "forward_fill_limit": MAX_FORWARD_FILL_DAYS,
        "minimum_train_coverage": MIN_TRAIN_COVERAGE,
    }
    (DATA_DIR / "data_manifest.json").write_text(json.dumps(metadata, indent=2))
    print(
        f"  Saved {len(log_returns):,} return rows and "
        f"{len(clean_prices.columns) - 1} eligible candidates"
    )
