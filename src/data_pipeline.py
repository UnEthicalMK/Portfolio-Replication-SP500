"""Market-data download and return preparation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from src.config import PipelineConfig
from src.universe import eligible_at, load_manifest, screen_liquidity


def _field(downloaded: pd.DataFrame, field: str) -> pd.DataFrame:
    if isinstance(downloaded.columns, pd.MultiIndex):
        if field in downloaded.columns.get_level_values(0):
            return downloaded[field]
        if field in downloaded.columns.get_level_values(1):
            return downloaded.xs(field, axis=1, level=1)
    if field in downloaded:
        return downloaded[[field]]
    return pd.DataFrame(index=downloaded.index)


def download_market_data(
    symbols: list[str], start: str, end: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    downloaded = yf.download(
        symbols,
        start=start,
        end=end,
        auto_adjust=False,
        group_by="column",
        progress=False,
    )
    if downloaded.empty:
        raise ValueError("Market-data download returned no observations")
    prices = _field(downloaded, "Adj Close")
    if prices.empty:
        prices = _field(downloaded, "Close")
    volumes = _field(downloaded, "Volume")
    prices = prices.reindex(columns=symbols)
    volumes = volumes.reindex(columns=symbols)
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    volumes.index = pd.to_datetime(volumes.index).tz_localize(None)
    return prices.sort_index(), volumes.sort_index()


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.index.has_duplicates:
        raise ValueError("Price data contains duplicate dates")
    if (prices.dropna() <= 0).any().any():
        raise ValueError("Price data contains non-positive prices")
    return np.log(prices).diff()


def prepare_universe_data(
    cfg: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest = load_manifest(Path(cfg.universe.manifest_path))
    formation_date = pd.Timestamp(f"{cfg.folds.first_test_year}-01-01")
    eligible = eligible_at(
        manifest,
        formation_date,
        cfg.universe.exchanges,
        cfg.universe.allowed_security_types,
    )
    symbols = [cfg.benchmark.primary, *eligible.index.tolist()]
    prices, volumes = download_market_data(
        symbols, cfg.download.start, cfg.download.end
    )
    screened = screen_liquidity(
        prices.drop(columns=[cfg.benchmark.primary], errors="ignore"),
        volumes.drop(columns=[cfg.benchmark.primary], errors="ignore"),
        eligible,
        formation_date,
        cfg.universe.liquidity_lookback_days,
        cfg.universe.min_history_days,
        cfg.universe.min_median_dollar_volume,
        cfg.universe.min_valid_volume_fraction,
    )
    selected = [cfg.benchmark.primary, *screened.index.tolist()]
    returns = compute_log_returns(prices[selected])
    return prices[selected], returns, screened


def save_data(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    manifest: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prices.to_csv(output / "prices.csv")
    returns.to_csv(output / "returns.csv")
    manifest.to_csv(output / "eligible_universe.csv")
