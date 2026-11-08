"""Point-in-time security eligibility and prior-only liquidity screening."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "ticker",
    "issuer_id",
    "exchange",
    "security_type",
    "first_trade_date",
    "last_trade_date",
    "sp500_member_at_start",
}


def load_manifest(path: str | Path) -> pd.DataFrame:
    manifest = pd.read_csv(path, parse_dates=["first_trade_date", "last_trade_date"])
    missing = REQUIRED_COLUMNS - set(manifest.columns)
    if missing:
        raise ValueError(f"Universe manifest missing columns: {sorted(missing)}")
    if manifest["ticker"].duplicated().any():
        raise ValueError("Universe manifest contains duplicate tickers")
    if manifest["issuer_id"].duplicated().any():
        raise ValueError("Universe manifest must contain one security per issuer")
    return manifest.set_index("ticker", drop=False)


def eligible_at(
    manifest: pd.DataFrame,
    as_of: pd.Timestamp,
    exchanges: tuple[str, ...],
    allowed_security_types: tuple[str, ...],
) -> pd.DataFrame:
    as_of = pd.Timestamp(as_of)
    return manifest.loc[
        manifest["exchange"].isin(exchanges)
        & manifest["security_type"].isin(allowed_security_types)
        & (manifest["first_trade_date"] <= as_of)
        & (manifest["last_trade_date"].isna() | (manifest["last_trade_date"] >= as_of))
        & ~manifest["sp500_member_at_start"].astype(bool)
    ].copy()


def screen_liquidity(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    eligible: pd.DataFrame,
    as_of: pd.Timestamp,
    lookback_days: int,
    min_history_days: int,
    min_median_dollar_volume: float,
    min_valid_volume_fraction: float,
) -> pd.DataFrame:
    prior_index = prices.index[prices.index < pd.Timestamp(as_of)]
    prior_index = prior_index[-lookback_days:]
    prior_prices = prices.reindex(prior_index)
    prior_volumes = volumes.reindex(prior_index)
    accepted: list[dict[str, object]] = []

    for ticker in eligible.index:
        if ticker not in prior_prices or ticker not in prior_volumes:
            continue
        dollar_volume = (prior_prices[ticker] * prior_volumes[ticker]).dropna()
        if len(dollar_volume) < min_history_days:
            continue
        if len(dollar_volume) / lookback_days < min_valid_volume_fraction:
            continue
        median_volume = float(dollar_volume.median())
        if median_volume >= min_median_dollar_volume:
            accepted.append(
                {
                    "ticker": ticker,
                    "median_dollar_volume": median_volume,
                    "valid_volume_fraction": len(dollar_volume) / lookback_days,
                }
            )

    if not accepted:
        return eligible.iloc[0:0].copy()
    return eligible.join(pd.DataFrame(accepted).set_index("ticker"), how="inner")
