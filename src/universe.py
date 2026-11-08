"""Point-in-time eligibility and liquidity screening for the research universe."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_manifest(path: str | Path) -> pd.DataFrame:
    """Load and validate the versioned security manifest."""
    manifest = pd.read_csv(path, parse_dates=["first_trade_date", "last_trade_date"])
    required = {
        "ticker",
        "issuer_id",
        "exchange",
        "security_type",
        "first_trade_date",
        "last_trade_date",
        "sp500_member_at_start",
    }
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Universe manifest missing columns: {sorted(missing)}")
    if manifest["ticker"].duplicated().any():
        raise ValueError("Universe manifest contains duplicate tickers")
    if manifest["issuer_id"].duplicated().any():
        raise ValueError(
            "Universe manifest contains duplicate issuers; choose one security per issuer"
        )
    return manifest.set_index("ticker", drop=False)


def eligible_at(
    manifest: pd.DataFrame,
    as_of: pd.Timestamp,
    exchanges: tuple[str, ...],
    allowed_security_types: tuple[str, ...],
) -> pd.DataFrame:
    """Apply non-price eligibility rules using information known at ``as_of``."""
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
    as_of: pd.Timestamp,
    eligible: pd.DataFrame,
    lookback_days: int,
    min_history_days: int,
    min_median_dollar_volume: float,
    min_valid_volume_fraction: float,
) -> pd.DataFrame:
    """Apply a prior-only median dollar-volume and history screen."""
    prior_prices = prices.loc[prices.index <= pd.Timestamp(as_of)].tail(lookback_days)
    prior_volumes = volumes.reindex(prior_prices.index).tail(lookback_days)
    rows: list[dict[str, object]] = []

    for ticker in eligible.index:
        if ticker not in prior_prices or ticker not in prior_volumes:
            continue
        valid = (prior_prices[ticker] * prior_volumes[ticker]).dropna()
        if len(valid) < min_history_days:
            continue
        if len(valid) / lookback_days < min_valid_volume_fraction:
            continue
        median_dollar_volume = float(valid.median())
        if median_dollar_volume >= min_median_dollar_volume:
            rows.append(
                {
                    "ticker": ticker,
                    "median_dollar_volume": median_dollar_volume,
                    "valid_volume_fraction": len(valid) / lookback_days,
                }
            )

    if not rows:
        return eligible.iloc[0:0].copy()
    return eligible.join(pd.DataFrame(rows).set_index("ticker"), how="inner")
