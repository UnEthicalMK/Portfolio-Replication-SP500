import pandas as pd
import pytest

from src.universe import eligible_at, load_manifest, screen_liquidity


def test_manifest_rejects_duplicate_issuers(tmp_path):
    path = tmp_path / "universe.csv"
    path.write_text(
        "ticker,issuer_id,exchange,security_type,first_trade_date,last_trade_date,sp500_member_at_start\n"
        "AAA,issuer,NASDAQ,common_stock,2010-01-01,,False\n"
        "AAB,issuer,NASDAQ,common_stock,2010-01-01,,False\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate issuers"):
        load_manifest(path)


def test_eligibility_is_point_in_time():
    manifest = pd.DataFrame(
        {
            "ticker": ["OLD", "NEW", "SPX"],
            "issuer_id": ["old", "new", "spx"],
            "exchange": ["NYSE", "NYSE", "NYSE"],
            "security_type": ["common_stock", "preferred", "common_stock"],
            "first_trade_date": pd.to_datetime(["2010-01-01", "2020-01-01", "2010-01-01"]),
            "last_trade_date": pd.to_datetime([None, None, None]),
            "sp500_member_at_start": [False, False, True],
        }
    ).set_index("ticker")
    result = eligible_at(
        manifest,
        pd.Timestamp("2015-01-01"),
        ("NYSE",),
        ("common_stock",),
    )
    assert list(result.index) == ["OLD"]


def test_liquidity_uses_only_prior_observations():
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    prices = pd.DataFrame({"AAA": [10, 10, 10, 10, 10]}, index=dates)
    volumes = pd.DataFrame({"AAA": [1_000_000] * 5}, index=dates)
    eligible = pd.DataFrame(index=["AAA"])
    result = screen_liquidity(
        prices,
        volumes,
        pd.Timestamp("2020-01-04"),
        eligible,
        lookback_days=3,
        min_history_days=3,
        min_median_dollar_volume=20_000_000,
        min_valid_volume_fraction=1.0,
    )
    assert result.empty
