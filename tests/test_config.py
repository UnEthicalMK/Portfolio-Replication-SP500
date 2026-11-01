"""S0 acceptance: config faithfully centralizes Research Specification v1.2.1."""

import pytest

from src.config import DEFAULT_CONFIG_PATH, load_config


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_spec_benchmark(cfg):
    # Spec §2: total return is the target; price index is a cross-check only.
    assert cfg.benchmark.primary == "^SP500TR"
    assert cfg.benchmark.price_check == "^GSPC"
    assert cfg.benchmark.investable_check == "SPY"
    assert cfg.benchmark.annual_return_tolerance_pp == 0.3
    assert cfg.benchmark.official_annual_returns["2023"]["total"] == pytest.approx(26.29)
    assert cfg.benchmark.official_annual_returns["2023"]["price"] == pytest.approx(24.23)


def test_spec_download_window(cfg):
    # Spec §2: 2007-01-01 -> 2023-12-31; covers W_refit(2014) = 2007-12-31 -> 2013-12-31.
    assert cfg.download.start == "2007-01-01"
    assert cfg.download.end == "2023-12-31"
    assert int(cfg.download.start[:4]) <= cfg.folds.first_test_year - 7
    assert int(cfg.download.end[:4]) >= cfg.folds.last_test_year


def test_spec_fold_geometry(cfg):
    # Spec §3.1: 5y estimation, 1y validation, annual roll, tests 2014-2023.
    assert cfg.folds.estimation_days == 1260
    assert cfg.folds.validation_days == 252
    assert cfg.folds.test_days == 252
    assert cfg.folds.step_days == 252
    assert (cfg.folds.first_test_year, cfg.folds.last_test_year) == (2014, 2023)


def test_spec_optimizer_constraints(cfg):
    # Spec §1.3/§1.4/§5: K candidates, fixed cap, numerical zero, 1-SE selection.
    assert cfg.optimizer.k_set == (7, 10, 15, 20)          # F-1 (K=5 removed: 5*0.15 < 1)
    assert cfg.optimizer.w_max == 0.15                      # fixed cap; no relaxed caps
    assert cfg.optimizer.zero_tol == 1e-12                  # numerical zero, NOT a threshold
    assert 0 < cfg.optimizer.zero_tol < 1e-9
    assert cfg.optimizer.k_selection == "one_se"            # F-10
    # every K candidate is cap-feasible: K * w_max >= 1 (v1.2.1 §1.3)
    assert min(cfg.optimizer.k_set) * cfg.optimizer.w_max >= 1.0


def test_spec_estimation_cash_parking(cfg):
    # Spec §1.2 / F-3: zero-return parking, pre-first-return days only.
    assert cfg.estimation.cash_park_return == 0.0


def test_spec_costs(cfg):
    # Spec §4.4: per-side bps; classification disjoint; ledgers split in S3.
    assert cfg.costs.em_adr_bps == 15
    assert cfg.costs.dm_adr_bps == 10
    assert not (set(cfg.costs.em_adrs) & set(cfg.costs.dm_adrs))


def test_spec_universe_eligibility(cfg):
    # Spec §2: eligibility floors (F-3 amendment).
    assert cfg.universe.min_history_days == 252
    assert cfg.universe.min_median_dollar_volume == 5_000_000


def test_validation_rejects_zero_min_history(tmp_path):
    broken = DEFAULT_CONFIG_PATH.read_bytes().replace(
        b"min_history_days = 252", b"min_history_days = 0"
    )
    p = tmp_path / "broken.toml"
    p.write_bytes(broken)
    with pytest.raises(ValueError, match="min_history_days"):
        load_config(p)


def test_validation_rejects_negative_volume_threshold(tmp_path):
    broken = DEFAULT_CONFIG_PATH.read_bytes().replace(
        b"min_median_dollar_volume = 5000000", b"min_median_dollar_volume = -1"
    )
    p = tmp_path / "broken.toml"
    p.write_bytes(broken)
    with pytest.raises(ValueError, match="min_median_dollar_volume"):
        load_config(p)


def test_spec_data_quality(cfg):
    # Spec §2.3 (DEC-2): temporary-interruption threshold.
    assert cfg.data_quality.interruption_max_days == 10


def test_spec_bootstrap_and_ci(cfg):
    # Spec §8: B, block length, and the 90% interval convention (§8.0).
    assert cfg.bootstrap.n_draws == 2000
    assert cfg.bootstrap.block_days == 21
    assert cfg.bootstrap.ci_level == 90
    assert 0 < cfg.bootstrap.ci_level < 100


def test_spec_seeds(cfg):
    assert cfg.seeds.main == 42
    assert len(cfg.seeds.sae_seeds) >= 10   # A3 protocol


def test_validation_rejects_infeasible_k_candidate(tmp_path):
    # Pins the A-1 mathematics: K=5 under w_max=0.15 can never reach 1'w = 1.
    broken = DEFAULT_CONFIG_PATH.read_bytes().replace(
        b"k_set = [7, 10, 15, 20]", b"k_set = [5, 10, 15, 20]"
    )
    p = tmp_path / "broken.toml"
    p.write_bytes(broken)
    with pytest.raises(ValueError, match="cap-feasibility|w_max"):
        load_config(p)


def test_validation_rejects_bad_fold_geometry(tmp_path):
    broken = DEFAULT_CONFIG_PATH.read_bytes().replace(
        b"estimation_days = 1260", b"estimation_days = 1000"
    )
    p = tmp_path / "broken.toml"
    p.write_bytes(broken)
    with pytest.raises(ValueError, match="estimation_days"):
        load_config(p)


def test_validation_rejects_bad_weight_cap(tmp_path):
    broken = DEFAULT_CONFIG_PATH.read_bytes().replace(b"w_max = 0.15", b"w_max = 1.2")
    p = tmp_path / "broken.toml"
    p.write_bytes(broken)
    with pytest.raises(ValueError, match="w_max"):
        load_config(p)


def test_validation_rejects_price_index_as_primary(tmp_path):
    # Guards the D-4 regression class: the price index must never be the target.
    broken = DEFAULT_CONFIG_PATH.read_bytes().replace(
        b'primary = "^SP500TR"', b'primary = "^GSPC"'
    )
    p = tmp_path / "broken.toml"
    p.write_bytes(broken)
    with pytest.raises(ValueError, match="Total Return"):
        load_config(p)


def test_validation_rejects_overlapping_cost_classes(tmp_path):
    broken = DEFAULT_CONFIG_PATH.read_bytes().replace(
        b'dm_adrs = ["ASML", "NVO", "SAP", "SHEL", "BCS", "TM", "SONY", "HMC"]',
        b'dm_adrs = ["ASML", "NVO", "SAP", "SHEL", "BCS", "TM", "SONY", "HMC", "TSM"]',
    )
    p = tmp_path / "broken.toml"
    p.write_bytes(broken)
    with pytest.raises(ValueError, match="overlap"):
        load_config(p)


def test_validation_rejects_nonzero_cash_park_return(tmp_path):
    # F-3: parking is zero-return; anything else silently changes the objective.
    broken = DEFAULT_CONFIG_PATH.read_bytes().replace(
        b"cash_park_return = 0.0", b"cash_park_return = 0.02"
    )
    p = tmp_path / "broken.toml"
    p.write_bytes(broken)
    with pytest.raises(ValueError, match="cash_park"):
        load_config(p)


def test_validation_rejects_insufficient_download_window(tmp_path):
    broken = DEFAULT_CONFIG_PATH.read_bytes().replace(
        b'start = "2007-01-01"', b'start = "2012-01-01"'
    )
    p = tmp_path / "broken.toml"
    p.write_bytes(broken)
    with pytest.raises(ValueError, match="download.start"):
        load_config(p)


def test_validation_rejects_out_of_range_ci_level(tmp_path):
    broken = DEFAULT_CONFIG_PATH.read_bytes().replace(b"ci_level = 90", b"ci_level = 101")
    p = tmp_path / "broken.toml"
    p.write_bytes(broken)
    with pytest.raises(ValueError, match="ci_level"):
        load_config(p)


def test_validation_rejects_zero_interruption_threshold(tmp_path):
    broken = DEFAULT_CONFIG_PATH.read_bytes().replace(
        b"interruption_max_days = 10", b"interruption_max_days = 0"
    )
    p = tmp_path / "broken.toml"
    p.write_bytes(broken)
    with pytest.raises(ValueError, match="interruption_max_days"):
        load_config(p)
