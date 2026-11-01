"""Single source of truth for pipeline parameters.

Loads ``configs/main.toml`` (Research Specification v1.2.1 — authoritative)
into a frozen dataclass tree and validates the spec's hard invariants so a
mis-configured run fails before any data is touched.

S1+ stages consume this module. Legacy modules still carry their own
constants until they are rewritten; this file does not change their behavior.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "main.toml"

_ALLOWED_ESTIMATION_DAYS = (1260, 756)   # Spec §3.1: 5y primary, 3y ablation
_ALLOWED_REBALANCE = ("monthly", "quarterly", "annual")   # Spec §3.2/§9
_ALLOWED_K_SELECTION = ("one_se",)       # F-10 / Spec §8.3: one-standard-error rule


@dataclass(frozen=True)
class PathsConfig:
    raw_dir: str
    derived_dir: str
    legacy_dir: str
    plots_dir: str
    models_dir: str


@dataclass(frozen=True)
class BenchmarkConfig:
    primary: str
    price_check: str
    investable_check: str
    annual_return_tolerance_pp: float
    official_annual_returns: dict[str, dict[str, float]]


@dataclass(frozen=True)
class DownloadConfig:
    start: str
    end: str


@dataclass(frozen=True)
class UniverseConfig:
    min_history_days: int
    min_median_dollar_volume: float


@dataclass(frozen=True)
class EstimationConfig:
    cash_park_return: float


@dataclass(frozen=True)
class FoldConfig:
    estimation_days: int
    validation_days: int
    test_days: int
    step_days: int
    first_test_year: int
    last_test_year: int


@dataclass(frozen=True)
class OptimizerConfig:
    k_set: tuple[int, ...]
    w_max: float
    zero_tol: float
    k_selection: str


@dataclass(frozen=True)
class CostConfig:
    em_adr_bps: float
    dm_adr_bps: float
    em_adrs: tuple[str, ...]
    dm_adrs: tuple[str, ...]


@dataclass(frozen=True)
class RebalanceConfig:
    frequency: str


@dataclass(frozen=True)
class DataQualityConfig:
    interruption_max_days: int


@dataclass(frozen=True)
class BootstrapConfig:
    n_draws: int
    block_days: int
    ci_level: float


@dataclass(frozen=True)
class SeedsConfig:
    main: int
    sae_seeds: tuple[int, ...]


@dataclass(frozen=True)
class PipelineConfig:
    paths: PathsConfig
    benchmark: BenchmarkConfig
    download: DownloadConfig
    universe: UniverseConfig
    estimation: EstimationConfig
    folds: FoldConfig
    optimizer: OptimizerConfig
    costs: CostConfig
    rebalance: RebalanceConfig
    data_quality: DataQualityConfig
    bootstrap: BootstrapConfig
    seeds: SeedsConfig


def load_config(path: str | Path | None = None) -> PipelineConfig:
    """Load and validate the pipeline configuration (Spec v1.2.1 §1–§10)."""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    try:
        with open(cfg_path, "rb") as fh:
            raw = tomllib.load(fh)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(
            f"Configuration file is not valid TOML: {cfg_path} ({exc})"
        ) from exc

    cfg = PipelineConfig(
        paths=PathsConfig(**raw["paths"]),
        benchmark=BenchmarkConfig(**raw["benchmark"]),
        download=DownloadConfig(**raw["download"]),
        universe=UniverseConfig(**raw["universe"]),
        estimation=EstimationConfig(
            cash_park_return=float(raw["estimation"]["cash_park_return"]),
        ),
        folds=FoldConfig(**raw["folds"]),
        optimizer=OptimizerConfig(
            k_set=tuple(int(k) for k in raw["optimizer"]["k_set"]),
            w_max=float(raw["optimizer"]["w_max"]),
            zero_tol=float(raw["optimizer"]["zero_tol"]),
            k_selection=str(raw["optimizer"]["k_selection"]),
        ),
        costs=CostConfig(
            em_adr_bps=float(raw["costs"]["em_adr_bps"]),
            dm_adr_bps=float(raw["costs"]["dm_adr_bps"]),
            em_adrs=tuple(raw["costs"]["em_adrs"]),
            dm_adrs=tuple(raw["costs"]["dm_adrs"]),
        ),
        rebalance=RebalanceConfig(**raw["rebalance"]),
        data_quality=DataQualityConfig(
            interruption_max_days=int(raw["data_quality"]["interruption_max_days"]),
        ),
        bootstrap=BootstrapConfig(
            n_draws=int(raw["bootstrap"]["n_draws"]),
            block_days=int(raw["bootstrap"]["block_days"]),
            ci_level=float(raw["bootstrap"]["ci_level"]),
        ),
        seeds=SeedsConfig(
            main=int(raw["seeds"]["main"]),
            sae_seeds=tuple(int(s) for s in raw["seeds"]["sae_seeds"]),
        ),
    )
    _validate(cfg)
    return cfg


def _validate(cfg: PipelineConfig) -> None:
    """Hard invariants from Research Specification v1.2.1. Fail fast, fail loud."""
    o, f, b, d = cfg.optimizer, cfg.folds, cfg.benchmark, cfg.download

    # -- Optimizer (Spec §1.3, §1.4, §5) --------------------------------
    if list(o.k_set) != sorted(set(o.k_set)):
        raise ValueError("optimizer.k_set must be strictly ascending and unique")
    if any(k < 7 for k in o.k_set):
        raise ValueError(
            "optimizer.k_set entries must be >= 7 (cap-feasibility floor, Spec §1.3)"
        )
    if min(o.k_set) * o.w_max < 1.0:
        raise ValueError("every K candidate must satisfy K * w_max >= 1 (Spec §1.3)")
    if o.w_max != 0.15:
        raise ValueError("optimizer.w_max must be 0.15 (fixed cap; no relaxed caps, Spec §1.3)")
    if not (0 < o.zero_tol < 1e-9):
        raise ValueError(
            "optimizer.zero_tol must lie in (0, 1e-9); it is a numerical zero, not a weight threshold"
        )
    if o.k_selection not in _ALLOWED_K_SELECTION:
        raise ValueError(f"optimizer.k_selection must be one of {_ALLOWED_K_SELECTION}")

    # -- Fold geometry (Spec §3.1) --------------------------------------
    if f.estimation_days not in _ALLOWED_ESTIMATION_DAYS:
        raise ValueError(
            f"folds.estimation_days must be one of {_ALLOWED_ESTIMATION_DAYS} (Spec §3.1)"
        )
    if (f.validation_days, f.test_days, f.step_days) != (252, 252, 252):
        raise ValueError("folds.validation_days/test_days/step_days must be 252 (Spec §3.1)")
    if f.first_test_year >= f.last_test_year:
        raise ValueError("folds.first_test_year must precede last_test_year")

    # -- Download window (Spec §2) --------------------------------------
    start = date.fromisoformat(d.start)
    end = date.fromisoformat(d.end)
    if start >= end:
        raise ValueError("download.start must precede download.end")
    if start.year > f.first_test_year - 7:
        raise ValueError(
            "download.start must reach at least 7 calendar years before the first "
            "test year (W_refit = 1512 observations + listing buffer, Spec §2/§3.3)"
        )
    if end.year < f.last_test_year:
        raise ValueError("download.end must cover the last test year")

    # -- Estimation (Spec §1.2, F-3) ------------------------------------
    if cfg.estimation.cash_park_return != 0.0:
        raise ValueError(
            "estimation.cash_park_return must be 0.0 (zero-return parking, pre-first-return days only)"
        )

    # -- Universe eligibility (Spec §2, F-3 amendment) ------------------
    if cfg.universe.min_history_days < 1:
        raise ValueError("universe.min_history_days must be >= 1")
    if cfg.universe.min_median_dollar_volume <= 0:
        raise ValueError("universe.min_median_dollar_volume must be > 0")

    # -- Costs (Spec §4.4) ----------------------------------------------
    if cfg.costs.em_adr_bps <= 0 or cfg.costs.dm_adr_bps <= 0:
        raise ValueError("costs must be positive")
    overlap = set(cfg.costs.em_adrs) & set(cfg.costs.dm_adrs)
    if overlap:
        raise ValueError(f"cost classification overlap: {sorted(overlap)}")

    # -- Benchmark (Spec §2) --------------------------------------------
    if b.primary != "^SP500TR":
        raise ValueError("benchmark.primary must be the S&P 500 Total Return index (Spec §2)")
    if b.annual_return_tolerance_pp <= 0:
        raise ValueError("benchmark.annual_return_tolerance_pp must be positive")

    # -- Rebalancing / data quality / bootstrap / seeds ------------------
    if cfg.rebalance.frequency not in _ALLOWED_REBALANCE:
        raise ValueError(f"rebalance.frequency must be one of {_ALLOWED_REBALANCE}")
    if cfg.data_quality.interruption_max_days < 1:
        raise ValueError("data_quality.interruption_max_days must be >= 1 (case D-1 threshold)")
    if cfg.bootstrap.n_draws < 1000 or cfg.bootstrap.block_days <= 0:
        raise ValueError("bootstrap requires n_draws >= 1000 and block_days > 0")
    if not (0 < cfg.bootstrap.ci_level < 100):
        raise ValueError("bootstrap.ci_level must lie in (0, 100); all intervals are 90% (Spec §8.0)")
    if len(cfg.seeds.sae_seeds) < 10:
        raise ValueError("A3 protocol requires >= 10 SAE seeds (Spec §5.6)")
