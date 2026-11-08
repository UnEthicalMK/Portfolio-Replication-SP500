"""Validated configuration for the research workflow."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "main.toml"


@dataclass(frozen=True)
class BenchmarkConfig:
    primary: str
    price_check: str
    investable_check: str


@dataclass(frozen=True)
class DownloadConfig:
    start: str
    end: str


@dataclass(frozen=True)
class UniverseConfig:
    manifest_path: str
    exchanges: tuple[str, ...]
    allowed_security_types: tuple[str, ...]
    min_history_days: int
    min_median_dollar_volume: float
    liquidity_lookback_days: int
    min_valid_volume_fraction: float


@dataclass(frozen=True)
class FoldConfig:
    estimation_days: int
    validation_days: int
    test_days: int
    first_test_year: int
    last_test_year: int


@dataclass(frozen=True)
class OptimizerConfig:
    k_set: tuple[int, ...]
    w_max: float
    alphas: tuple[float, ...]


@dataclass(frozen=True)
class CostConfig:
    default_bps: float


@dataclass(frozen=True)
class SAEConfig:
    latent_dim: int
    dropout: float
    noise_sigma: float
    reconstruction_weight: float
    l1_weight: float
    learning_rate: float
    epochs: int
    patience: int
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class PipelineConfig:
    benchmark: BenchmarkConfig
    download: DownloadConfig
    universe: UniverseConfig
    folds: FoldConfig
    optimizer: OptimizerConfig
    costs: CostConfig
    sae: SAEConfig
    output_dir: str


def load_config(path: str | Path | None = None) -> PipelineConfig:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with cfg_path.open("rb") as handle:
        raw = tomllib.load(handle)
    cfg = PipelineConfig(
        benchmark=BenchmarkConfig(**raw["benchmark"]),
        download=DownloadConfig(**raw["download"]),
        universe=UniverseConfig(
            manifest_path=str(raw["universe"]["manifest_path"]),
            exchanges=tuple(raw["universe"]["exchanges"]),
            allowed_security_types=tuple(raw["universe"]["allowed_security_types"]),
            min_history_days=int(raw["universe"]["min_history_days"]),
            min_median_dollar_volume=float(raw["universe"]["min_median_dollar_volume"]),
            liquidity_lookback_days=int(raw["universe"]["liquidity_lookback_days"]),
            min_valid_volume_fraction=float(raw["universe"]["min_valid_volume_fraction"]),
        ),
        folds=FoldConfig(**raw["folds"]),
        optimizer=OptimizerConfig(
            k_set=tuple(int(k) for k in raw["optimizer"]["k_set"]),
            w_max=float(raw["optimizer"]["w_max"]),
            alphas=tuple(float(alpha) for alpha in raw["optimizer"]["alphas"]),
        ),
        costs=CostConfig(default_bps=float(raw["costs"]["default_bps"])),
        sae=SAEConfig(
            latent_dim=int(raw["sae"]["latent_dim"]),
            dropout=float(raw["sae"]["dropout"]),
            noise_sigma=float(raw["sae"]["noise_sigma"]),
            reconstruction_weight=float(raw["sae"]["reconstruction_weight"]),
            l1_weight=float(raw["sae"]["l1_weight"]),
            learning_rate=float(raw["sae"]["learning_rate"]),
            epochs=int(raw["sae"]["epochs"]),
            patience=int(raw["sae"]["patience"]),
            seeds=tuple(int(seed) for seed in raw["sae"]["seeds"]),
        ),
        output_dir=str(raw["outputs"]["directory"]),
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: PipelineConfig) -> None:
    if cfg.benchmark.primary != "^SP500TR":
        raise ValueError("The primary benchmark must be ^SP500TR")
    if not cfg.universe.exchanges or not cfg.universe.allowed_security_types:
        raise ValueError("Universe exchange and instrument filters are required")
    if cfg.universe.min_history_days < 1:
        raise ValueError("min_history_days must be positive")
    if cfg.universe.liquidity_lookback_days < cfg.universe.min_history_days:
        raise ValueError("liquidity lookback must cover minimum history")
    if not 0 < cfg.universe.min_valid_volume_fraction <= 1:
        raise ValueError("min_valid_volume_fraction must be in (0, 1]")
    if not cfg.optimizer.k_set or tuple(sorted(set(cfg.optimizer.k_set))) != cfg.optimizer.k_set:
        raise ValueError("optimizer.k_set must be sorted and unique")
    if any(k * cfg.optimizer.w_max < 1 for k in cfg.optimizer.k_set):
        raise ValueError("Every K must support a fully invested capped portfolio")
    if not 0 < cfg.optimizer.w_max <= 1:
        raise ValueError("w_max must be in (0, 1]")
    if not cfg.optimizer.alphas or any(alpha <= 0 for alpha in cfg.optimizer.alphas):
        raise ValueError("optimizer.alphas must be positive")
    if cfg.costs.default_bps < 0:
        raise ValueError("default_bps cannot be negative")
    if cfg.sae.latent_dim < 1 or not 0 <= cfg.sae.dropout < 1:
        raise ValueError("SAE latent_dim and dropout are invalid")
    if cfg.sae.noise_sigma < 0 or cfg.sae.reconstruction_weight < 0 or cfg.sae.l1_weight < 0:
        raise ValueError("SAE regularization values cannot be negative")
    if cfg.sae.learning_rate <= 0 or cfg.sae.epochs < 1 or cfg.sae.patience < 1:
        raise ValueError("SAE training settings are invalid")
    if not cfg.sae.seeds:
        raise ValueError("At least one SAE seed is required")
    if cfg.folds.first_test_year > cfg.folds.last_test_year:
        raise ValueError("first_test_year must not exceed last_test_year")
