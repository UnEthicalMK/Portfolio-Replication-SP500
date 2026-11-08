"""Reproducible plots for walk-forward research results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _require_columns(results: pd.DataFrame) -> None:
    required = {
        "test_year",
        "model",
        "tracking_error",
        "correlation",
        "portfolio_return",
        "benchmark_return",
        "active_holdings",
    }
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Results missing plotting columns: {sorted(missing)}")


def generate_report_plots(
    results: pd.DataFrame,
    output_dir: str | Path,
) -> list[Path]:
    """Create fold-level and aggregate plots from saved evaluation metrics."""
    _require_columns(results)
    if results.empty:
        raise ValueError("Cannot generate plots from empty results")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    tracking_path = output / "tracking_error_by_fold.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    for model, frame in results.groupby("model", sort=False):
        ax.plot(frame["test_year"], frame["tracking_error"] * 100, marker="o", label=model)
    ax.set_title("Out-of-sample annualized tracking error")
    ax.set_xlabel("Test year")
    ax.set_ylabel("Tracking error (%)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(tracking_path, dpi=160)
    plt.close(fig)
    paths.append(tracking_path)

    returns_path = output / "portfolio_vs_benchmark_returns.png"
    summary = results.groupby("model", sort=False)[
        ["portfolio_return", "benchmark_return", "active_return"]
    ].mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    summary[["portfolio_return", "benchmark_return"]].mul(100).plot.bar(ax=ax)
    ax.set_title("Mean annual out-of-sample returns")
    ax.set_ylabel("Return (%)")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(returns_path, dpi=160)
    plt.close(fig)
    paths.append(returns_path)

    quality_path = output / "model_quality_summary.png"
    quality = results.groupby("model", sort=False)[
        ["tracking_error", "correlation", "active_holdings"]
    ].mean()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    quality["tracking_error"].mul(100).plot.bar(ax=axes[0], color="steelblue", title="Mean TE (%)")
    quality["correlation"].plot.bar(ax=axes[1], color="darkorange", title="Mean correlation")
    quality["active_holdings"].plot.bar(ax=axes[2], color="seagreen", title="Mean holdings")
    for ax in axes:
        ax.tick_params(axis="x", rotation=0)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(quality_path, dpi=160)
    plt.close(fig)
    paths.append(quality_path)

    return paths
