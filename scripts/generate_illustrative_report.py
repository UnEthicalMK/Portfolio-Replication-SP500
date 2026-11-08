"""Generate clearly labelled synthetic report previews for the README."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.reporting import generate_report_plots


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "plots"


def main() -> None:
    rng = np.random.default_rng(42)
    rows = []
    for year in range(2019, 2024):
        benchmark = 0.10 + rng.normal(0, 0.04)
        for model, tracking_error, correlation, holdings in (
            ("equal_weight", 0.12, 0.86, 20),
            ("ridge", 0.08, 0.93, 10),
        ):
            portfolio = benchmark + rng.normal(0, tracking_error / np.sqrt(252))
            rows.append(
                {
                    "test_year": year,
                    "model": model,
                    "tracking_error": tracking_error + rng.normal(0, 0.008),
                    "correlation": correlation + rng.normal(0, 0.01),
                    "portfolio_return": portfolio,
                    "benchmark_return": benchmark,
                    "active_return": portfolio - benchmark,
                    "active_holdings": holdings,
                }
            )

    generated = generate_report_plots(pd.DataFrame(rows), OUTPUT)
    for path in generated:
        target = OUTPUT / f"illustrative_{path.name}"
        path.replace(target)


if __name__ == "__main__":
    main()
