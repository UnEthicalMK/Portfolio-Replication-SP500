"""Command-line entry point for the research workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.data_pipeline import prepare_universe_data, save_data
from src.seeds import set_seed
from src.workflow import run_walk_forward, save_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the US-listed S&P 500 replication workflow")
    parser.add_argument("--config", default=None, help="Path to a TOML configuration file")
    parser.add_argument("--download", action="store_true", help="Download and save market data")
    parser.add_argument("--evaluate", action="store_true", help="Run walk-forward evaluation")
    args = parser.parse_args()
    if not args.download and not args.evaluate:
        parser.print_help()
        return
    cfg = load_config(args.config)
    set_seed(42)
    output = Path(cfg.output_dir)

    if args.download:
        prices, returns, manifest = prepare_universe_data(cfg)
        save_data(prices, returns, manifest, output)
    if args.evaluate:
        returns_path = output / "returns.csv"
        if not returns_path.exists():
            raise FileNotFoundError("Run with --download before --evaluate")
        returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
        result_path = save_results(run_walk_forward(returns, cfg), output)
        print(f"Saved walk-forward metrics to {result_path}")


if __name__ == "__main__":
    main()
