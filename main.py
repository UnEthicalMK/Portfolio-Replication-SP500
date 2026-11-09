import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import sys
from time import perf_counter

# Strictly require all core pipeline execution functions
from src.data_pipeline import run_data_pipeline
from src.features import run_feature_engineering
from src.models_pipeline import run_models_pipeline
from src.backtest import run_backtest
from src.diagnostics import run_diagnostics
from src.config import LOG_DIR

parser = argparse.ArgumentParser(description="Run the S&P 500 replication research pipeline.")

# Define execution flags
parser.add_argument('--all', action='store_true', help='Run the complete end-to-end pipeline')
parser.add_argument('--data', action='store_true', help='Milestone 1: Download and align global ADR data')
parser.add_argument('--features', action='store_true', help='Milestone 2: Generate correlation matrices')
parser.add_argument('--models', action='store_true', help='Milestone 3: Train simplex optimizer, SAE, and extract weights')
parser.add_argument('--backtest', action='store_true', help='Milestone 4: Execute 2023 out-of-sample simulation')
parser.add_argument('--diagnostics', action='store_true', help='Milestone 5: Generate performance tear sheets')

args = parser.parse_args()

if len(sys.argv) == 1:
    args.all = True

stages = [
    (args.all or args.data, "Data", run_data_pipeline),
    (args.all or args.features, "Features", run_feature_engineering),
    (args.all or args.models, "Models", run_models_pipeline),
    (args.all or args.backtest, "Backtest", run_backtest),
    (args.all or args.diagnostics, "Diagnostics", run_diagnostics),
]

print("\nS&P 500 replication research pipeline")
print("Benchmark: ^SP500TR | Universe: US-listed non-S&P 500 candidates")
print("-" * 72)
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_path = LOG_DIR / f"pipeline_{datetime.now():%Y%m%d_%H%M%S}.log"
print(f"Detailed stage output: {log_path}")
for enabled, name, stage in stages:
    if not enabled:
        continue
    started = perf_counter()
    print(f"  [run]  {name}")
    try:
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"\n{'=' * 72}\n{name} stage\n{'=' * 72}\n")
            with redirect_stdout(log_file), redirect_stderr(log_file):
                stage()
    except Exception:
        print(f"  [fail] {name} (details saved to log)")
        raise
    print(f"  [done] {name} ({perf_counter() - started:.1f}s)")
print("-" * 72)
print("[OK] Pipeline completed")