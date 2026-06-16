import argparse
import sys

from src.data_pipeline import run_data_pipeline
from src.features import run_feature_engineering
from src.models_pipeline import run_models_pipeline
from src.backtest import run_backtest
from src.diagnostics import run_diagnostics

print("=" * 60)
print("SYNTHETIC INDEX REPLICATION ENGINE")
print("Benchmark: S&P 500 Price Index (^GSPC)")
print("Universe : Global ADR Proxy Basket")
print("=" * 60)

parser = argparse.ArgumentParser(
    description="Execute the Synthetic Index Replication Pipeline."
)

# Execution Flags
parser.add_argument('--all', action='store_true',
                    help='Run the complete end-to-end pipeline')
parser.add_argument('--data', action='store_true',
                    help='Milestone 1: Download and preprocess market data')
parser.add_argument('--features', action='store_true',
                    help='Milestone 2: Feature engineering and scaling')
parser.add_argument('--models', action='store_true',
                    help='Milestone 3: Train portfolio replication models')
parser.add_argument('--backtest', action='store_true',
                    help='Milestone 4: Execute out-of-sample backtest')
parser.add_argument('--diagnostics', action='store_true',
                    help='Milestone 5: Generate performance diagnostics')

args = parser.parse_args()

# Default Behaviour
if len(sys.argv) == 1:
    print("\n[INFO] No command-line arguments detected.")
    print("[INFO] Executing complete pipeline (--all).\n")
    args.all = True

# Pipeline Execution
if args.all or args.data:
    run_data_pipeline()

if args.all or args.features:
    run_feature_engineering()

if args.all or args.models:
    run_models_pipeline()

if args.all or args.backtest:
    run_backtest()

if args.all or args.diagnostics:
    run_diagnostics()

print("\n[SUCCESS] Pipeline execution completed.")