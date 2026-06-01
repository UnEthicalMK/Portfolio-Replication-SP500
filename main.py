import argparse
import sys

# Strictly require all core pipeline execution functions
from src.data_pipeline import run_data_pipeline
from src.features import run_feature_engineering
from src.models_pipeline import run_models_pipeline
from src.backtest import run_backtest
from src.diagnostics import run_diagnostics

print("=" * 60)
print("  SYNTHETIC INDEX REPLICATION ENGINE")
print("  Target: S&P 500 Total Return")
print("  Proxy Universe: 31 Global ADRs (Strictly Non-US)")
print("=" * 60)

parser = argparse.ArgumentParser(description="Execute the Synthetic S&P 500 Replication Pipeline.")

# Define execution flags
parser.add_argument('--all', action='store_true', help='Run the complete end-to-end pipeline')
parser.add_argument('--data', action='store_true', help='Milestone 1: Download and align global ADR data')
parser.add_argument('--features', action='store_true', help='Milestone 2: Generate correlation matrices')
parser.add_argument('--models', action='store_true', help='Milestone 3: Train Elastic Net, SAE, and extract weights')
parser.add_argument('--backtest', action='store_true', help='Milestone 4: Execute 2023 out-of-sample simulation')
parser.add_argument('--diagnostics', action='store_true', help='Milestone 5: Generate performance tear sheets')

args = parser.parse_args()

# If no flags are passed (e.g., clicking 'Run' in an IDE), default to running everything
if len(sys.argv) == 1:
    print("No command line flags detected. Defaulting to full pipeline run (--all)...")
    args.all = True

# Execute the requested milestones in strict chronological order
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

print("\n[✔] Institutional Pipeline Execution Completed Successfully.")