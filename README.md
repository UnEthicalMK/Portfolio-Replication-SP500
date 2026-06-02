# Synthetic Index Replication Engine

> **Replicate the S&P 500 Total Return Index using 20 liquid Global ADRs — without holding a single US-listed stock.**

A full quantitative pipeline that constructs, backtests, and diagnostics three competing replication strategies: a **Simplex L2 Optimizer** (Lasso), a **Sparse Autoencoder** (SAE), and an **Equal-Weight Baseline**. Trained on 8 years of out-of-sample data, the engine achieves sub-9% annualised tracking error with a correlation above **0.80** to the S&P 500 Total Return benchmark.

---

## Performance Summary (2023 Out-of-Sample)

| Metric | Lasso (Simplex) | Autoencoder (SAE) | Equal Weight | S&P 500 |
|---|---|---|---|---|
| **Tracking Error (Ann.)** | **8.50%** | 9.42% | 8.77% | 0.00% |
| **Correlation** | 0.8029 | 0.8064 | 0.7963 | 1.0000 |
| **Max Drawdown** | **-6.93%** | -9.44% | -8.38% | -9.94% |
| **Stocks Used** | 20 | 20 | 20 | 500 |

> All three models were trained on 2014–2021 data, selected on a 2022 validation year, and evaluated on a fully held-out 2023 test set. No forward-looking information crosses any temporal wall.

---

## Architecture

```
synthetic-index-tracker/
│
├── main.py                          # CLI Orchestrator — run any combination of milestones
│
├── src/
│   ├── data_pipeline.py             # Milestone 1: yfinance ingestion, IPO filtering, holiday forward-fill
│   ├── features.py                  # Milestone 2: Temporal partitioning, StandardScaler, hierarchical correlation
│   ├── models_pipeline.py           # Milestone 3: Model training, apples-to-apples K-constraint handshake
│   ├── backtest.py                  # Milestone 4: Friction-adjusted buy-and-hold simulator (ADR tier fees)
│   ├── diagnostics.py               # Milestone 5: Institutional tear sheet, residual distribution, Max DD
│   │
│   └── models/
│       ├── equal_weight.py          # Model A: Top-K equal allocation ranked by training correlation
│       ├── lasso_model.py           # Model B: Scaled L2-penalised Simplex Optimizer (SLSQP, ftol=1e-9)
│       └── sparse_autoencoder.py    # Model C: L1-regularised Latent SAE with early stopping (PyTorch)
│
├── data/                            # Auto-generated at runtime
│   ├── raw_prices.csv               # Cleaned ADR + S&P 500 closing prices (2014–2023)
│   ├── log_returns.csv              # Continuous log returns for the full universe
│   ├── X_train_scaled.csv           # StandardScaler-transformed features (2014–2021, SAE input)
│   ├── X_val_scaled.csv             # Scaled validation features (2022)
│   ├── X_test_scaled.csv            # Scaled test features (2023)
│   ├── X_train_unscaled.csv         # Raw log returns — training split (Lasso + Equal-Weight input)
│   ├── X_val_unscaled.csv           # Raw log returns — validation split
│   ├── X_test_unscaled.csv          # Raw log returns — test split
│   ├── y_train.csv                  # S&P 500 TR log returns — training split
│   ├── y_val.csv                    # S&P 500 TR log returns — validation split
│   ├── y_test.csv                   # S&P 500 TR log returns — test split
│   ├── oos_portfolio_returns.csv    # Combined 2023 daily returns for all three models + benchmark
│   └── final_summary_metrics.csv   # Exported quantitative tear sheet (TE, Sharpe, drawdown, etc.)
│
├── models/                          # Auto-generated at runtime
│   ├── scaler.pkl                   # Fitted StandardScaler artifact (training-set statistics only)
│   └── frozen_weights_2022.csv      # Locked optimal portfolio weights (T-0 state, val-set selected)
│
└── plots/                           # Auto-generated at runtime
    ├── 01_price_history.png         # Normalised price levels (Base = 100, 2014–2023)
    ├── 02_missing_data.png          # Missing data map — IPO gaps and holiday structure
    ├── 03_correlation_matrix.png    # Hierarchical correlation matrix (training period only)
    ├── 04_lasso_sweep.png           # L2 alpha sweep: tracking error vs. sparsity frontier
    ├── 05_instrument_weights.png    # Frozen allocation vectors across all three models
    ├── 06_cumulative_returns.png    # 2023 growth of $1 — all models vs. benchmark
    ├── 07_rolling_tracking_error.png # Rolling 21-day annualised tracking error (2023)
    └── 08_residual_distribution.png # Daily residual distributions with fitted normals
```

---

## Proxy Universe — 31 Global ADRs

The strategy is intentionally constructed from **non-US-listed instruments only**, making it applicable to regulatory regimes that restrict direct US equity ownership (e.g., UCITS, certain sovereign wealth mandates).

| Region | Tickers |
|---|---|
| **China / HK** | BABA, JD, BIDU, TME, NTES, PDD |
| **India** | INFY, WIT, HDB, IBN, RDY, MMYT |
| **Latin America** | MELI, VALE, PBR, BBD |
| **Europe** | ASML, NVO, SAP, SHEL, BCS |
| **Japan** | TM, SONY, HMC |
| **Taiwan** | TSM, UMC |
| **South Korea** | SKM, KB |

---

## Pipeline Design Decisions

### Temporal Discipline
The pipeline enforces strict temporal walls with no data leakage across any split:

| Split | Period | Purpose |
|---|---|---|
| **Train** | 2014–2021 | Model fitting and scaler calibration |
| **Validation** | 2022 | Hyperparameter selection and weight freezing |
| **Test** | 2023 | Out-of-sample evaluation only |

### The Apples-to-Apples Handshake
The Lasso optimizer selects the optimal number of active instruments `K` on the validation set. Both the Equal-Weight and SAE models are then **constrained to the exact same `K`** at inference time, ensuring all comparisons are structurally fair.

### Transaction Cost Model
The backtest applies a tiered friction model based on MSCI market classification:

| ADR Tier | Fee |
|---|---|
| Emerging Market ADRs (China, India, LatAm, Taiwan, Korea) | **15 bps** |
| Developed Market ADRs (Europe, Japan) | **10 bps** |

This is applied as a one-time entry cost on the initial capital allocation, consistent with a buy-and-hold mandate.

### Data Quality
Late-IPO tickers (BABA, PDD, TME) are handled by forward-filling up to 5 consecutive business days across international holiday gaps. Any ticker with remaining NaNs after filling is dropped entirely. The target benchmark (`^SP500TR`) is protected by a hard circuit-breaker — if it is dropped, the pipeline halts.

---

## Quickstart

### Requirements

```bash
pip install yfinance pandas numpy scipy scikit-learn matplotlib seaborn joblib torch
```

### Run the Full Pipeline

```bash
python main.py --all
```

### Run Individual Milestones

```bash
python main.py --data          # Milestone 1: Download and clean ADR data
python main.py --features      # Milestone 2: Partition, scale, and correlate
python main.py --models        # Milestone 3: Train all three models
python main.py --backtest      # Milestone 4: 2023 out-of-sample simulation
python main.py --diagnostics   # Milestone 5: Tear sheet and residual analysis
```

> Running `main.py` with no flags defaults to `--all`.

---

## Visual Diagnostics

### Cumulative Returns — 2023 Out-of-Sample
All three proxy portfolios outperformed the S&P 500 Total Return benchmark in 2023, driven by strong performance in the European and Japanese ADR names. The Lasso portfolio exhibited the tightest co-movement with the benchmark, consistent with its lower tracking error.

### Rolling Tracking Error
The 21-day annualised tracking error for all models converged into the 6–8% band by mid-year, with the SAE showing moderately higher dispersion in the August regime shift. Lasso maintained the most stable profile throughout.

### Residual Distribution
All three models produce approximately zero-mean residual distributions centred near the dashed vertical at `0.000`. The SAE exhibits slightly fatter tails (wider Gaussian fit) relative to Lasso and Equal-Weight, consistent with its higher tracking error and max drawdown.

---

## Model Notes

**Model A — Equal Weight**: Ranks all proxy assets by training-period correlation to the benchmark and allocates `1/K` to the top-`K` names. Deliberately naïve — its purpose is to establish a diversification floor.

**Model B — Lasso (Simplex Optimizer)**: Solves a constrained quadratic program on the simplex (`Σwᵢ = 1`, `wᵢ ≥ 0`) with an L2 regularisation penalty. Returns are scaled by `×100` before optimisation to escape floating-point precision traps near machine epsilon. The solver uses `SLSQP` with `ftol=1e-9`.

**Model C — Sparse Autoencoder**: A `[D → 3 → D]` linear bottleneck network with a dedicated `tracking_head` that maps the 3-dimensional latent representation to a single scalar. Final weights are extracted via `softmax(tracking_head.weight @ encoder.weight)`. Sparsity is enforced via L1 penalties on encoder and tracking head weights. Trained with `AdamW`, dropout (`p=0.3`), Gaussian noise injection (`σ=0.02`), and patience-based early stopping.

---

## Caveats

- Returns data sourced from Yahoo Finance. Prices are adjusted closing prices and may differ from vendor-grade data sources.
- The backtest assumes a single-entry buy-and-hold with no rebalancing. Real-world index replication requires continuous drift management.
- All three portfolios outperformed the benchmark in 2023. This is partly attributable to favourable conditions for non-US ADRs (USD weakness, European energy recovery, Japan reflation). This alpha is not expected to be persistent.
- The pipeline does not implement short selling, leverage, or currency hedging.

---

## License

MIT License. See `LICENSE` for details.