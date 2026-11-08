# US-Listed S&P 500 Replication Research

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-Research%20Workflow-8A2BE2)

> **Can a small portfolio of US-listed stocks that are *not* S&P 500 constituents replicate the S&P 500 out of sample?**

This repository is a research workflow for testing that question with a curated universe of common stocks, ADRs, and foreign ordinary shares traded on US exchanges. The portfolio candidates are explicitly restricted to securities that are **not constituents of the S&P 500 at the portfolio formation date**. The project is intentionally stronger than a prototype while remaining smaller than a live portfolio-management system.

## Research objective

Determine whether an eligible, long-only, fully invested portfolio of **US-listed, non-S&P 500 stocks** can track the **S&P 500 Total Return Index** out of sample while respecting:

- A maximum portfolio size
- Per-security weight caps
- Minimum trading history
- Prior-only liquidity constraints
- Simplified transaction costs

The benchmark is the S&P 500; its constituents are the target being replicated, not the securities eligible for portfolio construction. The experiment therefore asks whether exposure to the index can be approximated using a smaller portfolio drawn from outside the index.

The project measures statistical tracking quality and simulated portfolio behavior. It does **not** claim that the resulting portfolio is a live or investment-ready substitute for the S&P 500.

## Workflow

```text
Versioned universe manifest
            ↓
Point-in-time eligibility
            ↓
Prior-only liquidity screen
            ↓
Explicit adjusted-price returns
            ↓
Walk-forward estimation / validation / test folds
            ↓
Equal-weight and constrained ridge trackers
            ↓
Out-of-sample metrics and saved results
```

## Universe rules

The initial universe is maintained in [`configs/universe.csv`](configs/universe.csv).

| Rule | Current policy |
| --- | --- |
| Exchanges | NYSE, Nasdaq, NYSE American |
| Instruments | Common stocks, ADRs, foreign ordinary shares |
| S&P 500 membership | Exclude securities that are S&P 500 constituents at formation |
| Excluded | ETFs, ETNs, funds, preferreds, warrants, rights, structured products |
| Benchmark | `^SP500TR` |
| Minimum history | 252 observations |
| Liquidity lookback | 252 prior observations |
| Liquidity measure | Median dollar volume |
| Minimum median dollar volume | `$5 million` |
| Valid volume requirement | 80% of lookback |

The manifest is a documented research approximation. It does not provide institutional-grade historical security-master or historical S&P 500 membership data.

## Models

### Equal-weight baseline

Ranks eligible assets by training-period correlation with the benchmark, selects the configured number of holdings, and allocates equal weights.

### Ridge tracker

Selects a candidate subset of non-S&P 500 securities and solves a long-only, fully invested, weight-capped optimization:

$$
\underset{w}{\operatorname{minimize}}\quad
\frac{1}{T}\sum_{t=1}^{T}\left(y_t - \sum_{i=1}^{N}w_i x_{t,i}\right)^2
\lambda\sum_{i=1}^{N}w_i^2
$$

subject to:

$$
\sum_{i=1}^{N} w_i = 1,\qquad
0 \leq w_i \leq w_{\max},\qquad
\lvert\operatorname{supp}(w)\rvert \leq K
$$

where:

- \(y_t\) is the S&P 500 Total Return Index return at time \(t\);
- \(x_{t,i}\) is the return of eligible non-S&P 500 security \(i\);
- \(w_i\) is the portfolio weight assigned to security \(i\);
- \(\lambda\) controls weight concentration;
- \(K\) is the maximum number of holdings.

The number of holdings and regularization strength are selected using validation data only.

### Tracking-error metric

For portfolio return \(r_{p,t} = \sum_i w_i x_{t,i}\) and benchmark return \(r_{b,t}\), annualized tracking error is:

$$
\operatorname{TE}_{\mathrm{annualized}}
= \sqrt{252}\;
\operatorname{Std}\left(r_{p,t} - r_{b,t}\right)
$$

## Evaluation

The workflow uses chronological walk-forward folds:

| Fold component | Default |
| --- | --- |
| Estimation window | 1,260 observations |
| Validation window | 252 observations |
| Test window | 252 observations |
| Test years | 2014–2023 |

Reported metrics include:

- Annualized tracking error
- Benchmark correlation
- Portfolio, benchmark, and active returns
- Active holdings
- Selected portfolio size and regularization

Initial transaction costs are deducted from the first test-day return. Rebalancing, market impact, tax treatment, and broker-specific execution are intentionally outside the current scope.

## Running the workflow

Install dependencies:

```bash
python -m pip install -e ".[dev]"
```

Download and prepare market data:

```bash
python main.py --download
```

Run walk-forward evaluation using saved returns:

```bash
python main.py --evaluate
```

Run tests:

```bash
pytest
```

The default output directory is `data/results/`. Downloaded data and generated results are not treated as source-of-truth code.

## Project structure

```text
configs/
├── main.toml          # Validated research configuration
└── universe.csv       # Versioned security manifest

src/
├── config.py          # Configuration loading and validation
├── data_pipeline.py   # Downloading, adjustment, and return preparation
├── universe.py        # Point-in-time eligibility and liquidity screening
├── folds.py           # Walk-forward fold construction
├── models.py          # Equal-weight and ridge trackers
├── evaluate.py        # Portfolio returns, costs, and metrics
├── workflow.py        # End-to-end fold evaluation
└── seeds.py           # Reproducibility helper

tests/
└── test_workflow.py   # Core workflow invariants
```

## Assumptions and limitations

- Yahoo Finance is used as an accessible research data source.
- Historical listings, delistings, suspensions, and S&P 500 membership are approximated by the manifest.
- Missing observations are preserved rather than blanket-forward-filled.
- The current workflow requires complete observations for assets used in each evaluated fold.
- Costs are simplified and do not model spreads, market impact, taxes, or broker fees.
- Results are historical research evidence, not investment advice or a performance guarantee.

## Future scope

Only after the core workflow is stable:

1. Add richer historical security metadata.
2. Add explicit rebalancing and turnover accounting.
3. Add sensitivity analysis for liquidity, costs, and portfolio size.
4. Evaluate alternative models only if they improve out-of-sample results.
