# S&P 500 Replication Prototype

This project evaluates whether a long-only portfolio of global ADRs can
statistically track the S&P 500 Total Return Index out of sample.

## Run locally

Install the dependencies and run the pipeline stages in order:

```powershell
python -m pip install -r requirements.txt
python main.py --all
```

The pipeline downloads prices, creates log returns, trains the portfolio
models, runs the 2023 backtest, and writes diagnostics to `data/` and `plots/`.

## Return convention

Price changes are stored as continuous log returns:

```text
log(price_t) - log(price_{t-1})
```

Cumulative performance is therefore calculated as
`exp(sum(log_returns)) - 1`.

## Interpretation limits

This is a research prototype, not an investment product. The current workflow
does not yet provide point-in-time constituent data, walk-forward validation,
turnover modelling, or a full liquidity and execution model. Results should be
treated as statistical tracking evidence rather than proof that the portfolio
is a live substitute for the S&P 500.
