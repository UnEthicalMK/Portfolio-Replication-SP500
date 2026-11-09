# Final Out-of-Sample Metrics

- Benchmark: `^SP500TR`
- Benchmark cumulative return: **26.79%**
- Common portfolio size: **K=15**
- Returns are net of the configured transaction-cost assumptions.

## Buy Hold

| Model | K | Tracking error | Cumulative return | Correlation | Max drawdown | Max daily deviation | Active instruments | Residual Sharpe |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Equal_Weight | 15 | 9.28% | 28.59% | 0.8101 | -9.75% | 2.21% | 15 | 0.15 |
| Lasso | 15 | 8.57% | 33.79% | 0.7946 | -6.57% | 1.95% | 15 | 0.63 |
| SAE | 15 | 9.53% | 32.24% | 0.8025 | -8.87% | 2.04% | 15 | 0.45 |

## Walk Forward Monthly

| Model | K | Tracking error | Cumulative return | Correlation | Max drawdown | Max daily deviation | Active instruments | Residual Sharpe |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Equal_Weight | 15 | 9.04% | 27.59% | 0.8082 | -9.84% | 2.44% | 15 | 0.07 |
| Lasso | 15 | 8.41% | 35.55% | 0.7986 | -6.29% | 1.86% | 15 | 0.80 |
| SAE | 15 | 9.42% | 35.24% | 0.8064 | -9.27% | 2.15% | 15 | 0.69 |

## Walk Forward Quarterly

| Model | K | Tracking error | Cumulative return | Correlation | Max drawdown | Max daily deviation | Active instruments | Residual Sharpe |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Equal_Weight | 15 | 9.03% | 27.31% | 0.8094 | -9.95% | 2.43% | 15 | 0.05 |
| Lasso | 15 | 8.42% | 34.96% | 0.7991 | -6.30% | 1.91% | 15 | 0.75 |
| SAE | 15 | 9.72% | 32.37% | 0.8039 | -10.09% | 1.89% | 15 | 0.45 |
