# Legacy artifacts — archived at S0

These two files were produced by the pre-`03454bc` pipeline (Lasso-based
models, `^SP500TR` total-return benchmark) and are written by no module in the
current codebase (Investigation Report finding **D-11**). They are retained
for provenance only: they document that the project originally targeted the
S&P 500 **Total Return** index before the silent switch to the price index in
commit `03454bc` ("Update data pipeline to use price returns").

| File | Contents | Origin |
|---|---|---|
| `benchmark_returns_2023.csv` | daily log returns of `^SP500TR`, 2023 (249 rows) | original (total-return) data pipeline |
| `portfolio_returns_2023.csv` | daily log returns, columns `Equal_Weight, Lasso, SAE` | Lasso-era models pipeline |

Relocated from `data/` to `artifacts/legacy/` at the S0 checkpoint on branch
`s0-engineering-foundation` (contents preserved; moved as tracked Git renames).

Rules: do not delete; do not use in any result; superseded by the S1 snapshot
layer (`data/raw/`, Spec §4.5).
