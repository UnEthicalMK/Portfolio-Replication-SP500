"""Additional diagnostics for robustness, exposures, costs, and regimes."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import DATA_DIR, PLOTS_DIR, TARGET

plt.style.use("seaborn-v0_8-whitegrid")


def _markdown_table(frame: pd.DataFrame, percent_columns: set[str] | None = None) -> list[str]:
    percent_columns = percent_columns or set()
    headers = list(frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in headers:
            value = row[column]
            if pd.isna(value):
                values.append("—")
            elif column in percent_columns:
                values.append(f"{float(value):.2%}")
            elif isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_research_markdown() -> None:
    """Create a readable GitHub report for the additional research outputs."""
    report = [
        "# Research Robustness Diagnostics",
        "",
        "These diagnostics are descriptive research outputs, not investment "
        "recommendations. CSV files are retained beside this report for analysis.",
        "",
        "## Rolling market exposure",
        "",
        "The 63-day rolling correlation and beta plot shows whether portfolio "
        "similarity to the benchmark is stable across market conditions.",
        "",
        "![Rolling market exposure](../plots/13_rolling_market_exposure.png)",
        "",
        "Data: [`rolling_market_diagnostics.csv`](rolling_market_diagnostics.csv)",
        "",
        "## Proxy-factor exposures",
        "",
        "These regressions use candidate-universe proxy factors. Momentum and "
        "low-volatility factors are descriptive and are not point-in-time "
        "academic factor portfolios.",
        "",
    ]
    factors = pd.read_csv(DATA_DIR / "factor_exposures.csv")
    factor_view = factors.copy()
    for column in ("Alpha (daily)", "Market Beta", "Momentum Beta", "LowVol Beta", "R-squared"):
        factor_view[column] = factor_view[column].astype(float)
    report += _markdown_table(factor_view)
    report += ["", "Data: [`factor_exposures.csv`](factor_exposures.csv)", "", "## Regime analysis", ""]

    regimes = pd.read_csv(DATA_DIR / "regime_metrics.csv")
    regime_view = regimes.copy()
    report += _markdown_table(
        regime_view,
        {"Cumulative Return", "Tracking Error", "Correlation"},
    )
    report += ["", "Data: [`regime_metrics.csv`](regime_metrics.csv)", "", "## Gross versus net cost attribution", ""]

    costs = pd.read_csv(DATA_DIR / "oos_cost_attribution.csv")
    report += _markdown_table(
        costs,
        {"Gross Return", "Net Return"},
    )
    report += ["", "Data: [`oos_cost_attribution.csv`](oos_cost_attribution.csv)", "", "## Sensitivity", ""]

    sensitivity = pd.read_csv(DATA_DIR / "sensitivity_summary.csv")
    report += _markdown_table(
        sensitivity,
        {"Validation Tracking Error"},
    )
    report += [
        "",
        "Data: [`sensitivity_summary.csv`](sensitivity_summary.csv)",
        "",
        "Interpretation: K sensitivity is measured on the validation period; "
        "turnover sensitivity is measured over the 2023 test period. Neither "
        "table should be interpreted as a guarantee of future performance.",
        "",
    ]
    (DATA_DIR / "research_analysis.md").write_text("\n".join(report), encoding="utf-8")


def _strategy_returns() -> tuple[pd.DataFrame, pd.Series]:
    returns = pd.read_csv(
        DATA_DIR / "oos_portfolio_returns.csv", index_col=0, parse_dates=True
    )
    benchmark = returns.pop("S&P 500 (Target)")
    return returns, benchmark


def write_rolling_market_diagnostics(window: int = 63) -> None:
    portfolios, benchmark = _strategy_returns()
    rows = []
    for column in portfolios:
        residual = portfolios[column] - benchmark
        rolling_corr = portfolios[column].rolling(window).corr(benchmark)
        rolling_beta = (
            portfolios[column].rolling(window).cov(benchmark)
            / benchmark.rolling(window).var()
        )
        for date in portfolios.index:
            rows.append(
                {
                    "Date": date,
                    "Strategy": column,
                    "Rolling Correlation": rolling_corr.loc[date],
                    "Rolling Beta": rolling_beta.loc[date],
                    "Rolling Tracking Error": residual.rolling(window).std().loc[date]
                    * np.sqrt(252),
                }
            )
    diagnostics = pd.DataFrame(rows).dropna()
    diagnostics.to_csv(DATA_DIR / "rolling_market_diagnostics.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    for strategy, group in diagnostics.groupby("Strategy"):
        group = group.set_index("Date")
        axes[0].plot(group.index, group["Rolling Correlation"], label=strategy)
        axes[1].plot(group.index, group["Rolling Beta"], label=strategy)
    axes[0].set_title(f"{window}-Day Rolling Correlation to S&P 500 Total Return")
    axes[0].set_ylabel("Correlation")
    axes[1].set_title(f"{window}-Day Rolling Market Beta")
    axes[1].set_ylabel("Beta")
    axes[1].set_xlabel("Date")
    for ax in axes:
        ax.axhline(0, color="black", linewidth=0.8)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "13_rolling_market_exposure.png", dpi=150)
    plt.close(fig)


def _build_proxy_factors(log_returns: pd.DataFrame) -> pd.DataFrame:
    candidates = log_returns.drop(columns=TARGET)
    trailing_momentum = candidates.rolling(252, min_periods=126).sum().shift(1)
    trailing_volatility = candidates.rolling(63, min_periods=21).std().shift(1)
    factors = pd.DataFrame(index=log_returns.index)
    factors["Market"] = log_returns[TARGET]
    momentum_factor = []
    low_vol_factor = []
    for date in log_returns.index:
        momentum = trailing_momentum.loc[date].dropna()
        volatility = trailing_volatility.loc[date].dropna()
        if len(momentum) < 4:
            momentum_factor.append(np.nan)
        else:
            midpoint = len(momentum) // 2
            winners = momentum.nlargest(midpoint).index
            losers = momentum.nsmallest(midpoint).index
            momentum_factor.append(
                candidates.loc[date, winners].mean()
                - candidates.loc[date, losers].mean()
            )
        if len(volatility) < 4:
            low_vol_factor.append(np.nan)
        else:
            midpoint = len(volatility) // 2
            low = volatility.nsmallest(midpoint).index
            high = volatility.nlargest(midpoint).index
            low_vol_factor.append(
                candidates.loc[date, low].mean()
                - candidates.loc[date, high].mean()
            )
    factors["Momentum"] = momentum_factor
    factors["LowVol"] = low_vol_factor
    return factors


def write_factor_exposures() -> None:
    portfolios, _ = _strategy_returns()
    log_returns = pd.read_csv(
        DATA_DIR / "log_returns.csv", index_col=0, parse_dates=True
    )
    factors = _build_proxy_factors(log_returns).loc[portfolios.index]
    rows = []
    for strategy in portfolios:
        data = pd.concat([portfolios[strategy], factors], axis=1).dropna()
        y = data[strategy].to_numpy()
        X = np.column_stack([np.ones(len(data)), data[factors.columns].to_numpy()])
        coefficients, *_ = np.linalg.lstsq(X, y, rcond=None)
        fitted = X @ coefficients
        ss_res = np.sum((y - fitted) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        rows.append(
            {
                "Strategy": strategy,
                "Alpha (daily)": coefficients[0],
                "Market Beta": coefficients[1],
                "Momentum Beta": coefficients[2],
                "LowVol Beta": coefficients[3],
                "R-squared": 1 - ss_res / ss_tot if ss_tot else np.nan,
                "Observations": len(data),
            }
        )
    pd.DataFrame(rows).to_csv(DATA_DIR / "factor_exposures.csv", index=False)


def write_regime_metrics() -> None:
    portfolios, benchmark = _strategy_returns()
    volatility = benchmark.rolling(63).std()
    threshold = volatility.loc[benchmark.index].quantile(0.75)
    rows = []
    for strategy in portfolios:
        for regime, mask in (
            ("Low volatility", volatility < threshold),
            ("High volatility", volatility >= threshold),
            ("Down market", benchmark < 0),
            ("Up market", benchmark >= 0),
        ):
            selected = portfolios.loc[mask, strategy].dropna()
            bench = benchmark.loc[selected.index]
            residual = selected - bench
            rows.append(
                {
                    "Strategy": strategy,
                    "Regime": regime,
                    "Observations": len(selected),
                    "Cumulative Return": np.expm1(selected.sum()),
                    "Tracking Error": residual.std() * np.sqrt(252),
                    "Correlation": selected.corr(bench),
                }
            )
    pd.DataFrame(rows).to_csv(DATA_DIR / "regime_metrics.csv", index=False)


def write_sensitivity_summary() -> None:
    k_results = pd.read_csv(DATA_DIR / "k_selection_validation.csv")
    turnover = pd.read_csv(DATA_DIR / "oos_turnover.csv", index_col=0)
    rows = []
    for _, row in k_results.iterrows():
        for model in ("Lasso", "Equal_Weight", "SAE"):
            rows.append(
                {
                    "Sensitivity": "K",
                    "Parameter": int(row["K"]),
                    "Model": model,
                    "Validation Tracking Error": row[model],
                    "Cumulative Turnover": np.nan,
                }
            )
    for policy in turnover.columns:
        for model in turnover.index:
            rows.append(
                {
                    "Sensitivity": "Execution policy",
                    "Parameter": policy,
                    "Model": model,
                    "Validation Tracking Error": np.nan,
                    "Cumulative Turnover": turnover.loc[model, policy],
                }
            )
    pd.DataFrame(rows).to_csv(DATA_DIR / "sensitivity_summary.csv", index=False)


def run_research_analysis() -> None:
    print("  Computing rolling market exposure diagnostics...")
    write_rolling_market_diagnostics()
    print("  Computing proxy-factor exposures...")
    write_factor_exposures()
    print("  Computing market-regime metrics...")
    write_regime_metrics()
    print("  Writing K and execution sensitivity summary...")
    write_sensitivity_summary()
    print("  Writing GitHub-readable research report...")
    write_research_markdown()
