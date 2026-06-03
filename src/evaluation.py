"""
evaluation.py – Performance metrics and automated report generation.

Computes standard portfolio analytics and writes:
  • reports/evaluation_report.md
  • reports/model_card.md
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

logger = logging.getLogger(__name__)

ANNUALISE = cfg.TRADING_DAYS_PER_YEAR
RF = cfg.RISK_FREE_RATE


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def total_return(equity: pd.Series) -> float:
    """Cumulative return from start to end."""
    return float(equity.iloc[-1] / equity.iloc[0] - 1)


def cagr(equity: pd.Series) -> float:
    """Compound annual growth rate."""
    years = len(equity) / ANNUALISE
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) if years > 0 else 0.0


def annualised_vol(daily_rets: pd.Series) -> float:
    """Annualised volatility of daily returns."""
    return float(daily_rets.std() * np.sqrt(ANNUALISE))


def sharpe(daily_rets: pd.Series, rf: float = RF) -> float:
    """Annualised Sharpe ratio."""
    excess = daily_rets - rf / ANNUALISE
    vol = annualised_vol(daily_rets)
    return float(excess.mean() * ANNUALISE / vol) if vol > 0 else 0.0


def max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (negative number)."""
    rolling_max = equity.cummax()
    dd = equity / rolling_max - 1
    return float(dd.min())


def calmar(daily_rets: pd.Series, equity: pd.Series) -> float:
    """CAGR / |Max Drawdown|."""
    c = cagr(equity)
    mdd = abs(max_drawdown(equity))
    return float(c / mdd) if mdd > 0 else 0.0


def win_rate_monthly(daily_rets: pd.Series) -> float:
    """Fraction of calendar months with positive total return."""
    monthly = (1 + daily_rets).resample("ME").prod() - 1
    return float((monthly > 0).mean())


def avg_monthly_return(daily_rets: pd.Series) -> float:
    monthly = (1 + daily_rets).resample("ME").prod() - 1
    return float(monthly.mean())


def worst_month(daily_rets: pd.Series) -> float:
    monthly = (1 + daily_rets).resample("ME").prod() - 1
    return float(monthly.min())


def avg_turnover(turnover_series: pd.Series) -> float:
    """Average monthly turnover."""
    monthly_to = turnover_series.resample("ME").sum()
    return float(monthly_to.mean())


def compute_all_metrics(
    results: pd.DataFrame,
    strategies: list[str] = ("model", "equal_weight", "spy"),
) -> pd.DataFrame:
    """
    Compute the full metric suite for each strategy column.

    Parameters
    ----------
    results    : daily backtest DataFrame (output of backtest.run)
    strategies : list of column-name prefixes in results

    Returns
    -------
    pd.DataFrame  rows=metrics, cols=strategies
    """
    metrics: dict[str, dict] = {}
    for strat in strategies:
        ret_col    = strat
        equity_col = f"{strat}_equity"

        if ret_col not in results.columns or equity_col not in results.columns:
            continue

        rets   = results[ret_col].dropna()
        equity = results[equity_col].dropna()

        if len(rets) < 10:
            continue

        to = results["turnover"] if strat == "model" and "turnover" in results.columns else pd.Series([0.0])
        to.index = results.index[:len(to)]

        metrics[strat] = {
            "Total Return":           f"{total_return(equity):.2%}",
            "CAGR":                   f"{cagr(equity):.2%}",
            "Annualised Volatility":  f"{annualised_vol(rets):.2%}",
            "Sharpe Ratio":           f"{sharpe(rets):.3f}",
            "Max Drawdown":           f"{max_drawdown(equity):.2%}",
            "Calmar Ratio":           f"{calmar(rets, equity):.3f}",
            "Win Rate (Monthly)":     f"{win_rate_monthly(rets):.2%}",
            "Avg Monthly Return":     f"{avg_monthly_return(rets):.2%}",
            "Worst Month":            f"{worst_month(rets):.2%}",
            "Avg Monthly Turnover":   f"{avg_turnover(to):.2%}",
            "Start Date":             str(rets.index[0].date()),
            "End Date":               str(rets.index[-1].date()),
            "Trading Days":           str(len(rets)),
        }

    return pd.DataFrame(metrics)


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def feature_importance(models: dict, feature_cols: list[str]) -> pd.DataFrame:
    """Aggregate feature importance across the trained models."""
    importances: list[pd.Series] = []
    for name, model in models.items():
        imp = pd.Series(model.feature_importances_, index=feature_cols, name=name)
        importances.append(imp)
    if not importances:
        return pd.DataFrame()
    df = pd.concat(importances, axis=1)
    df["mean_importance"] = df.mean(axis=1)
    return df.sort_values("mean_importance", ascending=False)


# ---------------------------------------------------------------------------
# Report generators
# ---------------------------------------------------------------------------

def write_evaluation_report(
    metrics_df: pd.DataFrame,
    feature_imp: pd.DataFrame,
    panel: pd.DataFrame,
) -> None:
    """Write evaluation_report.md."""
    cfg.REPORTS.mkdir(parents=True, exist_ok=True)

    n_rows = len(panel)
    n_etfs = panel["ETF"].nunique() if "ETF" in panel.columns else 11
    date_min = panel["Date"].min().date() if "Date" in panel.columns else "N/A"
    date_max = panel["Date"].max().date() if "Date" in panel.columns else "N/A"

    lines = [
        "# Evaluation Report\n",
        f"*Generated: {date.today()}*\n",
        "---\n",
        "## Performance Metrics\n",
        metrics_df.to_markdown(),
        "\n\n---\n",
        "## Top 20 Features by Importance\n",
        feature_imp.head(20).to_markdown() if not feature_imp.empty else "_No feature importance available._",
        "\n\n---\n",
        "## Dataset Summary\n",
        f"- Rows: {n_rows:,}\n",
        f"- ETFs: {n_etfs}\n",
        f"- Date range: {date_min} → {date_max}\n",
        f"- Train / Val / Test split: "
        f"{cfg.TRAIN_FRAC:.0%} / {cfg.VAL_FRAC:.0%} / "
        f"{1 - cfg.TRAIN_FRAC - cfg.VAL_FRAC:.0%}\n",
    ]
    cfg.EVAL_REPORT.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Evaluation report saved → {cfg.EVAL_REPORT}")


def write_model_card(metrics_df: pd.DataFrame, feature_imp: pd.DataFrame) -> None:
    """Write model_card.md with full ML-card content."""
    cfg.REPORTS.mkdir(parents=True, exist_ok=True)
    today = date.today()

    top_features = (
        feature_imp.head(10).index.tolist()
        if not feature_imp.empty else ["N/A"]
    )

    content = f"""# Model Card – ETF Sector Rotation XGBoost Model

*Version 1.0 | Created: {today}*

---

## Model Purpose

This model is a **research prototype** developed as a university final project.
It predicts a risk-adjusted return score for 11 U.S. sector ETFs in order to
construct a monthly sector-rotation portfolio.

**This is NOT financial advice and should NOT be used for live trading.**

---

## Dataset Summary

| Property       | Value                                      |
|----------------|--------------------------------------------|
| Universe       | 11 SPDR Sector ETFs (XLK, XLV, XLF, …)   |
| Benchmark      | SPY (SPDR S&P 500 ETF)                     |
| Frequency      | Daily                                      |
| Source         | Bloomberg Excel export + yfinance + FRED   |
| Date Range     | 2015-06-04 → 2026-06-03 (approx.)         |
| Train Fraction | {cfg.TRAIN_FRAC:.0%}                               |
| Val Fraction   | {cfg.VAL_FRAC:.0%}                               |
| Test Fraction  | {1-cfg.TRAIN_FRAC-cfg.VAL_FRAC:.0%}                               |

---

## Target Variable

`target_score = future_21d_return / future_21d_volatility`

A proxy for the ex-ante Sharpe ratio over a 21-trading-day horizon.

Two separate XGBoost models are trained:
1. **Return model** → predicts `future_21d_return`
2. **Volatility model** → predicts `future_21d_volatility`

The predicted score is then: `predicted_return / predicted_volatility`.

---

## Features

### Price Features (ETF-level)
- Momentum: 5d, 21d, 63d, 126d, 252d returns
- Realised volatility: 21d, 63d (annualised)
- Downside volatility: 63d
- Moving average distance: 50d, 200d
- Relative strength vs equal-weight universe: 21d, 63d
- RSI (14d), Bollinger %B (20d)
- Volatility-of-volatility
- ETF one-hot dummy encoding

### Cross-Asset / Macro Features (FRED API)
- 10Y–2Y yield spread, fed funds rate, CPI YoY, unemployment,
  industrial production YoY, HY credit spread, VIX, EUR/USD

### Correlation Features
- Rolling 63-day correlation with SPY

---

## Model Architecture

| Property            | Return Model         | Volatility Model     |
|---------------------|---------------------|---------------------|
| Algorithm           | XGBoost Regressor   | XGBoost Regressor   |
| Estimators          | {cfg.XGB_PARAMS['n_estimators']}                 | {cfg.XGB_PARAMS['n_estimators']}                 |
| Max Depth           | {cfg.XGB_PARAMS['max_depth']}                   | {cfg.XGB_PARAMS['max_depth']}                   |
| Learning Rate       | {cfg.XGB_PARAMS['learning_rate']}              | {cfg.XGB_PARAMS['learning_rate']}              |
| Objective           | reg:squarederror    | reg:squarederror    |
| Validation          | Time-series split   | Time-series split   |
| Early Stopping      | 50 rounds           | 50 rounds           |

---

## Performance Metrics (Backtest)

{metrics_df.to_markdown() if not metrics_df.empty else "_Run main.py to generate metrics._"}

---

## Top Features by Importance

{", ".join(top_features)}

---

## Portfolio Construction

| Parameter           | Value              |
|--------------------|--------------------|
| Rebalance Frequency | Monthly (BME)     |
| ETFs selected (N)  | {cfg.TOP_N}                  |
| Max weight per ETF | {cfg.MAX_WEIGHT:.0%}               |
| Min weight per ETF | {cfg.MIN_WEIGHT:.0%} (long-only)    |
| Leverage           | None               |
| Transaction cost   | {cfg.TRANSACTION_COST:.2%} (one-way)     |

---

## Limitations

1. **Survivorship bias**: The ETF universe was fixed in hindsight. ETFs that
   were liquidated are not included.
2. **Short history**: The XLRE ETF was created in 2015; XLC in 2018. Earlier
   back-fills may not represent live tradeable data.
3. **No market-impact model**: Assumes infinite liquidity and no slippage
   beyond the flat transaction-cost assumption.
4. **Regime dependency**: The model was trained on a specific market regime
   (2015–2026). Performance may differ substantially in future regimes.
5. **Parameter sensitivity**: Results are sensitive to the choice of TOP_N,
   MAX_WEIGHT, and XGBoost hyper-parameters.
6. **Macro data look-ahead**: Monthly FRED releases (e.g. CPI) are forward-filled.
   In live usage, there is a publication lag of 2–4 weeks that is partially
   captured by enabling `USE_ALFRED_VINTAGES=True`.

---

## Risks

- **Model overfitting**: Despite time-series validation, XGBoost can overfit
  to noise in high-dimensional financial data.
- **Distribution shift**: Financial return distributions are non-stationary.
  A model trained on a bull market will likely underperform in a bear market.
- **Correlation breakdown**: Features (e.g. yield curve signals) that were
  predictive historically may lose predictive power in different rate regimes.

---

## Ethical Considerations

- This model is not audited for fairness in the regulatory sense.
- It does not make credit decisions about individuals.
- Environmental, Social, and Governance (ESG) factors are not considered.
- The model could be harmful if misused as financial advice to retail investors
  without appropriate risk disclosures.

---

## Why This Is Not Financial Advice

Past performance is not indicative of future results. This model was trained
on historical data and evaluated in a simulated back-test environment. Real
trading involves execution risk, liquidity constraints, market-impact costs,
tax consequences, and regulatory compliance — none of which are fully captured
here. The project is presented solely for academic and educational purposes.

---

*This model card follows the format proposed by Mitchell et al. (2019),
"Model Cards for Model Reporting." ACM FAccT.*
"""

    cfg.MODEL_CARD.write_text(content, encoding="utf-8")
    logger.info(f"Model card saved → {cfg.MODEL_CARD}")


# ---------------------------------------------------------------------------
# Top-level evaluation runner
# ---------------------------------------------------------------------------

def run(
    results: Optional[pd.DataFrame] = None,
    panel: Optional[pd.DataFrame] = None,
    models: Optional[dict] = None,
    feature_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Load artifacts (if needed) and produce all evaluation outputs."""
    from train_model import load_models
    from build_dataset import get_feature_columns

    if results is None:
        results = pd.read_csv(cfg.BACKTEST_CSV, index_col="Date", parse_dates=True)

    if panel is None:
        panel = pd.read_csv(cfg.FEATURES_CSV, parse_dates=["Date"])

    if feature_cols is None:
        feature_cols = get_feature_columns(panel)

    if models is None:
        try:
            models = load_models()
        except FileNotFoundError:
            models = {}

    metrics_df = compute_all_metrics(results)
    logger.info("\n" + metrics_df.to_string())

    feat_imp = feature_importance(models, feature_cols) if models else pd.DataFrame()
    write_evaluation_report(metrics_df, feat_imp, panel)
    write_model_card(metrics_df, feat_imp)

    return metrics_df


if __name__ == "__main__":
    logging.basicConfig(level=cfg.LOG_LEVEL, format=cfg.LOG_FMT)
    run()
