"""
agents.py – Three Vertex AI Gemini agents for the ETF Sector Rotation Dashboard.

Agent 1 – Market Analyst:      reads macro snapshot, identifies regime + sector implications
Agent 2 – Portfolio Strategist: reviews allocation for concentration, rotation, blind spots
Agent 3 – Risk Monitor:         flags drawdown / vol / concentration with traffic-light rating

Each function returns a generator of text chunks (str).
Compatible with FastAPI StreamingResponse and st.write_stream().
"""
from __future__ import annotations

import os
from typing import Generator

import numpy as np
import pandas as pd
from google import genai
from google.genai import types

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY", "")
    if api_key:
        return genai.Client(api_key=api_key)
    # Fall back to Vertex AI (uses ADC — works automatically on Cloud Run)
    project = os.getenv("GOOGLE_CLOUD_PROJECT", "")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    if not project:
        raise ValueError(
            "Set GEMINI_API_KEY or GOOGLE_CLOUD_PROJECT in your .env file."
        )
    return genai.Client(vertexai=True, project=project, location=location)


def _stream(system: str, user: str) -> Generator[str, None, None]:
    """Stream a Gemini response as text chunks via google-genai + Vertex AI."""
    client = _client()
    config = types.GenerateContentConfig(
        system_instruction=system,
        max_output_tokens=1024,
        temperature=0.3,
    )
    for chunk in client.models.generate_content_stream(
        model=GEMINI_MODEL,
        contents=user,
        config=config,
    ):
        if chunk.text:
            yield chunk.text


def _fmt_kv(d: dict) -> str:
    lines = []
    for k, v in d.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            lines.append(f"  {k}: {v:.4f}")
        else:
            lines.append(f"  {k}: {v}")
    return "\n".join(lines)


def _fmt_rows(df: pd.DataFrame, cols: list[str]) -> str:
    rows = []
    for _, r in df.iterrows():
        parts = []
        for c in cols:
            if c not in r.index:
                continue
            v = r[c]
            if isinstance(v, float):
                parts.append(f"{c}={v:.4f}")
            else:
                parts.append(f"{c}={v}")
        rows.append("  " + ", ".join(parts))
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Agent 1 – Market Analyst
# ---------------------------------------------------------------------------

def market_analyst_stream(
    panel: pd.DataFrame,
    alloc: pd.DataFrame,
) -> Generator[str, None, None]:
    """Analyze the macro environment and identify the market regime."""
    latest_date = panel["Date"].max()
    latest_row = panel[panel["Date"] == latest_date].iloc[0]

    macro_keys = [
        "yield_spread_10y2y", "vix", "vix_pctile", "hy_spread",
        "fed_funds_rate", "cpi_yoy", "unemployment_rate",
        "indpro_yoy", "treasury_10y", "treasury_2y", "eur_usd",
    ]
    macro_data = {
        k: float(latest_row[k])
        for k in macro_keys
        if k in latest_row.index and pd.notna(latest_row[k])
    }

    score_lines = _fmt_rows(
        alloc.sort_values("predicted_score", ascending=False),
        ["ETF", "Sector", "predicted_score"],
    )

    system = (
        "You are a senior macro economist and market analyst specialising in U.S. sector ETF rotation. "
        "You interpret macro indicators to identify the current market regime and its implications for "
        "sector allocation. Be concise, data-driven, and direct. Use markdown headers exactly as shown."
    )

    user = f"""Analyze the macro environment for sector ETF rotation.

**As-of date:** {str(latest_date)[:10]}

**Macro Indicators:**
{_fmt_kv(macro_data)}

**ML Model Sector Scores (ranked high to low):**
{score_lines}

Reply with exactly these sections:

## Market Regime
(2-3 sentences — name the regime and its key characteristics)

## Key Macro Drivers
(Top 3 factors, each with a specific data reference)

## Sector Implications
(Which sectors benefit and which suffer in this regime, and why)

## Watch List
(1-2 macro risks to monitor over the next 30 days)"""

    return _stream(system, user)


# ---------------------------------------------------------------------------
# Agent 2 – Portfolio Strategist
# ---------------------------------------------------------------------------

def portfolio_strategist_stream(
    alloc: pd.DataFrame,
    metrics: pd.DataFrame,
    feat_imp: pd.DataFrame,
) -> Generator[str, None, None]:
    """Review the ML-generated allocation for quality and opportunities."""
    held = alloc[alloc["weight"] > 0].sort_values("pct_weight", ascending=False)
    not_held = alloc[alloc["weight"] == 0].sort_values("predicted_score", ascending=False)

    held_str = _fmt_rows(held, ["ETF", "Sector", "pct_weight", "predicted_score"])
    not_held_str = _fmt_rows(not_held, ["ETF", "Sector", "predicted_score"])

    metrics_str = ""
    if "model" in metrics.columns:
        metrics_str = _fmt_kv(metrics["model"].to_dict())

    feat_str = ""
    if not feat_imp.empty:
        top10 = feat_imp.head(10)
        feat_str = "\n".join(
            f"  {idx}: {row['mean_importance']:.4f}"
            for idx, row in top10.iterrows()
        )

    system = (
        "You are a quantitative portfolio manager specialising in systematic sector rotation strategies. "
        "You critically evaluate ML-generated allocations for quality, concentration risk, and missed "
        "opportunities. Be specific, reference the data, and give actionable recommendations. "
        "Use markdown headers exactly as shown."
    )

    user = f"""Review the current sector rotation portfolio allocation.

**Held Positions (by weight):**
{held_str}

**Not Held (by predicted score):**
{not_held_str}

**Backtest Metrics (Model):**
{metrics_str}

**Top 10 Predictive Features:**
{feat_str}

Reply with exactly these sections:

## Allocation Assessment
(Is the allocation sensible? 2-3 sentences)

## Concentration Risk
(Any dangerous concentration? Which positions concern you most?)

## Rotation Opportunities
(Near-the-cut ETFs worth watching? Score differentials?)

## Model Blind Spots
(What might the model miss given these top features?)

## Recommendation
(One specific, concrete action or monitoring point)"""

    return _stream(system, user)


# ---------------------------------------------------------------------------
# Agent 3 – Risk Monitor
# ---------------------------------------------------------------------------

def risk_monitor_stream(
    backtest: pd.DataFrame,
    alloc: pd.DataFrame,
) -> Generator[str, None, None]:
    """Assess portfolio risk and produce a traffic-light rating."""
    eq = backtest["model_equity"].dropna() if "model_equity" in backtest.columns else pd.Series(dtype=float)
    rets = backtest["model"].dropna() if "model" in backtest.columns else pd.Series(dtype=float)

    current_dd = float((eq.iloc[-1] / eq.cummax().iloc[-1] - 1) * 100) if len(eq) > 0 else 0.0
    max_dd = float((eq / eq.cummax() - 1).min() * 100) if len(eq) > 0 else 0.0
    ret_30d = float((eq.iloc[-1] / eq.iloc[-22] - 1) * 100) if len(eq) > 22 else 0.0
    vol_63d = float(rets.tail(63).std() * (252 ** 0.5) * 100) if len(rets) >= 10 else 0.0
    vol_21d = float(rets.tail(21).std() * (252 ** 0.5) * 100) if len(rets) >= 5 else 0.0

    w = alloc["weight"].values
    hhi = float((w ** 2).sum())
    n_pos = int((alloc["weight"] > 0).sum())

    held = alloc[alloc["weight"] > 0].sort_values("pct_weight", ascending=False)
    top_etf = held.iloc[0]["ETF"] if len(held) > 0 else "N/A"
    top_wt = float(held.iloc[0]["pct_weight"]) if len(held) > 0 else 0.0
    held_str = _fmt_rows(held, ["ETF", "Sector", "pct_weight"])

    system = (
        "You are a risk manager for a systematic quantitative fund. "
        "You monitor drawdown, concentration, and volatility regime. You flag risks clearly "
        "and recommend protective actions. Be direct — your mandate is capital preservation. "
        "Use markdown headers exactly as shown."
    )

    user = f"""Assess current portfolio risk and flag any concerns.

**Portfolio Risk Metrics:**
  Current drawdown:           {current_dd:.1f}%
  Max historical drawdown:    {max_dd:.1f}%
  30-day return:              {ret_30d:.1f}%
  Volatility 63d (ann.):      {vol_63d:.1f}%
  Volatility 21d (ann.):      {vol_21d:.1f}%
  Number of positions:        {n_pos}
  Herfindahl-Hirschman Index: {hhi:.3f}  (0=equal-weight, 1=single-ETF)
  Largest position:           {top_etf} at {top_wt:.1f}%

**Current Holdings (by weight):**
{held_str}

Reply with exactly these sections:

## Risk Status
(Start with exactly one of: 🟢 Green / 🟡 Amber / 🔴 Red — then one sentence justifying the rating)

## Drawdown Assessment
(Is current drawdown within acceptable bounds? At what level would you escalate?)

## Concentration Risk
(Is the HHI acceptable? Which position carries the highest single-name risk?)

## Volatility Regime
(Is current vol elevated relative to history? Any vol-spike concerns?)

## Recommended Action
(One concrete risk management action, or "No action required — continue monitoring" if all clear)"""

    return _stream(system, user)
