"""
app.py – Streamlit dashboard for the ETF Sector Rotation Model.

Launch with:
    streamlit run src/app.py

The dashboard loads pre-computed artifacts from data/ and reports/.
Run main.py first to generate them.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import config as cfg

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ETF Sector Rotation Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Data loaders (cached)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def load_backtest() -> pd.DataFrame | None:
    if not cfg.BACKTEST_CSV.exists():
        return None
    df = pd.read_csv(cfg.BACKTEST_CSV, index_col="Date", parse_dates=True)
    return df


@st.cache_data(ttl=3600)
def load_allocation() -> pd.DataFrame | None:
    if not cfg.ALLOCATION_CSV.exists():
        return None
    return pd.read_csv(cfg.ALLOCATION_CSV)


@st.cache_data(ttl=3600)
def load_panel() -> pd.DataFrame | None:
    if not cfg.FEATURES_CSV.exists():
        return None
    return pd.read_csv(cfg.FEATURES_CSV, parse_dates=["Date"])


@st.cache_resource
def load_models_cached():
    try:
        from train_model import load_models
        return load_models()
    except Exception:
        return {}


@st.cache_data(ttl=3600)
def load_feature_importance() -> pd.DataFrame:
    models = load_models_cached()
    panel = load_panel()
    if not models or panel is None:
        return pd.DataFrame()
    from build_dataset import get_feature_columns
    from evaluation import feature_importance
    fc = get_feature_columns(panel)
    return feature_importance(models, fc)


@st.cache_data(ttl=3600)
def load_etf_prices() -> pd.DataFrame | None:
    p = cfg.DATA_RAW / "etf_prices.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def equity_chart(backtest: pd.DataFrame) -> go.Figure:
    cols_map = {
        "model_equity":        "Model (Sector Rotation)",
        "equal_weight_equity": "Equal-Weight Sectors",
        "spy_equity":          "SPY Benchmark",
    }
    colours = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    fig = go.Figure()
    for i, (col, label) in enumerate(cols_map.items()):
        if col in backtest.columns:
            series = backtest[col].dropna()
            fig.add_trace(go.Scatter(
                x=series.index, y=series,
                mode="lines", name=label,
                line=dict(color=colours[i], width=2),
            ))
    fig.update_layout(
        title="Portfolio Equity Curve",
        xaxis_title="Date", yaxis_title="Portfolio Value",
        hovermode="x unified", legend=dict(orientation="h", y=1.05),
        height=450,
    )
    return fig


def drawdown_chart(backtest: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    cols_map = {
        "model_equity":        ("Model", "#1f77b4"),
        "equal_weight_equity": ("Equal-Weight", "#ff7f0e"),
        "spy_equity":          ("SPY", "#2ca02c"),
    }
    for col, (label, colour) in cols_map.items():
        if col in backtest.columns:
            eq = backtest[col].dropna()
            dd = eq / eq.cummax() - 1
            fig.add_trace(go.Scatter(
                x=dd.index, y=dd * 100,
                mode="lines", name=label,
                line=dict(color=colour, width=1.5),
                fill="tozeroy", fillcolor=colour.replace(")", ", 0.08)").replace("rgb", "rgba"),
            ))
    fig.update_layout(
        title="Drawdown (%)",
        xaxis_title="Date", yaxis_title="Drawdown (%)",
        hovermode="x unified", height=350,
    )
    return fig


def allocation_chart(alloc: pd.DataFrame) -> go.Figure:
    held = alloc[alloc["weight"] > 0].sort_values("weight", ascending=True)
    fig = go.Figure(go.Bar(
        y=held["ETF"] + " – " + held["Sector"],
        x=held["pct_weight"],
        orientation="h",
        marker_color="#1f77b4",
        text=held["pct_weight"].map("{:.1f}%".format),
        textposition="outside",
    ))
    fig.update_layout(
        title="Current Sector Allocation (%)",
        xaxis_title="Weight (%)", height=max(300, 60 * len(held)),
        margin=dict(l=200),
    )
    return fig


def sector_ranking_chart(alloc: pd.DataFrame) -> go.Figure:
    df = alloc.sort_values("predicted_score", ascending=True)
    colours = ["#2ca02c" if w > 0 else "#d62728" for w in df["weight"]]
    fig = go.Figure(go.Bar(
        y=df["ETF"],
        x=df["predicted_score"],
        orientation="h",
        marker_color=colours,
        text=df["predicted_score"].map("{:+.4f}".format),
        textposition="outside",
    ))
    fig.update_layout(
        title="Predicted Score Ranking (green = held)",
        xaxis_title="Predicted Score", height=420,
        margin=dict(l=60),
    )
    return fig


def feature_importance_chart(feat_imp: pd.DataFrame, top_n: int = 20) -> go.Figure:
    df = feat_imp.head(top_n).sort_values("mean_importance", ascending=True)
    fig = go.Figure(go.Bar(
        y=df.index,
        x=df["mean_importance"],
        orientation="h",
        marker_color="#9467bd",
    ))
    fig.update_layout(
        title=f"Top {top_n} Features by Mean Importance",
        xaxis_title="Importance", height=max(300, 28 * top_n),
        margin=dict(l=200),
    )
    return fig


def etf_price_chart(etf_prices: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for etf in etf_prices.columns:
        normed = etf_prices[etf] / etf_prices[etf].dropna().iloc[0]
        fig.add_trace(go.Scatter(
            x=etf_prices.index, y=normed,
            mode="lines", name=etf, opacity=0.8,
        ))
    fig.update_layout(
        title="ETF Price History (normalised to 1.0)",
        xaxis_title="Date", yaxis_title="Normalised Price",
        hovermode="x unified", height=400,
        legend=dict(orientation="h", y=1.05),
    )
    return fig


# ---------------------------------------------------------------------------
# Metrics table helper
# ---------------------------------------------------------------------------

def metrics_table(backtest: pd.DataFrame) -> pd.DataFrame:
    from evaluation import compute_all_metrics
    return compute_all_metrics(backtest)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def sidebar() -> None:
    st.sidebar.title("ETF Sector Rotation")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Universe**")
    for etf, sector in cfg.SECTOR_MAP.items():
        st.sidebar.markdown(f"- `{etf}` — {sector}")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Config**")
    st.sidebar.markdown(f"- Top N: `{cfg.TOP_N}`")
    st.sidebar.markdown(f"- Max weight: `{cfg.MAX_WEIGHT:.0%}`")
    st.sidebar.markdown(f"- Horizon: `{cfg.FORWARD_DAYS}d`")
    st.sidebar.markdown(f"- Rebalance: `Monthly`")
    st.sidebar.markdown("---")
    st.sidebar.caption("University ML Final Project | Not financial advice")


# ---------------------------------------------------------------------------
# Main dashboard
# ---------------------------------------------------------------------------

def main() -> None:
    sidebar()

    st.title("📈 ETF Sector Rotation Dashboard")
    st.caption("Sector allocation model using XGBoost · University Final Project · Not financial advice")

    # ── Check artifacts exist ──────────────────────────────────────────
    backtest = load_backtest()
    alloc    = load_allocation()
    panel    = load_panel()
    feat_imp = load_feature_importance()
    etf_prices = load_etf_prices()

    if backtest is None or alloc is None:
        st.warning(
            "Artifacts not found. Please run the pipeline first:\n"
            "```\npython main.py\n```"
        )
        st.stop()

    # ── KPI row ───────────────────────────────────────────────────────
    st.markdown("## Summary Metrics")
    metrics = metrics_table(backtest)

    kpi_cols = st.columns(4)
    for i, col in enumerate(["model", "equal_weight", "spy"]):
        if col in metrics.columns:
            kpi_cols[i].metric(
                label={"model": "Model", "equal_weight": "Equal-Weight", "spy": "SPY"}.get(col, col),
                value=metrics.loc["CAGR", col] if "CAGR" in metrics.index else "–",
                delta=metrics.loc["Sharpe Ratio", col] + " Sharpe" if "Sharpe Ratio" in metrics.index else None,
            )

    st.markdown("---")

    # ── Full metrics table ────────────────────────────────────────────
    with st.expander("Full Performance Metrics", expanded=True):
        st.dataframe(metrics, use_container_width=True)

    st.markdown("---")

    # ── Equity curve ─────────────────────────────────────────────────
    st.markdown("## Equity Curve")
    st.plotly_chart(equity_chart(backtest), use_container_width=True)

    # ── Drawdown ─────────────────────────────────────────────────────
    st.markdown("## Drawdown")
    st.plotly_chart(drawdown_chart(backtest), use_container_width=True)

    st.markdown("---")

    # ── Latest allocation ─────────────────────────────────────────────
    st.markdown("## Latest Sector Allocation")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.plotly_chart(allocation_chart(alloc), use_container_width=True)
    with col2:
        st.plotly_chart(sector_ranking_chart(alloc), use_container_width=True)

    st.markdown("### Allocation Table")
    display_cols = ["ETF", "Sector", "predicted_score", "rank", "pct_weight"]
    display_cols = [c for c in display_cols if c in alloc.columns]
    st.dataframe(
        alloc[display_cols].style.format({
            "predicted_score": "{:+.4f}",
            "pct_weight": "{:.2f}%",
        }),
        use_container_width=True,
    )

    st.markdown("---")

    # ── Feature importance ────────────────────────────────────────────
    st.markdown("## Feature Importance")
    if not feat_imp.empty:
        st.plotly_chart(feature_importance_chart(feat_imp), use_container_width=True)
    else:
        st.info("Feature importance not available. Train the model first.")

    st.markdown("---")

    # ── ETF price history ─────────────────────────────────────────────
    if etf_prices is not None:
        st.markdown("## ETF Price History")
        st.plotly_chart(etf_price_chart(etf_prices), use_container_width=True)

    # ── Rolling returns heatmap ───────────────────────────────────────
    st.markdown("## Monthly Returns Heatmap")
    _monthly_heatmap(backtest)

    # ── Model card link ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## Reports")
    col_a, col_b = st.columns(2)
    with col_a:
        if cfg.EVAL_REPORT.exists():
            st.download_button(
                "Download Evaluation Report",
                data=cfg.EVAL_REPORT.read_bytes(),
                file_name="evaluation_report.md",
                mime="text/markdown",
            )
    with col_b:
        if cfg.MODEL_CARD.exists():
            st.download_button(
                "Download Model Card",
                data=cfg.MODEL_CARD.read_bytes(),
                file_name="model_card.md",
                mime="text/markdown",
            )

    # ── AI Insights ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 🤖 AI Insights")
    st.caption("Live streaming analysis from 3 Claude AI agents · Requires ANTHROPIC_API_KEY")

    _api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not _api_key:
        st.warning(
            "Add your Anthropic API key to `.env` to enable AI Insights:\n"
            "```\nANTHROPIC_API_KEY=sk-ant-...\n```"
        )
    else:
        tab_market, tab_strategy, tab_risk = st.tabs([
            "📊 Market Analyst",
            "🎯 Portfolio Strategist",
            "⚠️ Risk Monitor",
        ])

        with tab_market:
            st.markdown(
                "Reads the latest macro snapshot (VIX, yield spread, Fed Funds, CPI, etc.) and "
                "identifies the current market regime with sector-level implications."
            )
            if panel is None:
                st.info("Feature panel unavailable — run `python main.py` first.")
            elif st.button("▶ Run Market Analysis", key="btn_market"):
                try:
                    from agents import market_analyst_stream
                    st.write_stream(market_analyst_stream(panel, alloc))
                except Exception as exc:
                    st.error(f"Agent error: {exc}")

        with tab_strategy:
            st.markdown(
                "Reviews the ML-generated allocation for concentration risk, rotation "
                "opportunities, and potential model blind spots."
            )
            if st.button("▶ Run Portfolio Review", key="btn_strategy"):
                try:
                    from agents import portfolio_strategist_stream
                    st.write_stream(portfolio_strategist_stream(alloc, metrics, feat_imp))
                except Exception as exc:
                    st.error(f"Agent error: {exc}")

        with tab_risk:
            st.markdown(
                "Monitors drawdown, 63-day annualised volatility, and position concentration. "
                "Returns a 🟢/🟡/🔴 traffic-light rating with specific escalation thresholds."
            )
            if st.button("▶ Run Risk Assessment", key="btn_risk"):
                try:
                    from agents import risk_monitor_stream
                    st.write_stream(risk_monitor_stream(backtest, alloc))
                except Exception as exc:
                    st.error(f"Agent error: {exc}")

    st.caption("This dashboard is for academic / educational use only. Not financial advice.")


def _monthly_heatmap(backtest: pd.DataFrame) -> None:
    """Render a year × month heatmap of model monthly returns."""
    if "model" not in backtest.columns:
        return
    monthly = (1 + backtest["model"].dropna()).resample("ME").prod() - 1
    monthly.index = pd.to_datetime(monthly.index)
    df = pd.DataFrame({
        "Year":  monthly.index.year,
        "Month": monthly.index.month,
        "Return": monthly.values * 100,
    })
    pivot = df.pivot(index="Year", columns="Month", values="Return")
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"][:len(pivot.columns)]
    fig = px.imshow(
        pivot,
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
        text_auto=".1f",
        labels=dict(color="Return (%)"),
        title="Monthly Returns (%) — Model Portfolio",
        aspect="auto",
    )
    fig.update_layout(height=max(300, 40 * len(pivot)))
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
