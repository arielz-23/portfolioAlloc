"""
config.py – Central configuration for the ETF Sector Rotation Model.

All tuneable parameters live here. Edit this file to change behaviour
without touching any other source file.
"""
from __future__ import annotations
import os
from pathlib import Path

# Load .env from the project root (one level up from src/)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass  # dotenv optional – env vars can be set manually instead

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW        = ROOT / "data" / "raw"
DATA_PROCESSED  = ROOT / "data" / "processed"
DATA_ARTIFACTS  = ROOT / "data" / "artifacts"
REPORTS         = ROOT / "reports"

EXCEL_FILE        = ROOT / "Book3.xlsx"
FEATURES_CSV      = DATA_PROCESSED / "features.csv"
BACKTEST_CSV      = DATA_ARTIFACTS / "backtest_results.csv"
ALLOCATION_CSV    = DATA_ARTIFACTS / "latest_allocation.csv"
MODEL_PATH        = DATA_ARTIFACTS / "xgb_model.pkl"
RETURN_MODEL_PATH = DATA_ARTIFACTS / "xgb_return_model.pkl"
VOL_MODEL_PATH    = DATA_ARTIFACTS / "xgb_vol_model.pkl"
EVAL_REPORT       = REPORTS / "evaluation_report.md"
MODEL_CARD        = REPORTS / "model_card.md"

# ---------------------------------------------------------------------------
# ETF Universe
# ---------------------------------------------------------------------------
ETFS: list[str] = [
    "XLK", "XLV", "XLF", "XLI", "XLY",
    "XLP", "XLE", "XLU", "XLB", "XLRE", "XLC",
]

SECTOR_MAP: dict[str, str] = {
    "XLK":  "Technology",
    "XLV":  "Healthcare",
    "XLF":  "Financials",
    "XLI":  "Industrials",
    "XLY":  "Consumer Discretionary",
    "XLP":  "Consumer Staples",
    "XLE":  "Energy",
    "XLU":  "Utilities",
    "XLB":  "Materials",
    "XLRE": "Real Estate",
    "XLC":  "Communication Services",
}

BENCHMARK = "SPY"

# ---------------------------------------------------------------------------
# Target / labelling
# ---------------------------------------------------------------------------
FORWARD_DAYS: int = 21          # prediction horizon (trading days)

# ---------------------------------------------------------------------------
# Feature engineering windows (trading days)
# ---------------------------------------------------------------------------
RETURN_WINDOWS:    list[int] = [5, 21, 63, 126, 252]
VOL_WINDOWS:       list[int] = [21, 63]
MA_WINDOWS:        list[int] = [50, 200]
CORR_WINDOW:       int       = 63   # rolling correlation with SPY

# ---------------------------------------------------------------------------
# Train / validation / test split fractions
# ---------------------------------------------------------------------------
TRAIN_FRAC: float = 0.70
VAL_FRAC:   float = 0.15
# test_frac = 1 - TRAIN_FRAC - VAL_FRAC = 0.15

# ---------------------------------------------------------------------------
# XGBoost hyper-parameters
# Loss: reg:squarederror  (MSE = mean squared error)
# Regularisation:
#   reg_alpha  – L1 (Lasso)  : drives small feature weights to zero (sparsity)
#   reg_lambda – L2 (Ridge)  : penalises large weights (shrinkage)
#   gamma      – min loss reduction required before a node is split (pruning)
#   min_child_weight – min sum of sample weights in a leaf (anti-overfit)
# ---------------------------------------------------------------------------
XGB_PARAMS: dict = {
    "n_estimators":       500,
    "max_depth":          4,
    "learning_rate":      0.05,
    "subsample":          0.8,
    "colsample_bytree":   0.7,
    "min_child_weight":   10,       # raised: prevents splits on tiny groups
    "gamma":              0.1,      # min loss reduction to split a node
    "reg_alpha":          0.5,      # L1 – sparsity over features
    "reg_lambda":         2.0,      # L2 – weight shrinkage
    "random_state":       42,
    "n_jobs":            -1,
    "objective":          "reg:squarederror",   # MSE loss
    "early_stopping_rounds": 50,
}

# Use two-model approach (return + volatility) instead of one combined model
TWO_MODEL: bool = True

# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------
TOP_N:      int   = 5      # number of ETFs to hold
MAX_WEIGHT: float = 0.35   # per-ETF weight cap
MIN_WEIGHT: float = 0.0    # long-only

# Portfolio optimisation method:
#   "score"      – proportional weights from predicted score (original)
#   "max_sharpe" – mean-variance tangency portfolio via SLSQP
#                  uses predicted_return as μ and rolling Ledoit-Wolf Σ
PORTFOLIO_METHOD: str = "max_sharpe"

# Covariance estimation
COV_WINDOW:  int  = 63     # rolling window for Σ (trading days)
SHRINKAGE:   bool = True   # Ledoit-Wolf shrinkage (more stable for small N)

# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------
REBALANCE_FREQ: str = "B"   # "B" = daily  |  "BME" = monthly

# Signal smoothing: EMA of predicted scores over N days before weight calc.
# Prevents noise-driven rank-flipping when rebalancing daily.
# Set to 1 to disable smoothing.
SCORE_EMA_SPAN: int = 21    # EMA span matches the 21-day prediction horizon

# Rebalance trigger: only trade if EITHER condition is met:
#   1. The top-N composition changes (a different ETF enters/exits the held set)
#   2. One-way portfolio turnover exceeds MIN_TURNOVER_THRESHOLD (weight drift catch-all)
# This avoids churning when the same ETFs are ranked top-N but scores shift slightly.
MIN_TURNOVER_THRESHOLD: float = 0.05   # 5% one-way threshold
INITIAL_CAPITAL: float = 1_000_000.0
TRANSACTION_COST: float = 0.0005  # 5 bps per leg (one-way)

# ---------------------------------------------------------------------------
# Annualisation
# ---------------------------------------------------------------------------
TRADING_DAYS_PER_YEAR: int = 252
RISK_FREE_RATE: float = 0.04   # for Sharpe / Calmar denominator

# ---------------------------------------------------------------------------
# FRED / ALFRED API
# ---------------------------------------------------------------------------
# Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_KEY: str = os.getenv("FRED_API_KEY", "")

# FRED series to download (all forward-filled to daily frequency)
FRED_SERIES: dict[str, str] = {
    "T10Y2Y":        "yield_spread_10y2y",    # daily
    "FEDFUNDS":      "fed_funds_rate",         # monthly
    "CPIAUCSL":      "cpi_yoy",               # monthly (we compute YoY pct chg)
    "UNRATE":        "unemployment_rate",      # monthly
    "INDPRO":        "indpro_yoy",            # monthly (YoY pct chg)
    "BAMLH0A0HYM2":  "hy_spread",             # daily
    "VIXCLS":        "vix",                   # daily
    "DEXUSEU":       "eur_usd",               # daily
    "DGS10":         "treasury_10y",          # daily
    "DGS2":          "treasury_2y",           # daily
}

# ALFRED vintage retrieval – set to True to use real-time vintages
#   (avoids look-ahead bias for monthly releases)
USE_ALFRED_VINTAGES: bool = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = "INFO"
LOG_FMT: str   = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
