"""
download_data.py – Data acquisition layer.

Responsibilities
----------------
1. Load ETF closing prices from the local Excel file (Book3.xlsx).
2. Download SPY benchmark from yfinance.
3. Download FRED / ALFRED macro series.
4. Persist everything to data/raw/ as Parquet files.
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import yfinance as yf

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ETF prices from Excel
# ---------------------------------------------------------------------------

def load_etf_prices_from_excel(path: Path | str = cfg.EXCEL_FILE) -> pd.DataFrame:
    """
    Read the Excel workbook produced by Bloomberg / internal systems.

    Column names look like 'XLE US Equity  (R1)' – we extract the ticker
    prefix (e.g. 'XLE') and return a DataFrame with Date as index.

    Returns
    -------
    pd.DataFrame
        Wide format: columns = ETF tickers, index = Date (daily, ascending).
    """
    path = Path(path)
    logger.info(f"Loading ETF prices from {path.name}")

    df = pd.read_excel(path, sheet_name="Sheet1", engine="openpyxl")
    df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)       # ascending chronological order

    # Rename 'XLE US Equity  (R1)' → 'XLE'
    rename = {}
    for col in df.columns:
        ticker = col.strip().split()[0]
        rename[col] = ticker
    df.rename(columns=rename, inplace=True)

    # Keep only ETFs defined in config
    keep = [c for c in cfg.ETFS if c in df.columns]
    missing = [e for e in cfg.ETFS if e not in df.columns]
    if missing:
        logger.warning(f"ETFs not found in Excel: {missing}")
    df = df[keep].copy()
    df = df.astype(float)
    df.dropna(how="all", inplace=True)

    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} ETFs "
                f"[{df.index[0].date()} → {df.index[-1].date()}]")
    return df


# ---------------------------------------------------------------------------
# SPY benchmark via yfinance
# ---------------------------------------------------------------------------

def download_spy(start: str, end: str) -> pd.Series:
    """
    Download SPY adjusted-close prices from Yahoo Finance.

    Parameters
    ----------
    start, end : str  (YYYY-MM-DD)
    """
    logger.info(f"Downloading SPY from yfinance [{start} → {end}]")
    ticker = yf.Ticker(cfg.BENCHMARK)
    hist = ticker.history(start=start, end=end, auto_adjust=True)
    if hist.empty:
        logger.warning("yfinance returned empty data for SPY – using NaN column")
        return pd.Series(dtype=float, name="SPY")

    spy = hist["Close"].rename("SPY")
    spy.index = pd.to_datetime(spy.index).tz_localize(None)
    spy.sort_index(inplace=True)
    logger.info(f"SPY: {len(spy)} rows [{spy.index[0].date()} → {spy.index[-1].date()}]")
    return spy


# ---------------------------------------------------------------------------
# FRED / ALFRED macro features
# ---------------------------------------------------------------------------

def _compute_yoy(series: pd.Series) -> pd.Series:
    """Year-over-year percentage change for monthly macro series."""
    return series.pct_change(12) * 100


def download_fred_series(
    start: str,
    end: str,
    api_key: str = cfg.FRED_API_KEY,
    use_alfred: bool = cfg.USE_ALFRED_VINTAGES,
) -> pd.DataFrame:
    """
    Fetch macro features from FRED (or ALFRED for vintage data).

    Requires `fredapi` package and a valid FRED_API_KEY in config / env.

    ALFRED vintages (use_alfred=True) retrieve the data that would have
    been available in real-time on each date, preventing look-ahead bias
    for lagged monthly releases.

    Parameters
    ----------
    start, end      : date range strings (YYYY-MM-DD)
    api_key         : FRED API key (free at fred.stlouisfed.org)
    use_alfred      : if True, use real-time vintage data from ALFRED

    Returns
    -------
    pd.DataFrame  with daily index (forward-filled) and macro columns.
    """
    if not api_key:
        logger.warning(
            "FRED_API_KEY not set. Skipping macro features. "
            "Set the environment variable FRED_API_KEY=<your_key> to enable."
        )
        return pd.DataFrame()

    try:
        from fredapi import Fred
    except ImportError:
        logger.warning("fredapi not installed (`pip install fredapi`). Skipping macro features.")
        return pd.DataFrame()

    fred = Fred(api_key=api_key)
    frames: dict[str, pd.Series] = {}

    for series_id, col_name in cfg.FRED_SERIES.items():
        for attempt in range(3):          # up to 3 retries on rate-limit errors
            try:
                import time as _time
                if attempt > 0:
                    _time.sleep(2 ** attempt)   # 2s, 4s back-off

                if use_alfred:
                    raw = fred.get_series_all_releases(series_id)
                    raw = (
                        raw.sort_values("realtime_start")
                        .drop_duplicates(subset="date", keep="last")
                        .set_index("date")["value"]
                    )
                else:
                    raw = fred.get_series(series_id, observation_start=start,
                                          observation_end=end)

                raw = pd.to_numeric(raw, errors="coerce")
                raw.index = pd.to_datetime(raw.index)

                if col_name in ("cpi_yoy", "indpro_yoy"):
                    raw = _compute_yoy(raw)

                frames[col_name] = raw
                logger.info(f"FRED {series_id} ({col_name}): {len(raw)} obs")
                break           # success – no retry needed

            except Exception as exc:
                logger.warning(f"FRED {series_id} attempt {attempt+1} failed: {exc}")
                if attempt == 2:
                    logger.warning(f"Skipping {series_id} after 3 attempts")

    if not frames:
        return pd.DataFrame()

    # Align to a daily calendar and forward-fill monthly releases
    daily_idx = pd.date_range(start=start, end=end, freq="B")
    macro = pd.DataFrame(index=daily_idx)
    for name, series in frames.items():
        macro[name] = series.reindex(daily_idx, method="ffill")

    # Derive yield-curve slope if both legs available
    if "treasury_10y" in macro.columns and "treasury_2y" in macro.columns:
        macro["yield_slope"] = macro["treasury_10y"] - macro["treasury_2y"]

    # Normalised VIX percentile rank (rolling 252-day)
    if "vix" in macro.columns:
        macro["vix_pctile"] = (
            macro["vix"]
            .rolling(252, min_periods=63)
            .rank(pct=True)
        )

    macro.dropna(how="all", inplace=True)
    logger.info(f"Macro DataFrame: {macro.shape} [{macro.index[0].date()} → {macro.index[-1].date()}]")
    return macro


# ---------------------------------------------------------------------------
# Master download / save routine
# ---------------------------------------------------------------------------

def run(save: bool = True) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Full data download pipeline.

    Returns
    -------
    etf_prices  : pd.DataFrame  (wide, daily, adjusted close)
    spy         : pd.Series     (daily adjusted close)
    macro       : pd.DataFrame  (daily macro features, may be empty)
    """
    # 1. ETF prices from Excel
    etf_prices = load_etf_prices_from_excel()

    start_str = etf_prices.index[0].strftime("%Y-%m-%d")
    end_str   = etf_prices.index[-1].strftime("%Y-%m-%d")

    # 2. SPY benchmark
    spy = download_spy(start=start_str, end=end_str)

    # 3. FRED macro
    macro = download_fred_series(start=start_str, end=end_str)

    if save:
        cfg.DATA_RAW.mkdir(parents=True, exist_ok=True)
        etf_prices.to_parquet(cfg.DATA_RAW / "etf_prices.parquet")
        spy.to_frame().to_parquet(cfg.DATA_RAW / "spy.parquet")
        if not macro.empty:
            macro.to_parquet(cfg.DATA_RAW / "macro.parquet")
        logger.info("Raw data saved to data/raw/")

    return etf_prices, spy, macro


if __name__ == "__main__":
    logging.basicConfig(level=cfg.LOG_LEVEL, format=cfg.LOG_FMT)
    run()
