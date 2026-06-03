"""
build_dataset.py – Assemble the panel dataset for model training.

Reads raw price data, computes features for every ETF, stacks into a
long panel (date × ETF rows), encodes categoricals, and saves to disk.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from features import build_features_for_etf, log_returns

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_panel(
    etf_prices: pd.DataFrame,
    spy: pd.Series | None = None,
    macro: pd.DataFrame | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Build the full long-panel feature dataset.

    Parameters
    ----------
    etf_prices : wide DataFrame of adjusted-close prices (index=Date, cols=ETFs)
    spy        : SPY adjusted-close series (optional)
    macro      : daily macro features from FRED (optional)
    save       : if True, write features.csv to data/processed/

    Returns
    -------
    pd.DataFrame  – long panel with one row per (date, ETF) and all features.
    """
    logger.info("Building feature panel…")

    # Equal-weight universe average (for relative-strength feature)
    etf_prices_sorted = etf_prices.sort_index()
    universe_avg = etf_prices_sorted.mean(axis=1)

    # SPY log-returns
    spy_log_rets: pd.Series | None = None
    if spy is not None and not spy.dropna().empty:
        spy_rets = np.log(spy / spy.shift(1))
        spy_log_rets = spy_rets.reindex(etf_prices_sorted.index)

    panels: list[pd.DataFrame] = []
    for ticker in cfg.ETFS:
        if ticker not in etf_prices_sorted.columns:
            logger.warning(f"Skipping {ticker} – not in price data")
            continue
        prices = etf_prices_sorted[ticker].dropna()
        logger.debug(f"  {ticker}: {len(prices)} price observations")

        feat_df = build_features_for_etf(
            ticker=ticker,
            prices=prices,
            universe_avg=universe_avg.reindex(prices.index),
            spy_log_rets=spy_log_rets,
            macro=macro,
        )
        panels.append(feat_df)

    panel = pd.concat(panels, axis=0)
    panel.reset_index(inplace=True)   # Date becomes a column
    panel.rename(columns={"index": "Date"}, inplace=True)
    panel["Date"] = pd.to_datetime(panel["Date"])
    panel.sort_values(["Date", "ETF"], inplace=True)
    panel.reset_index(drop=True, inplace=True)

    # One-hot encode ETF identity (drop one level to avoid multicollinearity)
    etf_dummies = pd.get_dummies(panel["ETF"], prefix="etf", drop_first=False, dtype=float)
    panel = pd.concat([panel, etf_dummies], axis=1)

    # Drop rows where ALL features are NaN (early period before windows fill)
    feature_cols = _feature_columns(panel)
    panel.dropna(subset=feature_cols, how="all", inplace=True)

    logger.info(f"Panel built: {panel.shape[0]:,} rows × {panel.shape[1]} cols")
    logger.info(f"Date range: {panel['Date'].min().date()} → {panel['Date'].max().date()}")

    if save:
        cfg.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        panel.to_csv(cfg.FEATURES_CSV, index=False)
        logger.info(f"Saved to {cfg.FEATURES_CSV}")

    return panel


# ---------------------------------------------------------------------------
# Train / validation / test split
# ---------------------------------------------------------------------------

def time_series_split(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological split on unique dates.

    Returns
    -------
    train, validation, test  –  three non-overlapping DataFrames.
    """
    dates = panel["Date"].sort_values().unique()
    n = len(dates)
    n_train = int(n * cfg.TRAIN_FRAC)
    n_val   = int(n * cfg.VAL_FRAC)

    train_dates = dates[:n_train]
    val_dates   = dates[n_train: n_train + n_val]
    test_dates  = dates[n_train + n_val:]

    train = panel[panel["Date"].isin(train_dates)].copy()
    val   = panel[panel["Date"].isin(val_dates)].copy()
    test  = panel[panel["Date"].isin(test_dates)].copy()

    logger.info(
        f"Split → train: {train['Date'].min().date()} – {train['Date'].max().date()} "
        f"({len(train):,} rows)  |  "
        f"val: {val['Date'].min().date()} – {val['Date'].max().date()} "
        f"({len(val):,} rows)  |  "
        f"test: {test['Date'].min().date()} – {test['Date'].max().date()} "
        f"({len(test):,} rows)"
    )
    return train, val, test


# ---------------------------------------------------------------------------
# Helper: identify feature columns
# ---------------------------------------------------------------------------

def _feature_columns(panel: pd.DataFrame) -> list[str]:
    """
    Return the list of pure feature columns (exclude metadata and targets).
    """
    exclude = {
        "Date", "ETF", "Sector",
        "future_21d_return", "future_21d_volatility", "target_score",
    }
    return [c for c in panel.columns if c not in exclude]


def get_feature_columns(panel: pd.DataFrame) -> list[str]:
    """Public API – same as _feature_columns."""
    return _feature_columns(panel)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def run(
    etf_prices: pd.DataFrame | None = None,
    spy: pd.Series | None = None,
    macro: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Convenience wrapper: optionally loads raw data from disk if arguments
    are not provided, then builds and saves the panel.
    """
    if etf_prices is None:
        raw_path = cfg.DATA_RAW / "etf_prices.parquet"
        etf_prices = pd.read_parquet(raw_path)
        logger.info(f"Loaded ETF prices from {raw_path}")

    if spy is None:
        spy_path = cfg.DATA_RAW / "spy.parquet"
        if spy_path.exists():
            spy = pd.read_parquet(spy_path)["SPY"]

    if macro is None:
        macro_path = cfg.DATA_RAW / "macro.parquet"
        if macro_path.exists():
            macro = pd.read_parquet(macro_path)

    return build_panel(etf_prices, spy=spy, macro=macro, save=True)


if __name__ == "__main__":
    logging.basicConfig(level=cfg.LOG_LEVEL, format=cfg.LOG_FMT)
    run()
