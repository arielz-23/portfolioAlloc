"""
features.py – Feature engineering for the ETF sector-rotation model.

All features are computed using only past data to prevent look-ahead bias.
The panel is built in build_dataset.py; this module contains the
individual feature functions that operate on a single ETF's price series.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Price-based single-ETF features
# ---------------------------------------------------------------------------

def log_returns(prices: pd.Series) -> pd.Series:
    """Daily log-return series."""
    return np.log(prices / prices.shift(1))


def momentum(prices: pd.Series, window: int) -> pd.Series:
    """Simple N-day price return (close-to-close)."""
    return prices / prices.shift(window) - 1


def realized_volatility(log_rets: pd.Series, window: int) -> pd.Series:
    """
    Annualised realised volatility from daily log-returns.
    std(log_rets over window) * sqrt(252)
    """
    return log_rets.rolling(window, min_periods=window // 2).std() * np.sqrt(cfg.TRADING_DAYS_PER_YEAR)


def downside_volatility(log_rets: pd.Series, window: int = 63, mar: float = 0.0) -> pd.Series:
    """
    Downside (semi-) deviation.
    Only negative excess returns below MAR are included.
    Annualised.
    """
    excess = log_rets - mar / cfg.TRADING_DAYS_PER_YEAR
    downside = excess.clip(upper=0)
    return downside.rolling(window, min_periods=window // 2).std() * np.sqrt(cfg.TRADING_DAYS_PER_YEAR)


def ma_distance(prices: pd.Series, window: int) -> pd.Series:
    """
    Distance from N-day moving average, expressed as a fraction.
    (price / MA(window)) - 1
    """
    ma = prices.rolling(window, min_periods=window // 2).mean()
    return prices / ma - 1


def relative_strength_vs_universe(
    prices: pd.Series,
    universe_avg_prices: pd.Series,
    window: int = 21,
) -> pd.Series:
    """
    Relative momentum: N-day return of this ETF minus N-day return of the
    equal-weight universe average. Positive → outperforming the group.
    """
    etf_ret = prices / prices.shift(window) - 1
    univ_ret = universe_avg_prices / universe_avg_prices.shift(window) - 1
    return etf_ret - univ_ret


def rolling_correlation(
    log_rets: pd.Series,
    benchmark_rets: pd.Series,
    window: int = cfg.CORR_WINDOW,
) -> pd.Series:
    """Rolling N-day correlation with a benchmark (e.g., SPY)."""
    return log_rets.rolling(window, min_periods=window // 2).corr(benchmark_rets)


def rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index (0-100)."""
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def bollinger_pct_b(prices: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.Series:
    """
    Bollinger Band %B: position of price within the band.
    0 = lower band, 0.5 = middle (SMA), 1 = upper band.
    """
    sma = prices.rolling(window, min_periods=window // 2).mean()
    std = prices.rolling(window, min_periods=window // 2).std()
    upper = sma + n_std * std
    lower = sma - n_std * std
    return (prices - lower) / (upper - lower + 1e-10)


def volume_adjusted_momentum(
    prices: pd.Series,
    volume: Optional[pd.Series],
    window: int = 21,
) -> Optional[pd.Series]:
    """
    If volume data is available, return volume-weighted momentum.
    Otherwise returns None (gracefully skipped in build_dataset).
    """
    if volume is None or volume.isna().all():
        return None
    vwap = (prices * volume).rolling(window).sum() / volume.rolling(window).sum()
    return prices / vwap - 1


# ---------------------------------------------------------------------------
# Target construction (uses FUTURE data – applied only at label creation time)
# ---------------------------------------------------------------------------

def future_return(prices: pd.Series, horizon: int = cfg.FORWARD_DAYS) -> pd.Series:
    """
    Forward N-day simple return. Computed using FUTURE prices.
    This MUST only be attached to the dataset as a label,
    and the feature matrix must be trained on rows where the label exists.
    """
    return prices.shift(-horizon) / prices - 1


def future_realized_vol(log_rets: pd.Series, horizon: int = cfg.FORWARD_DAYS) -> pd.Series:
    """
    Realised volatility of the NEXT `horizon` trading days.  Annualised.

    Vectorised: at date t, computes std of log_rets[t+1 … t+horizon].
    Uses a strided numpy view — O(N) and no Python loops.
    """
    n = horizon
    vals = log_rets.values.astype(float)
    result = np.full(len(vals), np.nan)
    # Build an (N-n) × n matrix of future windows using stride tricks
    length = len(vals) - n
    if length <= 0:
        return pd.Series(result, index=log_rets.index)
    shape   = (length, n)
    strides = (vals.strides[0], vals.strides[0])
    windows = np.lib.stride_tricks.as_strided(vals[1:], shape=shape, strides=strides)
    # Count valid observations per window
    valid = np.sum(~np.isnan(windows), axis=1)
    min_obs = max(n // 2, 5)
    mask = valid >= min_obs
    # nanstd per row
    with np.errstate(invalid="ignore", divide="ignore"):
        row_std = np.where(mask, np.nanstd(windows, ddof=1, axis=1), np.nan)
    result[:length] = row_std * np.sqrt(cfg.TRADING_DAYS_PER_YEAR)
    return pd.Series(result, index=log_rets.index)


def target_score(fwd_ret: pd.Series, fwd_vol: pd.Series) -> pd.Series:
    """
    Risk-adjusted return score: future_return / future_volatility.
    Analogous to the ex-ante Sharpe ratio.
    """
    return fwd_ret / fwd_vol.replace(0, np.nan)


# ---------------------------------------------------------------------------
# Panel-level feature assembly
# ---------------------------------------------------------------------------

def build_features_for_etf(
    ticker: str,
    prices: pd.Series,
    universe_avg: pd.Series,
    spy_log_rets: Optional[pd.Series] = None,
    macro: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compute all features for a single ETF and return a long DataFrame.

    Parameters
    ----------
    ticker          : ETF symbol string
    prices          : daily adjusted-close prices
    universe_avg    : equal-weight average of all ETF prices (same index)
    spy_log_rets    : log-return series for SPY (optional)
    macro           : daily macro DataFrame (optional, from FRED)

    Returns
    -------
    pd.DataFrame with feature columns and the ticker/sector as metadata.
    """
    prices = prices.dropna()
    log_rets = log_returns(prices)

    rows: dict[str, pd.Series] = {}

    # Momentum / returns
    for w in cfg.RETURN_WINDOWS:
        rows[f"ret_{w}d"] = momentum(prices, w)

    # Realised volatility
    for w in cfg.VOL_WINDOWS:
        rows[f"rvol_{w}d"] = realized_volatility(log_rets, w)

    # Downside volatility (63-day)
    rows["dvol_63d"] = downside_volatility(log_rets, 63)

    # Moving-average distance
    for w in cfg.MA_WINDOWS:
        rows[f"ma_dist_{w}d"] = ma_distance(prices, w)

    # Relative strength vs universe
    for w in [21, 63]:
        rows[f"rs_vs_univ_{w}d"] = relative_strength_vs_universe(prices, universe_avg, w)

    # Correlation with SPY
    if spy_log_rets is not None:
        spy_aligned = spy_log_rets.reindex(log_rets.index)
        rows["corr_spy_63d"] = rolling_correlation(log_rets, spy_aligned, cfg.CORR_WINDOW)

    # Technicals
    rows["rsi_14d"]      = rsi(prices, 14)
    rows["bband_pct_b"]  = bollinger_pct_b(prices, 20)

    # Volatility-of-volatility (dispersion of vol estimates)
    rows["vol_of_vol"] = (
        realized_volatility(log_rets, cfg.VOL_WINDOWS[0])
        .rolling(cfg.VOL_WINDOWS[1], min_periods=21)
        .std()
    )

    # Construct feature DataFrame
    feat = pd.DataFrame(rows)
    feat.index.name = "Date"

    # Attach ETF metadata
    feat.insert(0, "ETF",    ticker)
    feat.insert(1, "Sector", cfg.SECTOR_MAP.get(ticker, ticker))

    # Merge macro (forward-filled, so no look-ahead)
    if macro is not None and not macro.empty:
        feat = feat.join(macro.reindex(feat.index, method="ffill"), how="left")

    # Target labels (uses future data – kept for dataset construction)
    fwd_ret  = future_return(prices, cfg.FORWARD_DAYS)
    fwd_vol  = future_realized_vol(log_rets, cfg.FORWARD_DAYS)
    feat["future_21d_return"]     = fwd_ret.reindex(feat.index)
    feat["future_21d_volatility"] = fwd_vol.reindex(feat.index)
    feat["target_score"]          = target_score(
        feat["future_21d_return"], feat["future_21d_volatility"]
    )

    return feat
