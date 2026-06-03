"""
backtest.py – Vectorised portfolio backtest engine.

The engine applies model predictions at each monthly rebalance date,
computes daily portfolio returns, and benchmarks against:
  • SPY (if price data available)
  • Equal-weight sector portfolio (all 11 ETFs, no model)

No look-ahead bias: predictions at date t use only information up to t.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from portfolio import compute_weights, get_rebalance_dates

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core backtester
# ---------------------------------------------------------------------------

class Backtester:
    """
    Vectorised walk-forward backtester.

    Parameters
    ----------
    etf_prices  : wide DataFrame of adjusted-close prices (index=Date)
    panel       : long feature panel (for model predictions)
    models      : dict of trained XGBoost models
    feature_cols: feature column names
    spy_prices  : SPY price series (optional)
    """

    def __init__(
        self,
        etf_prices: pd.DataFrame,
        panel: pd.DataFrame,
        models: dict,
        feature_cols: list[str],
        spy_prices: Optional[pd.Series] = None,
    ) -> None:
        self.etf_prices   = etf_prices.sort_index()
        self.panel        = panel
        self.models       = models
        self.feature_cols = feature_cols
        self.spy_prices   = spy_prices.sort_index() if spy_prices is not None else None

        # Daily return matrix for ETFs
        self.etf_returns = self.etf_prices.pct_change()

    # ------------------------------------------------------------------
    def run(self, start_date: Optional[str] = None) -> pd.DataFrame:
        """
        Execute the backtest with daily rebalancing.

        All predictions are computed in a single vectorised batch before the
        loop begins, so per-day overhead is just a dict lookup + weight calc.

        Parameters
        ----------
        start_date : backtest start date (YYYY-MM-DD); defaults to full history.

        Returns
        -------
        (results_df, weight_history_df)
        """
        from train_model import batch_predict_all_dates
        from portfolio import precompute_rolling_cov

        dates = self.etf_returns.index
        if start_date:
            dates = dates[dates >= pd.Timestamp(start_date)]

        rebalance_dates = set(get_rebalance_dates(dates))
        logger.info(
            f"Backtest: {dates[0].date()} -> {dates[-1].date()}, "
            f"{len(rebalance_dates)} rebalance days "
            f"(freq='{cfg.REBALANCE_FREQ}', method='{cfg.PORTFOLIO_METHOD}')"
        )

        # ── Pre-compute all scores in one batch pass ───────────────────────
        logger.info("Running batch inference over full panel…")
        pred_tables = batch_predict_all_dates(
            self.panel, self.models, self.feature_cols
        )
        score_table  = pred_tables["score"]   # Date × ETF → predicted_score
        return_table = pred_tables["return"]  # Date × ETF → predicted_return

        # ── Smooth scores with EMA to dampen day-to-day noise ──────────────
        span = cfg.SCORE_EMA_SPAN
        if span > 1:
            score_table  = score_table.ewm(span=span, min_periods=1).mean()
            return_table = return_table.ewm(span=span, min_periods=1).mean()
            logger.info(f"EMA smoothing applied (span={span})")

        # ── Pre-compute rolling Ledoit-Wolf covariance matrices ────────────
        etfs_available = [e for e in cfg.ETFS if e in self.etf_prices.columns]
        cov_store: dict = {}
        if cfg.PORTFOLIO_METHOD == "max_sharpe":
            cov_store = precompute_rolling_cov(
                self.etf_returns[etfs_available], window=cfg.COV_WINDOW
            )

        # ── Pre-index as numpy arrays – eliminates per-row pandas overhead ──
        n_etfs  = len(etfs_available)
        use_ms  = cfg.PORTFOLIO_METHOD == "max_sharpe"
        etf_cols = score_table.columns.tolist()

        # ETF returns: (T, n_etfs) aligned to backtest dates
        ret_arr   = self.etf_returns[etfs_available].reindex(dates).fillna(0.0).values.astype(float)

        # Score / return signal arrays aligned to backtest dates
        score_arr  = score_table.reindex(dates).values.astype(float)
        return_arr = return_table.reindex(dates).values.astype(float)

        # SPY returns array
        spy_pct = self.spy_prices.pct_change().reindex(dates).values.astype(float) \
                  if self.spy_prices is not None else np.full(len(dates), np.nan)

        # Initialise numpy weight vectors
        eq_w     = np.full(n_etfs, 1.0 / n_etfs)
        cw_arr   = eq_w.copy()
        prev_arr = cw_arr.copy()

        model_rets    = np.zeros(len(dates))
        eq_rets       = np.zeros(len(dates))
        turnovers_out = np.zeros(len(dates))

        weight_history: list[dict] = []
        actual_rebalances = 0
        held_set: frozenset[str] = frozenset()

        for i, date in enumerate(dates):
            turnover = 0.0

            if date in rebalance_dates and not np.all(np.isnan(score_arr[i])):
                signal_vals = return_arr[i] if use_ms else score_arr[i]
                signal      = pd.Series(signal_vals, index=etf_cols)
                cov         = cov_store.get(date, None) if use_ms else None

                try:
                    cw_new     = compute_weights(signal, cfg.TOP_N, cfg.MAX_WEIGHT, cfg.MIN_WEIGHT, cov=cov)
                    cw_new_arr = np.array([cw_new.get(e, 0.0) for e in etfs_available])
                    new_set    = frozenset(e for e, w in cw_new.items() if w > 0)
                    to_cand    = float(np.abs(prev_arr - cw_new_arr).sum() / 2)

                    if new_set != held_set or to_cand >= cfg.MIN_TURNOVER_THRESHOLD:
                        turnover  = to_cand
                        cw_arr    = cw_new_arr
                        prev_arr  = cw_arr.copy()
                        held_set  = new_set
                        actual_rebalances += 1
                        weight_history.append({
                            "date": date,
                            **{f"w_{etfs_available[j]}": float(cw_arr[j]) for j in range(n_etfs)},
                        })
                except Exception as exc:
                    logger.debug(f"Weight compute failed on {date}: {exc}")

            model_rets[i]    = float(cw_arr @ ret_arr[i]) - turnover * cfg.TRANSACTION_COST
            eq_rets[i]       = float(eq_w   @ ret_arr[i])
            turnovers_out[i] = turnover

        results = pd.DataFrame({
            "model":        model_rets,
            "equal_weight": eq_rets,
            "spy":          spy_pct,
            "turnover":     turnovers_out,
        }, index=dates)

        for col in ["model", "equal_weight", "spy"]:
            results[f"{col}_equity"] = (1 + results[col].fillna(0)).cumprod()

        if cfg.INITIAL_CAPITAL != 1.0:
            for col in ["model", "equal_weight", "spy"]:
                results[f"{col}_equity"] *= cfg.INITIAL_CAPITAL

        logger.info(
            f"Backtest complete. Actual rebalances executed: {actual_rebalances} "
            f"(of {len(rebalance_dates)} eligible days, "
            f"threshold={cfg.MIN_TURNOVER_THRESHOLD:.0%})"
        )
        return results, pd.DataFrame(weight_history)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _portfolio_daily_return(weights: pd.Series, day_rets: pd.Series) -> float:
    """Dot-product of weights and daily ETF returns."""
    if weights.empty or day_rets.empty:
        return 0.0
    common = weights.index.intersection(day_rets.index)
    if common.empty:
        return 0.0
    return float((weights[common] * day_rets[common]).sum())


def _spy_return(spy_prices: Optional[pd.Series], date: pd.Timestamp) -> float:
    if spy_prices is None or date not in spy_prices.index:
        return np.nan
    idx = spy_prices.index.get_loc(date)
    if idx == 0:
        return np.nan
    prev = spy_prices.iloc[idx - 1]
    curr = spy_prices.iloc[idx]
    if prev == 0 or np.isnan(prev):
        return np.nan
    return float(curr / prev - 1)


def _compute_turnover(old_weights: pd.Series, new_weights: pd.Series) -> float:
    """
    One-way turnover: sum of absolute weight changes / 2.
    A turnover of 1.0 means the entire portfolio was replaced.
    """
    all_etfs = old_weights.index.union(new_weights.index)
    old = old_weights.reindex(all_etfs, fill_value=0.0)
    new = new_weights.reindex(all_etfs, fill_value=0.0)
    return float((old - new).abs().sum() / 2)


# ---------------------------------------------------------------------------
# Convenience run function
# ---------------------------------------------------------------------------

def run(
    etf_prices: Optional[pd.DataFrame] = None,
    panel: Optional[pd.DataFrame] = None,
    models: Optional[dict] = None,
    feature_cols: Optional[list[str]] = None,
    spy_prices: Optional[pd.Series] = None,
    start_date: Optional[str] = None,
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load data from disk (if not provided) and run the full backtest.

    Returns
    -------
    (results_df, weight_history_df)
    """
    from build_dataset import get_feature_columns
    from train_model import load_models

    if etf_prices is None:
        etf_prices = pd.read_parquet(cfg.DATA_RAW / "etf_prices.parquet")

    if panel is None:
        panel = pd.read_csv(cfg.FEATURES_CSV, parse_dates=["Date"])

    if feature_cols is None:
        feature_cols = get_feature_columns(panel)

    if models is None:
        models = load_models()

    if spy_prices is None:
        spy_path = cfg.DATA_RAW / "spy.parquet"
        if spy_path.exists():
            spy_prices = pd.read_parquet(spy_path)["SPY"]

    bt = Backtester(
        etf_prices=etf_prices,
        panel=panel,
        models=models,
        feature_cols=feature_cols,
        spy_prices=spy_prices,
    )
    results, weight_history = bt.run(start_date=start_date)

    if save:
        cfg.DATA_ARTIFACTS.mkdir(parents=True, exist_ok=True)
        results.to_csv(cfg.BACKTEST_CSV)
        weight_history.to_csv(cfg.DATA_ARTIFACTS / "weight_history.csv", index=False)
        logger.info(f"Backtest results saved → {cfg.BACKTEST_CSV}")

    return results, weight_history


if __name__ == "__main__":
    logging.basicConfig(level=cfg.LOG_LEVEL, format=cfg.LOG_FMT)
    results, _ = run()
    print(results[["model_equity", "equal_weight_equity", "spy_equity"]].tail(10))
