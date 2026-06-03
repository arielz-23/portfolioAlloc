"""
portfolio.py – Portfolio construction and weight optimisation.

Two construction methods (cfg.PORTFOLIO_METHOD):

  "score"      – weights proportional to max(0, predicted_score).
                 Fast, interpretable, but ignores correlations.

  "max_sharpe" – mean-variance tangency portfolio.
                 Maximises (wᵀμ - rf) / sqrt(wᵀΣw) via SLSQP.
                 μ  = model predicted annualised return per ETF
                 Σ  = Ledoit-Wolf shrunk rolling covariance (annualised)
                 Falls back to "score" if optimisation fails.

Constraints (both methods)
--------------------------
* Long-only  (w_i ≥ 0)
* Fully invested  (Σw = 1)
* Per-ETF cap: w_i ≤ MAX_WEIGHT (default 35 %)
* Monthly / daily rebalance cadence enforced by the backtester
* No leverage
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Covariance estimation
# ---------------------------------------------------------------------------

def ledoit_wolf_cov(returns: pd.DataFrame) -> np.ndarray:
    """
    Ledoit-Wolf shrunk sample covariance matrix.

    Shrinkage pulls the sample Σ toward a scaled identity matrix,
    reducing estimation error when the number of assets (11) is not
    much smaller than the window length (63 days).

    Returns an annualised covariance matrix (multiplied by 252).
    """
    from sklearn.covariance import LedoitWolf
    clean = returns.dropna(how="any")
    if len(clean) < 10:
        return np.eye(returns.shape[1]) * 0.04   # fallback: ~20 % vol each
    lw = LedoitWolf(assume_centered=False)
    lw.fit(clean.values)
    return lw.covariance_ * cfg.TRADING_DAYS_PER_YEAR   # annualise


def sample_cov(returns: pd.DataFrame) -> np.ndarray:
    """Plain sample covariance, annualised."""
    clean = returns.dropna(how="any")
    return clean.cov().values * cfg.TRADING_DAYS_PER_YEAR


def build_cov_matrix(returns_window: pd.DataFrame) -> np.ndarray:
    """Return the covariance matrix for a slice of ETF returns."""
    if cfg.SHRINKAGE:
        return ledoit_wolf_cov(returns_window)
    return sample_cov(returns_window)


# ---------------------------------------------------------------------------
# Max-Sharpe optimiser (SLSQP)
# ---------------------------------------------------------------------------

def _neg_sharpe(
    w: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    rf: float,
) -> float:
    """Objective: –Sharpe ratio (minimise to maximise Sharpe)."""
    port_ret = float(w @ mu)
    port_var = float(w @ cov @ w)
    if port_var <= 0:
        return 0.0
    return -(port_ret - rf) / np.sqrt(port_var)


def _tangency_warmstart(mu_sub: np.ndarray, cov_sub: np.ndarray, rf: float, cap: float) -> np.ndarray:
    """
    Analytical unconstrained tangency weights used as SLSQP warm-start.

    w* = Σ⁻¹(μ - rf·1) / 1ᵀΣ⁻¹(μ - rf·1)

    Clipped to [0, cap] and normalised.  Starting closer to the optimum
    typically halves the number of SLSQP iterations needed.
    """
    excess = mu_sub - rf
    try:
        w = np.linalg.solve(cov_sub, excess)
    except np.linalg.LinAlgError:
        return np.ones(len(mu_sub)) / len(mu_sub)
    w = np.clip(w, 0, cap)
    total = w.sum()
    return w / total if total > 0 else np.ones(len(mu_sub)) / len(mu_sub)


def optimize_max_sharpe(
    mu: pd.Series,
    cov: np.ndarray,
    top_n: int = cfg.TOP_N,
    max_weight: float = cfg.MAX_WEIGHT,
    rf: float = cfg.RISK_FREE_RATE,
) -> pd.Series:
    """
    Solve the maximum-Sharpe-ratio (tangency) portfolio via SLSQP.

    Speed optimisations
    -------------------
    * Universe pre-filtered to top_n by μ (reduces n from 11 → 5).
    * Analytical tangency solution used as warm-start → fewer SLSQP iterations.
    * Looser ftol (1e-7) acceptable for portfolio weights.
    """
    tickers = mu.index.tolist()

    # Restrict to top_n by predicted return
    ranked  = mu.sort_values(ascending=False)
    eligible = ranked.index[:top_n].tolist()
    idx      = [tickers.index(t) for t in eligible]

    mu_sub  = mu[eligible].values.astype(float)
    cov_sub = cov[np.ix_(idx, idx)]
    n_sub   = len(eligible)

    w0          = _tangency_warmstart(mu_sub, cov_sub, rf, max_weight)
    bounds      = [(0.0, max_weight)] * n_sub
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]

    try:
        result = minimize(
            _neg_sharpe,
            w0,
            args=(mu_sub, cov_sub, rf),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 300},
        )
        w_opt = np.clip(result.x, 0.0, max_weight)
        total = w_opt.sum()
        if total > 1e-8:
            w_opt /= total
            weights = pd.Series(w_opt, index=eligible)
        else:
            weights = _score_weights(mu[eligible], max_weight)
    except Exception as exc:
        logger.debug(f"Optimiser error: {exc}")
        weights = _score_weights(mu[eligible], max_weight)

    full = pd.Series(0.0, index=tickers)
    full[eligible] = weights
    return full.sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Score-proportional weights (original method / fallback)
# ---------------------------------------------------------------------------

def _clip_and_redistribute(weights: np.ndarray, cap: float, max_iter: int = 50) -> np.ndarray:
    """Iterative cap-and-redistribute."""
    w = weights.copy().astype(float)
    for _ in range(max_iter):
        capped   = w >= cap
        uncapped = ~capped & (w > 0)
        excess   = (w[capped] - cap).sum()
        if excess < 1e-9:
            break
        w[capped] = cap
        uncapped_sum = w[uncapped].sum()
        if uncapped_sum <= 0:
            w = np.full(len(w), cap)
            w /= w.sum()
            break
        w[uncapped] += excess * (w[uncapped] / uncapped_sum)
    total = w.sum()
    if total > 0:
        w /= total
    return w


def _score_weights(scores: pd.Series, max_weight: float) -> pd.Series:
    """Proportional weights from positive predicted scores."""
    raw = scores.clip(lower=0)
    if raw.sum() <= 1e-10:
        raw = pd.Series(1.0, index=scores.index)
    normed = raw / raw.sum()
    w_arr = _clip_and_redistribute(normed.values, cap=max_weight)
    return pd.Series(w_arr, index=scores.index)


def compute_weights(
    scores: pd.Series,
    top_n: int = cfg.TOP_N,
    max_weight: float = cfg.MAX_WEIGHT,
    min_weight: float = cfg.MIN_WEIGHT,
    cov: Optional[np.ndarray] = None,
    method: str = cfg.PORTFOLIO_METHOD,
) -> pd.Series:
    """
    Unified weight computation entry point.

    Parameters
    ----------
    scores     : predicted score per ETF (used for ranking and as μ proxy)
    top_n      : ETFs to consider
    max_weight : per-ETF cap
    min_weight : floor (0 = long-only)
    cov        : annualised covariance matrix (required for 'max_sharpe')
    method     : 'max_sharpe' | 'score'

    Returns
    -------
    pd.Series {ticker: weight} for ALL input tickers (non-held = 0).
    """
    scores = scores.dropna()
    if len(scores) == 0:
        return pd.Series(dtype=float)

    # Always restrict to top_n by score before optimising
    top_n   = min(top_n, len(scores))
    ranked  = scores.sort_values(ascending=False).iloc[:top_n]

    if method == "max_sharpe" and cov is not None:
        # Use predicted score as μ proxy (annualised: score × sqrt(252))
        # The return model predicts 21-day return; scale to annual.
        mu_annual = ranked * np.sqrt(cfg.TRADING_DAYS_PER_YEAR / cfg.FORWARD_DAYS)
        weights = optimize_max_sharpe(mu_annual, cov, top_n, max_weight)
        weights = weights[weights > 0]
    else:
        weights = _score_weights(ranked, max_weight)

    if min_weight > 0:
        weights = weights[weights >= min_weight]
        if weights.sum() > 0:
            weights /= weights.sum()

    # Expand back to full universe (non-selected = 0)
    all_tickers = scores.index
    full = pd.Series(0.0, index=all_tickers)
    for t, w in weights.items():
        if t in full.index:
            full[t] = w

    assert abs(full.sum() - 1.0) < 1e-5, f"Weights sum {full.sum():.6f} != 1"
    return full.sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Pre-compute rolling covariance store (for backtest)
# ---------------------------------------------------------------------------

def precompute_rolling_cov(
    etf_returns: pd.DataFrame,
    window: int = cfg.COV_WINDOW,
) -> dict[pd.Timestamp, np.ndarray]:
    """
    Build a {date: annualised_cov_matrix} store using vectorised pandas rolling.

    Strategy
    --------
    * Compute the full rolling sample covariance in one pandas call (fast).
    * Apply analytical Ledoit-Wolf shrinkage per-date using the oracle formula
      (no sklearn loop needed — ~10× faster than fitting LedoitWolf per date).

    Ledoit-Wolf oracle shrinkage (constant-correlation target):
        cov_shrunk = (1-δ)*S + δ*μ̄*I
    where S is the sample cov, μ̄ = trace(S)/p is the mean eigenvalue,
    and δ is chosen to minimise mean-squared estimation error.
    """
    logger.info(
        f"Pre-computing rolling covariance "
        f"(window={window}, shrinkage={cfg.SHRINKAGE})…"
    )
    rets = etf_returns.ffill().dropna(how="all")
    p     = rets.shape[1]
    dates = rets.index
    vals  = rets.values.astype(np.float64)   # (T, p)
    T     = len(vals)

    # ── Fully vectorised batch covariance ─────────────────────────────────
    # Build sliding windows using stride tricks: shape (T-w+1, w, p)
    w = window
    windows = np.lib.stride_tricks.sliding_window_view(vals, (w, p)).reshape(-1, w, p)
    # windows[i] = vals[i : i+w]   →  final date index = i + w - 1

    # Batch de-mean
    means   = windows.mean(axis=1, keepdims=True)      # (B, 1, p)
    dX      = windows - means                           # (B, w, p)

    # Batch sample covariance: S_b = dX_b.T @ dX_b / (w-1),  shape (B, p, p)
    S_batch = np.einsum("bni,bnj->bij", dX, dX) / (w - 1)

    if cfg.SHRINKAGE:
        # Oracle Approximating Shrinkage (OAS, Chen et al. 2010).
        # Target: scaled identity  T = (tr(S)/p)·I
        # Optimal ρ formula (closed-form, no outer-product loop needed):
        #   ρ = [(1 - 2/p)·tr(S²) + tr(S)²] / [(n+1 - 2/p)·(tr(S²) - tr(S)²/p)]
        #   S* = (1-ρ)·S + ρ·(tr(S)/p)·I

        n_obs    = float(w)
        trace_S  = np.einsum("bii->b", S_batch)               # (B,) tr(S)
        # tr(S²) = sum of squared entries of S (since tr(S²) = ||S||²_F for sym S)
        trace_S2 = np.einsum("bij,bji->b", S_batch, S_batch)  # (B,) tr(S²)
        trace_S_sq = trace_S ** 2                             # (B,) tr(S)²

        numer_oas = (1 - 2.0 / p) * trace_S2 + trace_S_sq
        denom_oas = (n_obs + 1 - 2.0 / p) * (trace_S2 - trace_S_sq / p)

        rho = np.where(denom_oas > 0,
                       np.clip(numer_oas / denom_oas, 0.0, 1.0),
                       0.0)                                    # (B,)

        mu_oas   = (trace_S / p)[:, None, None] * np.eye(p)[None]  # (B,p,p)
        S_batch  = (1 - rho)[:, None, None] * S_batch + rho[:, None, None] * mu_oas

    # Annualise
    S_batch *= cfg.TRADING_DAYS_PER_YEAR

    # Map batch index → date  (batch index b → date index b + w - 1)
    cov_store: dict[pd.Timestamp, np.ndarray] = {}
    for b in range(len(S_batch)):
        date_idx = b + w - 1
        if date_idx < T:
            cov_store[dates[date_idx]] = S_batch[b]

    logger.info(f"Covariance store built: {len(cov_store)} matrices")
    return cov_store


# ---------------------------------------------------------------------------
# Allocation table builder
# ---------------------------------------------------------------------------

def build_allocation(
    predictions: pd.DataFrame,
    rebalance_date: pd.Timestamp | None = None,
    cov: Optional[np.ndarray] = None,
    save: bool = True,
) -> pd.DataFrame:
    """Build and save the latest sector allocation."""
    pred = predictions.copy()
    scores = pred.set_index("ETF")["predicted_score"]
    weights = compute_weights(scores, cfg.TOP_N, cfg.MAX_WEIGHT, cfg.MIN_WEIGHT, cov=cov)

    pred["weight"]     = pred["ETF"].map(weights).fillna(0.0)
    pred["rank"]       = pred["predicted_score"].rank(ascending=False).astype(int)
    pred["pct_weight"] = (pred["weight"] * 100).round(2)
    pred.sort_values("rank", inplace=True)
    pred.reset_index(drop=True, inplace=True)

    if rebalance_date is not None:
        pred.insert(0, "rebalance_date", rebalance_date.date())

    logger.info("Latest allocation:")
    for _, row in pred[pred["weight"] > 0].iterrows():
        logger.info(
            f"  {row['ETF']:5s} ({row['Sector']:<25s})  "
            f"score={row['predicted_score']:+.4f}  weight={row['pct_weight']:.1f}%"
        )

    if save:
        cfg.DATA_ARTIFACTS.mkdir(parents=True, exist_ok=True)
        pred.to_csv(cfg.ALLOCATION_CSV, index=False)
        logger.info(f"Allocation saved -> {cfg.ALLOCATION_CSV}")

    return pred


# ---------------------------------------------------------------------------
# Rebalance schedule helper
# ---------------------------------------------------------------------------

def get_rebalance_dates(date_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Snap business-period-end dates to actual trading days."""
    all_bme = pd.date_range(
        start=date_index.min(),
        end=date_index.max(),
        freq=cfg.REBALANCE_FREQ,
    )
    rebalance: list[pd.Timestamp] = []
    for bme in all_bme:
        available = date_index[date_index <= bme]
        if len(available) > 0:
            rebalance.append(available[-1])
    return pd.DatetimeIndex(sorted(set(rebalance)))
