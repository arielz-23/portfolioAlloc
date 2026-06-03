"""
train_model.py – XGBoost model training for the ETF sector-rotation model.

Two-model approach (cfg.TWO_MODEL = True):
  • Return model  → predicts future_21d_return
  • Volatility model → predicts future_21d_volatility
  predicted_score = predicted_return / predicted_volatility

Single-model fallback (cfg.TWO_MODEL = False):
  • One XGBoost model predicts target_score directly.

Both models are trained with time-series-aware cross-validation (no shuffle).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pickle
from xgboost import XGBRegressor

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from build_dataset import get_feature_columns, time_series_split

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _clean(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """Drop rows where target or all features are NaN."""
    mask = df[target_col].notna()
    df_clean = df[mask].copy()
    X = df_clean[feature_cols].copy()
    y = df_clean[target_col].copy()

    # Fill remaining NaN in features with column median (training median)
    X.fillna(X.median(), inplace=True)
    return X, y


def _train_single(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    label: str,
    params: dict | None = None,
) -> XGBRegressor:
    """Fit one XGBRegressor with early stopping on the validation set."""
    p = {**cfg.XGB_PARAMS, **(params or {})}
    early = p.pop("early_stopping_rounds", 50)

    model = XGBRegressor(**p, early_stopping_rounds=early)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    val_pred = model.predict(X_val)
    valid_mask = ~np.isnan(y_val.values) & ~np.isnan(val_pred)
    corr = float(np.corrcoef(y_val.values[valid_mask], val_pred[valid_mask])[0, 1]) if valid_mask.sum() > 2 else 0.0
    best_iter = model.best_iteration if hasattr(model, "best_iteration") and model.best_iteration else p.get("n_estimators", 500)
    logger.info(f"  [{label}] val IC={corr:.4f}  best_iter={best_iter}")
    return model


# ---------------------------------------------------------------------------
# Public training API
# ---------------------------------------------------------------------------

def train(
    panel: pd.DataFrame | None = None,
    feature_cols: list[str] | None = None,
    save: bool = True,
) -> dict[str, XGBRegressor]:
    """
    Train return, volatility (and optionally combined-score) XGBoost models.

    Parameters
    ----------
    panel        : full feature panel; loaded from disk if None
    feature_cols : explicit list of feature columns (derived if None)
    save         : persist models to data/artifacts/

    Returns
    -------
    dict with keys 'return', 'volatility' (and 'score' if TWO_MODEL=False).
    """
    if panel is None:
        logger.info(f"Loading panel from {cfg.FEATURES_CSV}")
        panel = pd.read_csv(cfg.FEATURES_CSV, parse_dates=["Date"])

    if feature_cols is None:
        feature_cols = get_feature_columns(panel)

    logger.info(f"Feature count: {len(feature_cols)}")

    train_df, val_df, test_df = time_series_split(panel)
    models: dict[str, XGBRegressor] = {}

    if cfg.TWO_MODEL:
        # --- Return model ---
        logger.info("Training return model…")
        X_tr, y_tr = _clean(train_df, "future_21d_return", feature_cols)
        X_v,  y_v  = _clean(val_df,   "future_21d_return", feature_cols)
        # Align columns (val may have different NaN fill)
        X_v = X_v.reindex(columns=X_tr.columns).fillna(X_tr.median())
        ret_model = _train_single(X_tr, y_tr, X_v, y_v, "return_model")
        models["return"] = ret_model

        # --- Volatility model ---
        logger.info("Training volatility model…")
        X_tr_v, y_tr_v = _clean(train_df, "future_21d_volatility", feature_cols)
        X_v_v,  y_v_v  = _clean(val_df,   "future_21d_volatility", feature_cols)
        X_v_v = X_v_v.reindex(columns=X_tr_v.columns).fillna(X_tr_v.median())
        vol_model = _train_single(X_tr_v, y_tr_v, X_v_v, y_v_v, "vol_model")
        models["volatility"] = vol_model

    else:
        # --- Single score model ---
        logger.info("Training combined score model…")
        X_tr, y_tr = _clean(train_df, "target_score", feature_cols)
        X_v,  y_v  = _clean(val_df,   "target_score", feature_cols)
        X_v = X_v.reindex(columns=X_tr.columns).fillna(X_tr.median())
        score_model = _train_single(X_tr, y_tr, X_v, y_v, "score_model")
        models["score"] = score_model

    # --- Evaluate on held-out test set ---
    logger.info("Test-set evaluation:")
    _evaluate_on_test(models, test_df, feature_cols)

    if save:
        _save_models(models)

    return models


def _evaluate_on_test(
    models: dict[str, XGBRegressor],
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> None:
    """Log information coefficient (rank correlation) on the test set."""
    try:
        from scipy.stats import spearmanr
    except ImportError:
        logger.warning("scipy not installed – skipping Spearman IC computation")
        return

    if cfg.TWO_MODEL:
        for target, key in [("future_21d_return", "return"), ("future_21d_volatility", "volatility")]:
            X_t, y_t = _clean(test_df, target, feature_cols)
            X_t = X_t.reindex(columns=feature_cols).fillna(X_t.median())
            preds = models[key].predict(X_t)
            ic, _ = spearmanr(y_t, preds)
            mse = float(np.mean((y_t.values - preds) ** 2))
            logger.info(f"  [{key}] test Spearman IC={ic:.4f}  MSE={mse:.6f}")
    else:
        X_t, y_t = _clean(test_df, "target_score", feature_cols)
        X_t = X_t.reindex(columns=feature_cols).fillna(X_t.median())
        preds = models["score"].predict(X_t)
        ic, _ = spearmanr(y_t, preds)
        mse = float(np.mean((y_t.values - preds) ** 2))
        logger.info(f"  [score] test Spearman IC={ic:.4f}  MSE={mse:.6f}")


def _save_models(models: dict[str, XGBRegressor]) -> None:
    """Persist models to disk as pickle files."""
    cfg.DATA_ARTIFACTS.mkdir(parents=True, exist_ok=True)

    path_map = {
        "return":     cfg.RETURN_MODEL_PATH,
        "volatility": cfg.VOL_MODEL_PATH,
        "score":      cfg.MODEL_PATH,
    }
    for key, model in models.items():
        path = path_map.get(key, cfg.DATA_ARTIFACTS / f"xgb_{key}_model.pkl")
        with open(path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"Saved {key} model → {path}")


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def predict_scores(
    panel_latest: pd.DataFrame,
    models: dict[str, XGBRegressor],
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Generate predicted scores for a slice of the panel (e.g., latest date).

    Parameters
    ----------
    panel_latest : rows for the prediction date (one row per ETF)
    models       : dict returned by train()
    feature_cols : same feature columns used during training

    Returns
    -------
    DataFrame with columns ['ETF', 'Sector', 'predicted_return',
                            'predicted_volatility', 'predicted_score']
    """
    X = panel_latest[feature_cols].copy()
    X = X.fillna(X.median())

    result = panel_latest[["ETF", "Sector"]].copy()

    if cfg.TWO_MODEL and "return" in models and "volatility" in models:
        result["predicted_return"]     = models["return"].predict(X)
        result["predicted_volatility"] = np.clip(models["volatility"].predict(X), 1e-6, None)
        result["predicted_score"]      = (
            result["predicted_return"] / result["predicted_volatility"]
        )
    elif "score" in models:
        result["predicted_score"] = models["score"].predict(X)
        result["predicted_return"]     = np.nan
        result["predicted_volatility"] = np.nan
    else:
        raise ValueError("models dict must contain 'return'+'volatility' or 'score'")

    result.reset_index(drop=True, inplace=True)
    return result


def batch_predict_all_dates(
    panel: pd.DataFrame,
    models: dict[str, XGBRegressor],
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Run inference on the entire panel in one vectorised pass.

    This is the fast path for daily rebalancing: instead of calling predict
    once per trading day inside the backtest loop, we score every (date, ETF)
    row upfront and return a wide pivot table.

    Parameters
    ----------
    panel        : full feature panel (all dates and ETFs)
    models       : dict from train() or load_models()
    feature_cols : feature columns used during training

    Returns
    -------
    pd.DataFrame  – index=Date, columns=ETF tickers, values=predicted_score
    """
    X = panel[feature_cols].copy()
    # Fill NaNs with column median computed across the whole panel
    X.fillna(X.median(), inplace=True)

    scores = panel[["Date", "ETF"]].copy().reset_index(drop=True)

    if cfg.TWO_MODEL and "return" in models and "volatility" in models:
        pred_ret = models["return"].predict(X)
        pred_vol = np.clip(models["volatility"].predict(X), 1e-6, None)
        scores["predicted_return"] = pred_ret
        scores["predicted_score"]  = pred_ret / pred_vol
    elif "score" in models:
        scores["predicted_score"]  = models["score"].predict(X)
        scores["predicted_return"] = scores["predicted_score"]   # proxy
    else:
        raise ValueError("models dict must contain 'return'+'volatility' or 'score'")

    # Pivot to wide tables: rows=Date, cols=ETF
    score_wide  = scores.pivot(index="Date", columns="ETF", values="predicted_score")
    return_wide = scores.pivot(index="Date", columns="ETF", values="predicted_return")
    for w in (score_wide, return_wide):
        w.sort_index(inplace=True)

    return {"score": score_wide, "return": return_wide}


def load_models() -> dict[str, XGBRegressor]:
    """Load persisted models from disk."""
    models: dict[str, XGBRegressor] = {}
    for key, path in [("return", cfg.RETURN_MODEL_PATH),
                      ("volatility", cfg.VOL_MODEL_PATH),
                      ("score", cfg.MODEL_PATH)]:
        if path.exists():
            with open(path, "rb") as f:
                models[key] = pickle.load(f)
    if not models:
        raise FileNotFoundError("No trained models found. Run main.py first.")
    return models


if __name__ == "__main__":
    logging.basicConfig(level=cfg.LOG_LEVEL, format=cfg.LOG_FMT)
    train(save=True)
