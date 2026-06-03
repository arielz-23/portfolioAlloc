"""
main.py – End-to-end pipeline runner.

Usage:
    python main.py                  # full pipeline
    python main.py --skip-download  # skip yfinance/FRED (use cached raw data)
    python main.py --skip-train     # skip training (use cached models)

Steps
-----
1. Download / load data
2. Build feature panel
3. Train XGBoost models
4. Run backtest
5. Generate latest allocation
6. Write evaluation reports
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import config as cfg

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(level=cfg.LOG_LEVEL, format=cfg.LOG_FMT)
logger = logging.getLogger("main")


def banner(msg: str) -> None:
    line = "─" * 60
    logger.info(line)
    logger.info(f"  {msg}")
    logger.info(line)


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step_download(skip: bool = False):
    if skip and (cfg.DATA_RAW / "etf_prices.parquet").exists():
        logger.info("Skipping download – using cached raw data")
        import pandas as pd
        etf_prices = pd.read_parquet(cfg.DATA_RAW / "etf_prices.parquet")
        spy_path = cfg.DATA_RAW / "spy.parquet"
        spy = pd.read_parquet(spy_path)["SPY"] if spy_path.exists() else None
        macro_path = cfg.DATA_RAW / "macro.parquet"
        macro = pd.read_parquet(macro_path) if macro_path.exists() else None
        return etf_prices, spy, macro

    banner("STEP 1 – Download Data")
    from download_data import run as download_run
    return download_run(save=True)


def step_build_dataset(etf_prices, spy, macro, skip: bool = False):
    if skip and cfg.FEATURES_CSV.exists():
        logger.info("Skipping dataset build – using cached features.csv")
        import pandas as pd
        return pd.read_csv(cfg.FEATURES_CSV, parse_dates=["Date"])

    banner("STEP 2 – Build Feature Panel")
    from build_dataset import build_panel
    return build_panel(etf_prices, spy=spy, macro=macro, save=True)


def step_train(panel, skip: bool = False):
    if skip and cfg.MODEL_PATH.exists():
        logger.info("Skipping training – loading cached models")
        from train_model import load_models
        return load_models()

    banner("STEP 3 – Train XGBoost Models")
    from train_model import train
    from build_dataset import get_feature_columns
    feature_cols = get_feature_columns(panel)
    return train(panel=panel, feature_cols=feature_cols, save=True)


def step_backtest(etf_prices, panel, models, spy):
    banner("STEP 4 – Run Backtest")
    from build_dataset import get_feature_columns
    from backtest import run as backtest_run
    feature_cols = get_feature_columns(panel)
    results, weight_history = backtest_run(
        etf_prices=etf_prices,
        panel=panel,
        models=models,
        feature_cols=feature_cols,
        spy_prices=spy,
        save=True,
    )
    return results, weight_history


def step_allocation(panel, models):
    banner("STEP 5 – Latest Allocation")
    import pandas as pd
    from build_dataset import get_feature_columns
    from train_model import predict_scores
    from portfolio import build_allocation

    feature_cols = get_feature_columns(panel)
    latest_date = panel["Date"].max()
    latest_panel = panel[panel["Date"] == latest_date].copy()

    if latest_panel.empty:
        logger.warning("No data for latest date – skipping allocation step")
        return None

    predictions = predict_scores(latest_panel, models, feature_cols)
    allocation = build_allocation(predictions, rebalance_date=latest_date, save=True)
    return allocation


def step_evaluate(results, panel, models):
    banner("STEP 6 – Evaluation & Reports")
    from build_dataset import get_feature_columns
    from evaluation import run as eval_run
    feature_cols = get_feature_columns(panel)
    return eval_run(results=results, panel=panel, models=models, feature_cols=feature_cols)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ETF Sector Rotation – Full Pipeline")
    p.add_argument("--skip-download", action="store_true",
                   help="Skip download step; use cached raw data")
    p.add_argument("--skip-build",    action="store_true",
                   help="Skip dataset build; use cached features.csv")
    p.add_argument("--skip-train",    action="store_true",
                   help="Skip training; use cached model files")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    banner("ETF Sector Rotation – University ML Project")
    logger.info(f"Working directory : {cfg.ROOT}")
    logger.info(f"FRED API key set  : {'YES' if cfg.FRED_API_KEY else 'NO (macro features disabled)'}")
    logger.info(f"Two-model mode    : {cfg.TWO_MODEL}")

    # 1. Data
    etf_prices, spy, macro = step_download(skip=args.skip_download)

    # 2. Features
    panel = step_build_dataset(
        etf_prices, spy, macro, skip=args.skip_build
    )

    # 3. Models
    models = step_train(panel, skip=args.skip_train)

    # 4. Backtest
    results, weight_history = step_backtest(etf_prices, panel, models, spy)

    # 5. Allocation
    allocation = step_allocation(panel, models)

    # 6. Evaluation
    metrics = step_evaluate(results, panel, models)

    # ── Summary ───────────────────────────────────────────────────────
    elapsed = time.time() - t0
    banner(f"Pipeline complete in {elapsed:.1f}s")
    logger.info("\nNext steps:")
    logger.info("  • View reports : open reports/evaluation_report.md")
    logger.info("  • Launch dashboard : streamlit run src/app.py")

    if not metrics.empty and "model" in metrics.columns:
        logger.info("\nModel performance (test period):")
        for metric in ["CAGR", "Sharpe Ratio", "Max Drawdown"]:
            if metric in metrics.index:
                logger.info(f"  {metric:25s}: {metrics.loc[metric, 'model']}")


if __name__ == "__main__":
    main()
