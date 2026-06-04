"""
api.py – FastAPI backend for the ETF Sector Rotation Dashboard.

Serves:
  • REST endpoints for all pre-computed artifacts (backtest, allocation, metrics, etc.)
  • Server-Sent Events (SSE) endpoints for the three Vertex AI Gemini agents
  • Static frontend files from /frontend

Launch with:
    uvicorn src.api:app --reload --port 8000
  or from project root:
    python -m uvicorn src.api:app --reload --port 8000
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

logger = logging.getLogger("api")

app = FastAPI(title="ETF Sector Rotation API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Startup: load all artifacts into memory
# ---------------------------------------------------------------------------

class AppState:
    backtest: pd.DataFrame | None = None
    allocation: pd.DataFrame | None = None
    panel: pd.DataFrame | None = None
    metrics: pd.DataFrame | None = None
    feat_imp: pd.DataFrame | None = None
    etf_prices: pd.DataFrame | None = None
    ready: bool = False


_state = AppState()


def _pull_from_gcs() -> bool:
    """Download latest artifacts from GCS. Returns True if successful."""
    if not os.getenv("GCS_BUCKET"):
        return False
    try:
        from gcs_sync import download_artifacts
        n = download_artifacts()
        logger.info(f"GCS pull complete: {n} files")
        return True
    except Exception as exc:
        logger.warning(f"GCS pull failed (using local data): {exc}")
        return False


def _reload_state() -> None:
    """(Re)load all artifacts from local disk into _state."""
    _state.backtest = None
    _state.allocation = None
    _state.panel = None
    _state.metrics = None
    _state.feat_imp = None
    _state.etf_prices = None
    _state.ready = False

    if cfg.BACKTEST_CSV.exists():
        _state.backtest = pd.read_csv(cfg.BACKTEST_CSV, index_col="Date", parse_dates=True)

    if cfg.ALLOCATION_CSV.exists():
        _state.allocation = pd.read_csv(cfg.ALLOCATION_CSV)

    if cfg.FEATURES_CSV.exists():
        _state.panel = pd.read_csv(cfg.FEATURES_CSV, parse_dates=["Date"])

    etf_path = cfg.DATA_RAW / "etf_prices.parquet"
    if etf_path.exists():
        _state.etf_prices = pd.read_parquet(etf_path)

    if _state.backtest is not None:
        try:
            from evaluation import compute_all_metrics
            _state.metrics = compute_all_metrics(_state.backtest)
        except Exception:
            pass

    if _state.panel is not None:
        try:
            from train_model import load_models
            from build_dataset import get_feature_columns
            from evaluation import feature_importance
            models = load_models()
            if models:
                fc = get_feature_columns(_state.panel)
                _state.feat_imp = feature_importance(models, fc)
        except Exception:
            pass

    _state.ready = (
        _state.backtest is not None and _state.allocation is not None
    )
    logger.info(f"State loaded — ready={_state.ready}")


@app.on_event("startup")
def _load_artifacts() -> None:
    _pull_from_gcs()   # no-op if GCS_BUCKET not set
    _reload_state()


# ---------------------------------------------------------------------------
# JSON serialisation helper (handles NaN, dates, numpy types)
# ---------------------------------------------------------------------------

def _clean(obj: Any) -> Any:
    if isinstance(obj, float):
        return None if (obj != obj) else obj  # NaN → null
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if v != v else v
    if isinstance(obj, np.ndarray):
        return [_clean(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(x) for x in obj]
    return obj


def _json(data: Any) -> str:
    return json.dumps(_clean(data))


# ---------------------------------------------------------------------------
# API routes – data
# ---------------------------------------------------------------------------

@app.get("/api/status")
def status():
    return {
        "ready": _state.ready,
        "has_panel": _state.panel is not None,
        "has_metrics": _state.metrics is not None,
        "has_feat_imp": _state.feat_imp is not None,
        "gcs_enabled": bool(os.getenv("GCS_BUCKET")),
    }


@app.get("/api/refresh")
def refresh():
    """Pull latest artifacts from GCS and reload into memory.
    Call this after the pipeline job finishes to update the live dashboard."""
    if not os.getenv("GCS_BUCKET"):
        return JSONResponse({"status": "skipped", "reason": "GCS_BUCKET not configured"})
    try:
        pulled = _pull_from_gcs()
        _reload_state()
        return {"status": "ok", "gcs_pulled": pulled, "dashboard_ready": _state.ready}
    except Exception as exc:
        return JSONResponse({"status": "error", "detail": str(exc)}, status_code=500)


@app.get("/api/metrics")
def get_metrics():
    if _state.metrics is None:
        return {"metrics": []}
    df = _state.metrics.reset_index().rename(columns={"index": "metric"})
    records = df.to_dict(orient="records")
    return {"metrics": _clean(records)}


@app.get("/api/allocation")
def get_allocation():
    if _state.allocation is None:
        return {"allocation": []}
    return {"allocation": _clean(_state.allocation.to_dict(orient="records"))}


@app.get("/api/backtest")
def get_backtest():
    if _state.backtest is None:
        return {}
    bt = _state.backtest.copy()
    bt.index = bt.index.strftime("%Y-%m-%d")
    cols = ["model", "equal_weight", "spy",
            "model_equity", "equal_weight_equity", "spy_equity"]
    cols = [c for c in cols if c in bt.columns]
    result = {"dates": bt.index.tolist()}
    for c in cols:
        result[c] = _clean(bt[c].tolist())
    return result


@app.get("/api/feature-importance")
def get_feature_importance():
    if _state.feat_imp is None or _state.feat_imp.empty:
        return {"features": []}
    df = _state.feat_imp[["mean_importance"]].head(20).reset_index()
    df.columns = ["feature", "importance"]
    return {"features": _clean(df.to_dict(orient="records"))}


@app.get("/api/etf-prices")
def get_etf_prices():
    if _state.etf_prices is None:
        return {}
    prices = _state.etf_prices.copy()
    prices.index = prices.index.strftime("%Y-%m-%d")
    result: dict = {"dates": prices.index.tolist()}
    for col in prices.columns:
        normed = prices[col] / prices[col].dropna().iloc[0]
        result[col] = _clean(normed.tolist())
    return result


# ---------------------------------------------------------------------------
# API routes – SSE agent streaming
# ---------------------------------------------------------------------------

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}


def _sse_stream(gen_fn, *args):
    """Wrap a generator function in SSE-formatted chunks."""
    def event_stream():
        try:
            for chunk in gen_fn(*args):
                payload = json.dumps({"text": chunk})
                yield f"data: {payload}\n\n"
        except Exception as exc:
            payload = json.dumps({"error": str(exc)})
            yield f"data: {payload}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=_SSE_HEADERS)


@app.get("/api/agents/market-analyst")
def agent_market_analyst():
    if not _state.ready or _state.panel is None:
        def _err():
            yield f"data: {json.dumps({'error': 'Data not ready. Run python main.py first.'})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(_err(), media_type="text/event-stream", headers=_SSE_HEADERS)

    from agents import market_analyst_stream
    return _sse_stream(market_analyst_stream, _state.panel, _state.allocation)


@app.get("/api/agents/portfolio-strategist")
def agent_portfolio_strategist():
    if not _state.ready:
        def _err():
            yield f"data: {json.dumps({'error': 'Data not ready. Run python main.py first.'})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(_err(), media_type="text/event-stream", headers=_SSE_HEADERS)

    metrics = _state.metrics if _state.metrics is not None else pd.DataFrame()
    feat_imp = _state.feat_imp if _state.feat_imp is not None else pd.DataFrame()
    from agents import portfolio_strategist_stream
    return _sse_stream(portfolio_strategist_stream, _state.allocation, metrics, feat_imp)


@app.get("/api/agents/risk-monitor")
def agent_risk_monitor():
    if not _state.ready:
        def _err():
            yield f"data: {json.dumps({'error': 'Data not ready. Run python main.py first.'})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(_err(), media_type="text/event-stream", headers=_SSE_HEADERS)

    from agents import risk_monitor_stream
    return _sse_stream(risk_monitor_stream, _state.backtest, _state.allocation)


# ---------------------------------------------------------------------------
# Serve frontend (must be last — catches all other paths)
# ---------------------------------------------------------------------------

_frontend = Path(__file__).resolve().parent.parent / "frontend"
if _frontend.exists():
    app.mount("/", StaticFiles(directory=str(_frontend), html=True), name="frontend")
