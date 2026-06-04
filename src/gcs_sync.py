"""
gcs_sync.py – Upload / download pipeline artifacts to/from GCS.

Used by:
  • main.py  → upload_artifacts() after pipeline completes
  • api.py   → download_artifacts() on startup and /api/refresh

Required env vars:
  GCS_BUCKET  – bucket name (e.g. "verticalmain-portfolio-data")
  GCS_PREFIX  – optional subfolder prefix (default: "")

Authentication uses ADC (Application Default Credentials), which works
automatically on Cloud Run and Cloud Run Jobs via the attached service account.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

logger = logging.getLogger(__name__)

GCS_BUCKET = os.getenv("GCS_BUCKET", "")
GCS_PREFIX = os.getenv("GCS_PREFIX", "")     # e.g. "portfolio/" for multi-env sharing

# Directories that get synced in both directions
_SYNC_DIRS = [
    cfg.DATA_RAW,
    cfg.DATA_PROCESSED,
    cfg.DATA_ARTIFACTS,
    cfg.REPORTS,
]

_SKIP_EXTENSIONS = {".pyc", ".pyo", ".pyd"}
_SKIP_NAMES = {".gitkeep", "__pycache__"}


def _client():
    from google.cloud import storage
    return storage.Client()


def _blob_name(file_path: Path) -> str:
    rel = file_path.relative_to(cfg.ROOT).as_posix()  # always forward slashes
    return f"{GCS_PREFIX}{rel}" if GCS_PREFIX else rel


def upload_artifacts() -> int:
    """Upload all pipeline output files to GCS. Returns number of files uploaded."""
    if not GCS_BUCKET:
        logger.info("GCS_BUCKET not set — skipping upload")
        return 0

    client = _client()
    bucket = client.bucket(GCS_BUCKET)
    count = 0

    for dir_path in _SYNC_DIRS:
        if not dir_path.exists():
            continue
        for file_path in dir_path.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix in _SKIP_EXTENSIONS:
                continue
            if file_path.name in _SKIP_NAMES:
                continue

            blob_name = _blob_name(file_path)
            bucket.blob(blob_name).upload_from_filename(str(file_path))
            logger.info(f"  uploaded → gs://{GCS_BUCKET}/{blob_name}")
            count += 1

    logger.info(f"GCS upload complete: {count} files → gs://{GCS_BUCKET}/{GCS_PREFIX}")
    return count


def download_artifacts() -> int:
    """Download all pipeline artifacts from GCS to local paths. Returns file count."""
    if not GCS_BUCKET:
        logger.info("GCS_BUCKET not set — skipping download")
        return 0

    client = _client()
    bucket = client.bucket(GCS_BUCKET)
    prefix = GCS_PREFIX
    count = 0

    for blob in bucket.list_blobs(prefix=prefix):
        # Strip the prefix to get the relative path inside the repo
        rel = blob.name[len(prefix):].lstrip("/")
        if not rel:
            continue

        local_path = cfg.ROOT / rel
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_path))
        logger.info(f"  downloaded ← gs://{GCS_BUCKET}/{blob.name}")
        count += 1

    logger.info(f"GCS download complete: {count} files from gs://{GCS_BUCKET}/{prefix}")
    return count
