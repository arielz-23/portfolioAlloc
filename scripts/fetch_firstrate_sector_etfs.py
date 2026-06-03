#!/usr/bin/env python3
"""Download and normalize FirstRate 1-day OHLCV data for sector ETFs."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import zipfile
from datetime import date
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen


API_ENDPOINT = "https://firstratedata.com/api/data_file"
ARCHIVE_NAME = "etf_X_1day_UNADJUSTED_full.zip"
COMBINED_NAME = "sector_etfs_1day_unadjusted.csv"
DEFAULT_START_DATE = "2010-01-01"

ETF_SECTORS = {
    "XLK": "Technology",
    "XLV": "Healthcare",
    "XLF": "Financials",
    "XLI": "Industrials",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLE": "Energy",
    "XLU": "Utilities",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
}

FIELDNAMES = ["date", "ticker", "sector", "open", "high", "low", "close", "volume"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download FirstRate's full X-range ETF 1-day UNADJUSTED archive, "
            "extract sector ETFs, and write normalized CSV outputs."
        )
    )
    parser.add_argument(
        "--userid",
        default=os.environ.get("FIRSTRATE_USERID"),
        help="FirstRate userid. Defaults to FIRSTRATE_USERID.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download the archive again even if the cached zip already exists.",
    )
    parser.add_argument(
        "--raw-dir",
        default="data/firstrate/raw",
        type=Path,
        help="Directory for the downloaded FirstRate zip archive.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/firstrate/sector_etfs",
        type=Path,
        help="Directory for normalized output CSV files.",
    )
    parser.add_argument(
        "--start-date",
        default=DEFAULT_START_DATE,
        help="Inclusive output start date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end-date",
        help="Inclusive output end date in YYYY-MM-DD format. Defaults to latest available.",
    )
    return parser.parse_args()


def parse_iso_date(value: str, label: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be in YYYY-MM-DD format: {value}") from exc


def download_archive(userid: str, archive_path: Path, force: bool) -> None:
    if archive_path.exists() and not force:
        print(f"Reusing cached archive: {archive_path}")
        return

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    params = {
        "type": "etf",
        "period": "full",
        "ticker_range": "X",
        "timeframe": "1day",
        "adjustment": "UNADJUSTED",
        "userid": userid,
    }
    url = f"{API_ENDPOINT}?{urlencode(params)}"

    print(f"Downloading FirstRate archive to {archive_path}")
    try:
        with urlopen(url, timeout=120) as response:
            content_type = response.headers.get("content-type", "")
            data = response.read()
    except HTTPError as exc:
        if exc.code == 403:
            raise RuntimeError(
                "FirstRate returned HTTP 403 for the data_file archive request. "
                "The userid may still be valid for metadata endpoints, but archive "
                "downloads require an active FirstRate bundle/API subscription with "
                "ETF data access. Confirm the Customer Download Page/API docs for "
                "this userid include ETF archive downloads."
            ) from exc
        raise RuntimeError(f"FirstRate request failed with HTTP {exc.code}: {exc.reason}") from exc
    except URLError as exc:
        raise RuntimeError(f"FirstRate request failed: {exc.reason}") from exc

    if not data:
        raise RuntimeError("FirstRate returned an empty response.")

    archive_path.write_bytes(data)

    if not zipfile.is_zipfile(archive_path):
        preview = data[:500].decode("utf-8", errors="replace").strip()
        archive_path.unlink(missing_ok=True)
        raise RuntimeError(
            "FirstRate response was not a zip archive. "
            f"Content-Type: {content_type or 'unknown'}. "
            f"Response preview: {preview}"
        )


def ticker_from_member(member_name: str) -> str | None:
    stem = Path(member_name).stem.upper()
    tokens = stem.replace("-", "_").split("_")
    for ticker in ETF_SECTORS:
        if ticker in tokens or stem == ticker or stem.startswith(f"{ticker}_"):
            return ticker
    return None


def parse_first_rate_row(row: list[str], ticker: str) -> dict[str, str]:
    if len(row) < 6:
        raise ValueError(f"{ticker}: expected at least 6 columns, got {len(row)}")

    timestamp, open_, high, low, close, volume = [value.strip() for value in row[:6]]
    date = timestamp.split(" ", 1)[0]
    return {
        "date": date,
        "ticker": ticker,
        "sector": ETF_SECTORS[ticker],
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }


def looks_like_header(row: list[str]) -> bool:
    if not row:
        return True
    first = row[0].strip().lower()
    return first in {"datetime", "date", "timestamp", "time"}


def read_target_rows(archive_path: Path) -> list[dict[str, str]]:
    found_tickers: set[str] = set()
    rows: list[dict[str, str]] = []

    with zipfile.ZipFile(archive_path) as archive:
        members = [member for member in archive.namelist() if not member.endswith("/")]
        for member in members:
            ticker = ticker_from_member(member)
            if ticker is None:
                continue

            found_tickers.add(ticker)
            with archive.open(member) as file:
                text = (line.decode("utf-8-sig", errors="replace") for line in file)
                reader = csv.reader(text)
                for row_number, row in enumerate(reader, start=1):
                    if looks_like_header(row):
                        continue
                    try:
                        rows.append(parse_first_rate_row(row, ticker))
                    except ValueError as exc:
                        raise ValueError(f"{member}:{row_number}: {exc}") from exc

    missing = sorted(set(ETF_SECTORS) - found_tickers)
    if missing:
        raise RuntimeError(
            "The archive did not contain all requested ETFs. "
            f"Missing: {', '.join(missing)}"
        )

    rows.sort(key=lambda item: (item["ticker"], item["date"]))
    return rows


def filter_rows_by_date(
    rows: list[dict[str, str]], start_date: date, end_date: date | None
) -> list[dict[str, str]]:
    filtered = []
    for row in rows:
        row_date = parse_iso_date(row["date"], "row date")
        if row_date < start_date:
            continue
        if end_date is not None and row_date > end_date:
            continue
        filtered.append(row)
    return filtered


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(rows: list[dict[str, str]], out_dir: Path) -> None:
    write_csv(out_dir / COMBINED_NAME, rows)

    by_ticker_dir = out_dir / "by_ticker"
    for ticker in sorted(ETF_SECTORS):
        ticker_rows = [row for row in rows if row["ticker"] == ticker]
        write_csv(by_ticker_dir / f"{ticker}_1day_unadjusted.csv", ticker_rows)


def main() -> int:
    args = parse_args()
    if not args.userid:
        print(
            "Missing FirstRate userid. Set FIRSTRATE_USERID or pass --userid.",
            file=sys.stderr,
        )
        return 2

    archive_path = args.raw_dir / ARCHIVE_NAME

    try:
        start_date = parse_iso_date(args.start_date, "start-date")
        end_date = parse_iso_date(args.end_date, "end-date") if args.end_date else None
        if end_date is not None and end_date < start_date:
            raise ValueError("end-date must be greater than or equal to start-date")

        download_archive(args.userid, archive_path, args.force)
        rows = read_target_rows(archive_path)
        rows = filter_rows_by_date(rows, start_date, end_date)
        write_outputs(rows, args.out_dir)
    except (RuntimeError, ValueError, zipfile.BadZipFile) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    tickers = ", ".join(sorted(ETF_SECTORS))
    date_range = f"{start_date.isoformat()} to {end_date.isoformat() if end_date else 'latest'}"
    print(f"Wrote {len(rows):,} rows for {len(ETF_SECTORS)} ETFs: {tickers}")
    print(f"Date range: {date_range}")
    print(f"Combined CSV: {args.out_dir / COMBINED_NAME}")
    print(f"Per-ticker CSVs: {args.out_dir / 'by_ticker'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
