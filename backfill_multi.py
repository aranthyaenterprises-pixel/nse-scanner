"""
Multi-year backfill: downloads NSE bhavcopies from START_YEAR onward using
the OLD archive format (pre July-2024), storing one compressed file per
year in data/hist_YYYY.csv.gz (keeps each file far below GitHub's 100MB
limit). Resume-safe: years that already have a complete file are skipped.

The recent period (July 2024 onward) is already covered by the regular
backfill (data/history.csv.gz) in the new UDiFF format; the V2 backtest
loads and concatenates both automatically.

Run: python backfill_multi.py
"""

import io
import os
import time
import zipfile
from datetime import date, timedelta

import pandas as pd
import requests

from scanner import HEADERS, warm_session, fetch_bhavcopy

START_YEAR = 2014
OLD_FORMAT_END = date(2024, 7, 5)   # old archive URL reliable until ~here
DATA_DIR = "data"

MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
          "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


def old_url(d: date) -> str:
    mon = MONTHS[d.month - 1]
    return ("https://nsearchives.nseindia.com/content/historical/EQUITIES/"
            f"{d.year}/{mon}/cm{d.strftime('%d')}{mon}{d.year}bhav.csv.zip")


def fetch_old(d: date, session: requests.Session) -> pd.DataFrame | None:
    try:
        r = session.get(old_url(d), headers=HEADERS, timeout=30)
    except requests.RequestException:
        return None
    if r.status_code != 200 or len(r.content) < 1000:
        return None
    try:
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            df = pd.read_csv(zf.open(zf.namelist()[0]))
    except Exception:
        return None
    df = df.rename(columns={
        "SYMBOL": "symbol", "SERIES": "series", "OPEN": "open",
        "HIGH": "high", "LOW": "low", "CLOSE": "close",
        "PREVCLOSE": "prev_close", "TOTTRDQTY": "volume",
        "TOTTRDVAL": "turnover"})
    need = ["symbol", "series", "open", "high", "low", "close",
            "prev_close", "volume", "turnover"]
    if any(c not in df.columns for c in need):
        return None
    df = df[need]
    df = df[df["series"] == "EQ"].copy()
    df["delivery_pct"] = float("nan")
    df["date"] = d.isoformat()
    return df


def backfill_year(year: int, session: requests.Session) -> None:
    path = os.path.join(DATA_DIR, f"hist_{year}.csv.gz")
    if os.path.exists(path):
        print(f"{year}: already present, skipping")
        return
    frames = []
    d = date(year, 1, 1)
    end = min(date(year, 12, 31), date.today())
    while d <= end:
        if d.weekday() < 5:
            if d <= OLD_FORMAT_END:
                df = fetch_old(d, session)
            else:
                df = fetch_bhavcopy(d, session)   # new UDiFF format
            if df is not None:
                frames.append(df)
            time.sleep(0.35)
        d += timedelta(days=1)
    if frames:
        out = pd.concat(frames, ignore_index=True)
        os.makedirs(DATA_DIR, exist_ok=True)
        out.to_csv(path, index=False, compression="gzip")
        print(f"{year}: saved {out['date'].nunique()} sessions, "
              f"{len(out)} rows -> {path}")
    else:
        print(f"{year}: no data fetched")


def main() -> None:
    session = warm_session()
    for year in range(START_YEAR, date.today().year + 1):
        backfill_year(year, session)
    print("Multi-year backfill complete.")


if __name__ == "__main__":
    main()
