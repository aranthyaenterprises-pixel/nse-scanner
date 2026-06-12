"""
NSE Full-Market Momentum Scanner
---------------------------------
Downloads the official NSE bhavcopy (every listed stock) daily, maintains a
rolling 260-session price history, and screens for high-probability momentum
setups. Emails a ranked shortlist every evening.

Setups screened (momentum-only, by design):
  1. BREAKOUT      - close breaks above the prior 20-session high after a
                     tight consolidation, on >= 2x average volume
  2. 52W-HIGH ZONE - close within 3% of 52-week high, volume >= 2x average,
                     strong close (top 30% of day's range)
  3. ACCUMULATION  - volume >= 3x average AND delivery % >= 60% with price
                     up 3-9% (institutional footprint, not circuit junk)

Liquidity filters (applied before any setup logic):
  - EQ series only (no BE/BZ/T2T, no SME)
  - price >= 50
  - 20-day average turnover >= Rs. 5 crore

Run modes:
  python scanner.py backfill   -> first run: builds 260 sessions of history
  python scanner.py daily      -> daily run: fetch today, scan, email report
"""

import io
import os
import smtplib
import sys
import time
import zipfile
from datetime import date, datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import pandas as pd
import requests

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
HISTORY_FILE = "data/history.csv.gz"   # rolling price history (committed to repo)
HISTORY_DAYS = 260                     # ~1 trading year
MIN_PRICE = 50.0
MIN_AVG_TURNOVER = 5e7                 # Rs. 5 crore
VOL_AVG_WINDOW = 20

GMAIL_USER = os.environ.get("GMAIL_USER", "")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "")
RECIPIENT = os.environ.get("RECIPIENT", GMAIL_USER)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "*/*",
    "Referer": "https://www.nseindia.com/",
}

# ----------------------------------------------------------------------------
# Bhavcopy download (UDiFF format, in force since July 2024)
# ----------------------------------------------------------------------------

def bhavcopy_url(d: date) -> str:
    return (
        "https://nsearchives.nseindia.com/content/cm/"
        f"BhavCopy_NSE_CM_0_0_0_{d.strftime('%Y%m%d')}_F_0000.csv.zip"
    )


def fetch_bhavcopy(d: date, session: requests.Session) -> pd.DataFrame | None:
    """Return a normalized DataFrame for one trading day, or None (holiday)."""
    url = bhavcopy_url(d)
    try:
        r = session.get(url, headers=HEADERS, timeout=30)
    except requests.RequestException as e:
        print(f"  {d} network error: {e}")
        return None
    if r.status_code != 200 or len(r.content) < 1000:
        return None  # weekend / holiday / not yet published
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        name = zf.namelist()[0]
        df = pd.read_csv(zf.open(name))
    # UDiFF column names
    df = df.rename(columns={
        "TckrSymb": "symbol", "SctySrs": "series", "OpnPric": "open",
        "HghPric": "high", "LwPric": "low", "ClsPric": "close",
        "LastPric": "last", "PrvsClsgPric": "prev_close",
        "TtlTradgVol": "volume", "TtlTrfVal": "turnover",
        "DlvryQty": "delivery_qty", "DlvryPct": "delivery_pct",
        "TradDt": "trade_date",
    })
    keep = ["symbol", "series", "open", "high", "low", "close", "prev_close",
            "volume", "turnover"]
    for col in ("delivery_qty", "delivery_pct"):
        if col in df.columns:
            keep.append(col)
    df = df[[c for c in keep if c in df.columns]].copy()
    df = df[df["series"] == "EQ"]
    df = df[~df["symbol"].apply(looks_like_etf)]
    df["date"] = d.isoformat()
    if "delivery_pct" not in df.columns:
        df["delivery_pct"] = float("nan")
    # merge security-wise delivery data (separate NSE file)
    dlv = fetch_delivery(d, session)
    if dlv is not None and not dlv.empty:
        df = df.drop(columns=["delivery_pct"]).merge(dlv, on="symbol", how="left")
    return df


def fetch_delivery(d: date, session: requests.Session) -> pd.DataFrame | None:
    """NSE security-wise delivery (MTO) file: symbol -> delivery %."""
    url = ("https://nsearchives.nseindia.com/archives/equities/mto/"
           f"MTO_{d.strftime('%d%m%Y')}.DAT")
    try:
        r = session.get(url, headers=HEADERS, timeout=30)
    except requests.RequestException:
        return None
    if r.status_code != 200 or len(r.content) < 200:
        return None
    rows = []
    for line in r.text.splitlines():
        parts = line.split(",")
        # data records start with "20": rec,srno,symbol,series,traded,delivered,pct
        if len(parts) >= 7 and parts[0].strip() == "20" and parts[3].strip() == "EQ":
            try:
                rows.append({"symbol": parts[2].strip(),
                             "delivery_pct": float(parts[6])})
            except ValueError:
                continue
    return pd.DataFrame(rows) if rows else None


# crude ETF/non-stock symbol filter (ETFs often trade in EQ series)
ETF_KEYWORDS = ("ETF", "BEES", "BETA", "NIFTY", "SENSEX", "GOLD", "SILVER",
                "LIQUID", "GILT", "SDL", "GSEC", "MAFANG", "MON100", "MOM50",
                "ALPHA", "QUAL", "VALUE", "LOWVOL", "MOMENTUM", "IT", "PSUBNK",
                "PVTBAN", "HDFCM", "ICICIM", "ABSLN", "UTIN", "KOTAKN")


def looks_like_etf(symbol: str) -> bool:
    s = symbol.upper()
    return any(k in s for k in ("ETF", "BEES", "BETA", "IETF", "NETF")) or \
        s.endswith(("NIFTY", "GOLD", "SILVER", "LIQUID"))


def warm_session() -> requests.Session:
    s = requests.Session()
    try:
        s.get("https://www.nseindia.com", headers=HEADERS, timeout=20)
    except requests.RequestException:
        pass
    return s


# ----------------------------------------------------------------------------
# History management
# ----------------------------------------------------------------------------

def load_history() -> pd.DataFrame:
    if os.path.exists(HISTORY_FILE):
        return pd.read_csv(HISTORY_FILE, compression="gzip")
    return pd.DataFrame()


def save_history(df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    # keep only the most recent HISTORY_DAYS sessions
    sessions = sorted(df["date"].unique())[-HISTORY_DAYS:]
    df = df[df["date"].isin(sessions)]
    df.to_csv(HISTORY_FILE, index=False, compression="gzip")
    print(f"History saved: {len(sessions)} sessions, {len(df)} rows")


def backfill() -> None:
    session = warm_session()
    history = load_history()
    have = set(history["date"].unique()) if not history.empty else set()
    frames = [history] if not history.empty else []
    d = date.today()
    fetched = 0
    while fetched < HISTORY_DAYS and d > date.today() - timedelta(days=420):
        if d.weekday() < 5 and d.isoformat() not in have:
            df = fetch_bhavcopy(d, session)
            if df is not None:
                frames.append(df)
                fetched += 1
                print(f"  {d}: {len(df)} rows ({fetched}/{HISTORY_DAYS})")
            time.sleep(0.6)  # be polite to NSE servers
        d -= timedelta(days=1)
    if frames:
        save_history(pd.concat(frames, ignore_index=True))


def fetch_today() -> bool:
    session = warm_session()
    history = load_history()
    d = date.today()
    # walk back to the most recent trading day not already in history
    for _ in range(7):
        if d.weekday() < 5:
            if not history.empty and d.isoformat() in set(history["date"].unique()):
                print(f"{d} already in history; nothing to fetch.")
                return False
            df = fetch_bhavcopy(d, session)
            if df is not None:
                save_history(pd.concat([history, df], ignore_index=True))
                print(f"Fetched {d}: {len(df)} rows")
                return True
        d -= timedelta(days=1)
    print("No new bhavcopy found (holiday?).")
    return False


# ----------------------------------------------------------------------------
# Indicators + setup detection
# ----------------------------------------------------------------------------

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def scan(history: pd.DataFrame) -> pd.DataFrame:
    history = history.sort_values(["symbol", "date"])
    latest_date = history["date"].max()
    results = []

    for symbol, g in history.groupby("symbol"):
        g = g.reset_index(drop=True)
        if len(g) < 60 or g.iloc[-1]["date"] != latest_date:
            continue

        c, h, l, v = g["close"], g["high"], g["low"], g["volume"]
        today = g.iloc[-1]
        price = today["close"]

        # ---- liquidity filters ----
        avg_turnover = g["turnover"].tail(VOL_AVG_WINDOW).mean()
        if price < MIN_PRICE or avg_turnover < MIN_AVG_TURNOVER:
            continue

        avg_vol = v.iloc[:-1].tail(VOL_AVG_WINDOW).mean()
        vol_ratio = today["volume"] / avg_vol if avg_vol > 0 else 0
        day_chg = (price / today["prev_close"] - 1) * 100 if today["prev_close"] else 0
        day_range = today["high"] - today["low"]
        close_strength = ((price - today["low"]) / day_range) if day_range > 0 else 1.0

        ema20, ema50 = ema(c, 20).iloc[-1], ema(c, 50).iloc[-1]
        rsi14 = rsi(c).iloc[-1]
        hi_52w = h.tail(250).max()
        prior_20_high = h.iloc[:-1].tail(20).max()
        # consolidation tightness: range of prior 20 closes
        prior_20 = c.iloc[:-1].tail(20)
        tightness = (prior_20.max() - prior_20.min()) / prior_20.min() * 100

        setups = []

        # 1) BREAKOUT from tight 20-day range
        if (price > prior_20_high and vol_ratio >= 2.0 and tightness <= 15
                and price > ema50 and close_strength >= 0.6 and day_chg < 15):
            setups.append("BREAKOUT")

        # 2) 52-WEEK-HIGH ZONE momentum
        if (price >= 0.97 * hi_52w and vol_ratio >= 2.0
                and close_strength >= 0.7 and price > ema20 > ema50
                and 0 < day_chg < 12):
            setups.append("52W-HIGH")

        # 3) ACCUMULATION (volume + delivery footprint)
        dlv = today.get("delivery_pct", float("nan"))
        if (vol_ratio >= 3.0 and pd.notna(dlv) and dlv >= 60
                and 3 <= day_chg <= 9 and price > ema50
                and close_strength >= 0.6):
            setups.append("ACCUMULATION")

        if not setups:
            continue

        # composite score for ranking
        score = (
            len(setups) * 25
            + min(vol_ratio, 6) * 8
            + close_strength * 20
            + (10 if price >= 0.97 * hi_52w else 0)
            + (10 if 55 <= rsi14 <= 75 else 0)   # strong but not blown out
            - (15 if rsi14 > 82 else 0)
        )

        results.append({
            "symbol": symbol,
            "setups": "+".join(setups),
            "close": round(price, 2),
            "day_chg_%": round(day_chg, 1),
            "vol_x": round(vol_ratio, 1),
            "delivery_%": round(dlv, 0) if pd.notna(dlv) else "-",
            "rsi": round(rsi14, 0),
            "dist_52w_high_%": round((price / hi_52w - 1) * 100, 1),
            "turnover_cr": round(avg_turnover / 1e7, 1),
            "suggested_SL": round(today["low"] * 0.995, 2),  # below day's low
            "score": round(score, 1),
        })

    out = pd.DataFrame(results)
    if not out.empty:
        out = out.sort_values("score", ascending=False).head(15)
    return out


# ----------------------------------------------------------------------------
# Email report
# ----------------------------------------------------------------------------

def send_email(shortlist: pd.DataFrame, scan_date: str) -> None:
    if not GMAIL_USER or not GMAIL_APP_PASSWORD:
        print("Email credentials not set; printing report instead.\n")
        print(shortlist.to_string(index=False) if not shortlist.empty
              else "No setups today.")
        return

    if shortlist.empty:
        body = "<p>No qualifying momentum setups today. Sit on hands.</p>"
        subject = f"NSE Scan {scan_date}: no setups"
    else:
        body = (
            "<p>Top momentum candidates (ranked). Paste this list to Claude "
            "tomorrow morning for full chart analysis before entering.</p>"
            + shortlist.to_html(index=False, border=0)
            + "<p style='color:#888'>SL shown is a mechanical level (below "
            "day's low). Final entry/SL/target comes from the morning chart "
            "review. Risk max 1-2% of capital per trade.</p>"
        )
        subject = (f"NSE Scan {scan_date}: {len(shortlist)} setups - "
                   f"top: {shortlist.iloc[0]['symbol']}")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = GMAIL_USER
    msg["To"] = RECIPIENT
    msg.attach(MIMEText(body, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(GMAIL_USER, GMAIL_APP_PASSWORD)
        server.send_message(msg)
    print(f"Report emailed to {RECIPIENT}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "daily"
    if mode == "backfill":
        backfill()
        return

    fetch_today()
    history = load_history()
    if history.empty:
        print("No history. Run: python scanner.py backfill")
        sys.exit(1)
    scan_date = history["date"].max()
    shortlist = scan(history)
    print(f"\nScan for {scan_date}: {len(shortlist)} candidates")
    if not shortlist.empty:
        print(shortlist.to_string(index=False))
    send_email(shortlist, scan_date)


if __name__ == "__main__":
    main()
