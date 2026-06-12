"""
Backtest V2 (multi-year edition): regime filter + pullback entry + strict
universe, evaluated over every year of data available in data/ (per-year
files from backfill_multi.py plus the rolling history.csv.gz).

Reports per-variant totals AND a year-by-year breakdown, because the goal
is to see how the system behaves across regimes (2014 chop, 2017 trend,
2018 grind, 2020 crash+boom, 2021 bull, 2022 chop, 2024-26 downtrend).

Verdict bar (pre-committed): judged on robustness across variants and
years, not on the single best number.
"""

import glob

import numpy as np
import pandas as pd

from scanner import send_notice, looks_like_etf

START_EQUITY = 1_700_000
RISK_PCT = 0.015
MAX_POSITIONS = 4
MAX_POS_VALUE_PCT = 0.25
MAX_HEAT_PCT = 0.045
MAX_NEW_PER_DAY = 2
MIN_TURNOVER = 25e7
MIN_PRICE = 50.0
TRAIL_PCT = 0.07
TIME_STOP = 30
COST_RT = 0.0025
PB_VARIANTS = [0.01, 0.02, 0.03]
ORDER_VALID = 5
WARMUP = 70


def load_all() -> pd.DataFrame:
    paths = sorted(glob.glob("data/hist_*.csv.gz")) + ["data/history.csv.gz"]
    frames = []
    for p in paths:
        try:
            frames.append(pd.read_csv(p, compression="gzip"))
        except FileNotFoundError:
            continue
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["symbol", "date"], keep="last")
    for col in ("open", "high", "low", "close", "prev_close",
                "volume", "turnover"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
    df = df.dropna(subset=["close", "high", "low", "open"])
    return df.sort_values(["symbol", "date"])


def prep(hist: pd.DataFrame):
    """Per-symbol indicator frames + vectorised signal flags."""
    hist = hist[~hist["symbol"].apply(looks_like_etf)]
    per_sym = {}
    for sym, g in hist.groupby("symbol"):
        if len(g) < 60:
            continue
        g = g.reset_index(drop=True)
        c, h, v = g["close"], g["high"], g["volume"]
        ema50 = c.ewm(span=50, adjust=False).mean()
        d = c.diff()
        gain = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
        loss = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
        rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
        avg_vol = v.shift(1).rolling(20).mean()
        avg_to = g["turnover"].rolling(20).mean()
        hi52 = h.rolling(250, min_periods=60).max()
        hi20 = h.shift(1).rolling(20).max()
        p20 = c.shift(1).rolling(20)
        tight = (p20.max() - p20.min()) / p20.min() * 100

        chg = (c / g["prev_close"] - 1) * 100
        volx = v / avg_vol
        rng = (g["high"] - g["low"]).replace(0, np.nan)
        strength = (c - g["low"]) / rng
        sig = ((chg.between(3, 9)) & (volx >= 2) & (strength >= 0.6)
               & (c > ema50) & (rsi <= 78)
               & (avg_to >= MIN_TURNOVER) & (c >= MIN_PRICE)
               & ((c >= 0.95 * hi52) | ((c > hi20) & (tight <= 12))))
        g["signal"] = sig.fillna(False)
        per_sym[sym] = g.set_index("date")
    return per_sym


def build_regime(per_sym: dict, sessions: list) -> dict:
    """Equal-weight mean daily return composite (no survivorship in
    membership: every stock contributes on days it traded)."""
    rets = pd.DataFrame({
        s: g["close"].pct_change() for s, g in per_sym.items()})
    rets = rets.reindex(sessions)
    daily = rets.mean(axis=1, skipna=True).fillna(0)
    composite = (1 + daily).cumprod()
    ma50 = composite.rolling(50).mean()
    on = (composite > ma50) & (ma50 > ma50.shift(5))
    return on.to_dict()


def build_signal_calendar(per_sym: dict) -> dict:
    """date -> list of (symbol, signal_close, signal_low)."""
    cal: dict[str, list] = {}
    for sym, g in per_sym.items():
        for dt, row in g[g["signal"]].iterrows():
            cal.setdefault(dt, []).append((sym, float(row["close"]),
                                           float(row["low"])))
    return cal


def run_variant(per_sym, sessions, regime, cal, pb: float):
    equity = START_EQUITY
    open_pos, closed, orders = {}, [], []
    curve_dates, curve_vals = [], []

    for i in range(WARMUP, len(sessions)):
        day = sessions[i]

        for sym in list(open_pos):
            p = open_pos[sym]
            g = per_sym[sym]
            if day not in g.index:
                continue
            bar = g.loc[day]
            p["n"] += 1
            ex = None
            if bar["open"] <= p["sl"]:
                ex = float(bar["open"])
            elif bar["low"] <= p["sl"]:
                ex = p["sl"]
            elif p["n"] >= TIME_STOP:
                ex = float(bar["close"])
            if ex is not None:
                pnl = (ex - p["entry"]) * p["qty"] \
                    - (p["entry"] + ex) * p["qty"] * COST_RT / 2
                equity += pnl
                closed.append({"symbol": sym, "pnl": pnl, "exit_date": day,
                               "r": (ex - p["entry"]) / p["risk_ps"]})
                del open_pos[sym]
                continue
            p["hic"] = max(p["hic"], float(bar["close"]))
            p["sl"] = max(p["sl"], p["hic"] * (1 - TRAIL_PCT))

        fills, still = 0, []
        for o in orders:
            sym = o["symbol"]
            o["age"] += 1
            if o["age"] > ORDER_VALID or sym in open_pos:
                continue
            g = per_sym[sym]
            if day not in g.index:
                still.append(o)
                continue
            bar = g.loc[day]
            if bar["close"] < o["sl"]:
                continue
            if bar["low"] <= o["limit"] and fills < MAX_NEW_PER_DAY \
                    and len(open_pos) < MAX_POSITIONS:
                entry = min(float(bar["open"]), o["limit"])
                if entry <= o["sl"]:
                    continue
                risk_ps = entry - o["sl"]
                heat = sum((q["entry"] - q["sl"]) * q["qty"]
                           for q in open_pos.values() if q["sl"] < q["entry"])
                if (heat + RISK_PCT * equity) / equity > MAX_HEAT_PCT:
                    still.append(o)
                    continue
                qty = int(equity * RISK_PCT / risk_ps)
                qty = min(qty, int(equity * MAX_POS_VALUE_PCT / entry))
                if qty > 0:
                    open_pos[sym] = {"entry": entry, "sl": o["sl"],
                                     "qty": qty, "risk_ps": risk_ps,
                                     "hic": entry, "n": 0}
                    fills += 1
                continue
            still.append(o)
        orders = still

        if regime.get(day, False):
            for sym, sc, sl_ in cal.get(day, []):
                if sym not in open_pos:
                    orders.append({"symbol": sym, "age": 0,
                                   "limit": sc * (1 - pb),
                                   "sl": sl_ * 0.99})

        mtm = equity
        for s, p in open_pos.items():
            g = per_sym[s]
            if day in g.index:
                mtm += (float(g.loc[day]["close"]) - p["entry"]) * p["qty"]
        curve_dates.append(day)
        curve_vals.append(mtm)

    curve = pd.Series(curve_vals, index=pd.to_datetime(curve_dates))
    dd = ((curve - curve.cummax()) / curve.cummax()).min() * 100
    n = len(closed)
    wins = [t for t in closed if t["pnl"] > 0]
    years = (len(curve) / 250) if len(curve) else 1
    total = curve.iloc[-1] / START_EQUITY
    summary = {
        "pullback_%": pb * 100,
        "total_return_%": (total - 1) * 100,
        "CAGR_%": (total ** (1 / years) - 1) * 100,
        "max_dd_%": dd,
        "trades": n,
        "win_rate_%": len(wins) / n * 100 if n else 0,
        "avg_R": sum(t["r"] for t in closed) / n if n else 0,
    }
    yearly = curve.resample("YE").last()
    prev = START_EQUITY
    by_year = {}
    for ts, val in yearly.items():
        by_year[ts.year] = (val / prev - 1) * 100
        prev = val
    return summary, by_year


def main() -> None:
    hist = load_all()
    if hist.empty:
        print("No data files found.")
        return
    sessions = sorted(hist["date"].unique())
    print(f"Data: {sessions[0]} to {sessions[-1]} ({len(sessions)} sessions)")
    per_sym = prep(hist)
    print(f"Universe prepared: {len(per_sym)} symbols")
    regime = build_regime(per_sym, sessions)
    on = sum(1 for d in sessions[WARMUP:] if regime.get(d, False))
    print(f"Regime ON: {on}/{len(sessions)-WARMUP} sessions "
          f"({on/(len(sessions)-WARMUP)*100:.0f}%)")
    cal = build_signal_calendar(per_sym)

    summaries, yearly_all = [], {}
    for pb in PB_VARIANTS:
        s, by_year = run_variant(per_sym, sessions, regime, cal, pb)
        summaries.append(s)
        yearly_all[f"pb{pb*100:.0f}%"] = by_year
        print(f"variant {pb*100:.0f}% done: "
              f"{s['total_return_%']:+.1f}% total, CAGR {s['CAGR_%']:+.1f}%")

    rep = pd.DataFrame(summaries).round(2)
    ytab = pd.DataFrame(yearly_all).round(1)
    print("\n" + rep.to_string(index=False))
    print("\nYear-by-year returns (%):\n" + ytab.to_string())

    pos_years = (ytab > 0).sum().sum()
    tot_years = ytab.size
    verdict = ("PASS: positive CAGR in all variants, majority of years "
               "positive - proceed to paper trading"
               if all(s["CAGR_%"] > 4 for s in summaries)
               and pos_years / tot_years >= 0.6
               else "MIXED: judge carefully against buy-and-hold and FD rates"
               if any(s["CAGR_%"] > 0 for s in summaries)
               else "FAIL: do not deploy")
    print("\nVERDICT:", verdict)

    html = ("<h3>Backtest V2 - multi-year</h3>"
            f"<p>Period {sessions[0]} to {sessions[-1]}. Regime ON "
            f"{on/(len(sessions)-WARMUP)*100:.0f}% of sessions.</p>"
            + rep.to_html(index=False, border=0)
            + "<h4>Year-by-year returns (%)</h4>"
            + ytab.to_html(border=0)
            + f"<p><b>VERDICT: {verdict}</b></p>"
            "<p style='color:#888'>Costs included (0.25% round trip). "
            "Tests mechanical rules only; discretionary morning filter not "
            "simulated. Past regimes, not future promises.</p>")
    best = max(s["CAGR_%"] for s in summaries)
    send_notice(f"Backtest V2 multi-year: best CAGR {best:+.1f}% | "
                f"{verdict.split(':')[0]}", html)


if __name__ == "__main__":
    main()
