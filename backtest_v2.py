"""
Backtest V2 - redesigned after V1's -45% failure. Three structural fixes:

1. MARKET REGIME FILTER: no new entries unless the market composite (equal-
   weight index built from the liquid universe itself) is above its rising
   50-day average. Momentum longs only in uptrends.
2. PULLBACK ENTRY (no chasing): V1 bought above the high of a +5-9% surge
   day and won only 12% of the time. V2 identifies the same strong stock,
   then places a LIMIT order at a pullback below the signal close, valid 5
   sessions. Exhaustion spikes never pull back calmly -> never fill -> free
   misses. Stop sits below the signal-day low.
3. STRICT UNIVERSE: 20-day avg turnover >= Rs 25 Cr (vs 5 Cr), price >= 50,
   no ETFs. Kills the RNBDENIMS class of gap-down catastrophe.

Robustness: the same engine runs THREE pre-defined pullback depths
(1%, 2%, 3%). We require sane results across all three - not one pretty
curve. Verdict rules are printed with the results.

Run: python backtest_v2.py   (reads data/history.csv.gz, emails report)
"""

import numpy as np
import pandas as pd

from scanner import load_history, send_notice, looks_like_etf

START_EQUITY = 1_700_000
RISK_PCT = 0.015
MAX_POSITIONS = 4
MAX_POS_VALUE_PCT = 0.25
MAX_HEAT_PCT = 0.045
MAX_NEW_PER_DAY = 2
MIN_TURNOVER = 25e7          # Rs 25 Cr
MIN_PRICE = 50.0
TRAIL_PCT = 0.07
TIME_STOP = 30
COST_RT = 0.0025
PB_VARIANTS = [0.01, 0.02, 0.03]   # pullback depth below signal close
ORDER_VALID = 5                     # sessions a pullback limit stays live
WARMUP = 70


def prep(hist: pd.DataFrame):
    hist = hist[~hist["symbol"].apply(looks_like_etf)].copy()
    hist = hist.sort_values(["symbol", "date"])
    out = {}
    for sym, g in hist.groupby("symbol"):
        if len(g) < 60:
            continue
        g = g.reset_index(drop=True)
        c, h, v = g["close"], g["high"], g["volume"]
        g["ema20"] = c.ewm(span=20, adjust=False).mean()
        g["ema50"] = c.ewm(span=50, adjust=False).mean()
        d = c.diff()
        gain = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
        loss = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
        g["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
        g["avg_vol"] = v.shift(1).rolling(20).mean()
        g["avg_to"] = g["turnover"].rolling(20).mean()
        g["hi52"] = h.rolling(250, min_periods=60).max()
        g["hi20"] = h.shift(1).rolling(20).max()
        p20 = c.shift(1).rolling(20)
        g["tight"] = (p20.max() - p20.min()) / p20.min() * 100
        out[sym] = g.set_index("date")
    return out


def build_regime(per_sym: dict) -> dict:
    """Equal-weight composite of the universe -> regime ON/OFF per date."""
    closes = pd.DataFrame({s: g["close"] for s, g in per_sym.items()})
    closes = closes.dropna(axis=1, thresh=int(len(closes) * 0.9))
    norm = closes / closes.iloc[0]
    composite = norm.mean(axis=1)
    ma50 = composite.rolling(50).mean()
    rising = ma50 > ma50.shift(5)
    on = (composite > ma50) & rising
    return on.to_dict()


def signal(row) -> bool:
    if pd.isna(row["avg_vol"]) or pd.isna(row["avg_to"]):
        return False
    if row["close"] < MIN_PRICE or row["avg_to"] < MIN_TURNOVER:
        return False
    chg = (row["close"] / row["prev_close"] - 1) * 100 if row["prev_close"] else 0
    volx = row["volume"] / row["avg_vol"] if row["avg_vol"] > 0 else 0
    rng = row["high"] - row["low"]
    strength = (row["close"] - row["low"]) / rng if rng > 0 else 1
    near_hi = row["close"] >= 0.95 * row["hi52"]
    brk = (row["close"] > row["hi20"]) and row["tight"] <= 12
    return (3 <= chg <= 9 and volx >= 2 and strength >= 0.6
            and row["close"] > row["ema50"] and row["rsi"] <= 78
            and (near_hi or brk))


def run_variant(per_sym, sessions, regime, pb: float) -> dict:
    equity = START_EQUITY
    open_pos, closed, orders = {}, [], []
    curve = []

    for i in range(WARMUP, len(sessions)):
        day = sessions[i]

        # manage open positions
        for sym in list(open_pos):
            p = open_pos[sym]
            if day not in per_sym[sym].index:
                continue
            bar = per_sym[sym].loc[day]
            p["n"] += 1
            ex = None
            if bar["open"] <= p["sl"]:
                ex = bar["open"]
            elif bar["low"] <= p["sl"]:
                ex = p["sl"]
            elif p["n"] >= TIME_STOP:
                ex = bar["close"]
            if ex is not None:
                pnl = (ex - p["entry"]) * p["qty"] \
                    - (p["entry"] + ex) * p["qty"] * COST_RT / 2
                equity += pnl
                closed.append({"symbol": sym, "pnl": pnl,
                               "r": (ex - p["entry"]) / p["risk_ps"]})
                del open_pos[sym]
                continue
            p["hic"] = max(p["hic"], bar["close"])
            p["sl"] = max(p["sl"], p["hic"] * (1 - TRAIL_PCT))

        # try fills on resting pullback limit orders
        fills = 0
        still = []
        for o in orders:
            sym = o["symbol"]
            o["age"] += 1
            if o["age"] > ORDER_VALID or sym in open_pos:
                continue
            if day not in per_sym[sym].index:
                still.append(o)
                continue
            bar = per_sym[sym].loc[day]
            if bar["close"] < o["sl"]:
                continue                       # setup broke; cancel
            if bar["low"] <= o["limit"] and fills < MAX_NEW_PER_DAY \
                    and len(open_pos) < MAX_POSITIONS:
                entry = min(bar["open"], o["limit"])
                if entry <= o["sl"]:
                    continue                   # opened below stop; skip
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

        # new signals today (orders for coming sessions) - regime gated
        if regime.get(day, False):
            for sym, g in per_sym.items():
                if sym in open_pos or day not in g.index:
                    continue
                row = g.loc[day]
                if signal(row):
                    orders.append({"symbol": sym, "age": 0,
                                   "limit": row["close"] * (1 - pb),
                                   "sl": row["low"] * 0.99})

        mtm = equity + sum((per_sym[s].loc[day]["close"] - p["entry"]) * p["qty"]
                           for s, p in open_pos.items()
                           if day in per_sym[s].index)
        curve.append(mtm)

    ser = pd.Series(curve)
    dd = ((ser - ser.cummax()) / ser.cummax()).min() * 100 if len(ser) else 0
    n = len(closed)
    wins = [t for t in closed if t["pnl"] > 0]
    return {
        "pb": pb * 100,
        "return": (ser.iloc[-1] / START_EQUITY - 1) * 100 if len(ser) else 0,
        "max_dd": dd,
        "trades": n,
        "win_rate": len(wins) / n * 100 if n else 0,
        "avg_r": sum(t["r"] for t in closed) / n if n else 0,
        "open_end": len(open_pos),
    }


def main() -> None:
    hist = load_history()
    if hist.empty:
        print("No history file.")
        return
    per_sym = prep(hist)
    sessions = sorted(hist["date"].unique())
    regime = build_regime(per_sym)
    on_days = sum(1 for d in sessions[WARMUP:] if regime.get(d, False))
    print(f"Universe: {len(per_sym)} symbols | regime ON "
          f"{on_days}/{len(sessions)-WARMUP} sessions")

    rows = [run_variant(per_sym, sessions, regime, pb) for pb in PB_VARIANTS]
    rep = pd.DataFrame(rows).round(2)
    rep.columns = ["pullback_%", "return_%", "max_dd_%", "trades",
                   "win_rate_%", "avg_R", "open_at_end"]
    print("\n" + rep.to_string(index=False))

    verdict = ("PASS: positive across variants - proceed to paper trading"
               if all(r["return"] > 0 for r in rows) and
                  all(r["max_dd"] > -15 for r in rows)
               else "MIXED: some variants negative - judge carefully"
               if any(r["return"] > 0 for r in rows)
               else "FAIL: negative across variants - do not deploy")
    print("\nVERDICT:", verdict)

    html = ("<h3>Backtest V2 (regime filter + pullback entry + strict "
            "universe)</h3>"
            f"<p>Regime was ON for {on_days} of {len(sessions)-WARMUP} "
            "sessions - entries only allowed then.</p>"
            + rep.to_html(index=False, border=0)
            + f"<p><b>VERDICT: {verdict}</b></p>"
            "<p style='color:#888'>Pre-committed bar: all three variants "
            "positive with max drawdown above -15% before paper trading "
            "begins. No parameter tuning beyond these variants.</p>")
    best = max(r["return"] for r in rows)
    send_notice(f"Backtest V2: best {best:+.1f}% | {verdict.split(':')[0]}",
                html)


if __name__ == "__main__":
    main()
