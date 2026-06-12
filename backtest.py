"""
Backtest: replays the scanner's exact signals over the stored history file
(data/history.csv.gz) and simulates trading them under the agreed framework.

Rules simulated (mirrors the live plan):
  - Signals: scanner.scan() run on each historical day (identical logic)
  - Entry: next day, buy-stop above signal-day high (+0.1% buffer).
           Gap above trigger -> filled at open (honest, includes gap cost)
  - Initial SL: signal day's low * 0.995 (scanner's suggested_SL)
  - Trailing: SL ratchets up to highest close since entry * 0.93
  - Time stop: exit after 40 sessions if neither SL nor trail hit
  - Sizing: 1.5% of current equity at risk per trade
  - Caps: max 4 open positions, 25% of equity per position, 4.5% open heat
  - Max 2 new entries per day (top-scored first), no pyramiding same symbol
  - Costs: 0.25% of turnover round-trip (brokerage+STT+slippage estimate)

Output: printed report + email (uses scanner.py's email credentials).
Run: python backtest.py
"""

import pandas as pd

from scanner import scan, load_history, send_notice, looks_like_etf

RISK_PCT = 0.015
MAX_POSITIONS = 4
MAX_POS_VALUE_PCT = 0.25
MAX_HEAT_PCT = 0.045
MAX_NEW_PER_DAY = 2
TRAIL_PCT = 0.07
TIME_STOP_SESSIONS = 40
COST_RT = 0.0025          # round-trip cost on turnover
START_EQUITY = 1_700_000
WARMUP_SESSIONS = 70


def run() -> None:
    hist = load_history()
    if hist.empty:
        print("No history file found. Run backfill first.")
        return
    hist = hist[~hist["symbol"].apply(looks_like_etf)]
    hist = hist.sort_values(["symbol", "date"])
    sessions = sorted(hist["date"].unique())
    if len(sessions) <= WARMUP_SESSIONS + 5:
        print("Not enough history to backtest.")
        return

    # fast per-symbol day lookup: bars[symbol][date] -> row
    bars: dict[str, dict[str, dict]] = {}
    for sym, g in hist.groupby("symbol"):
        bars[sym] = {r["date"]: r for r in g.to_dict("records")}

    equity = START_EQUITY
    open_pos: dict[str, dict] = {}
    closed: list[dict] = []
    pending: list[dict] = []          # signals generated yesterday
    equity_curve: list[tuple[str, float]] = []

    for i in range(WARMUP_SESSIONS, len(sessions)):
        day = sessions[i]

        # ---- 1) manage open positions on today's bar ----
        for sym in list(open_pos):
            p = open_pos[sym]
            bar = bars[sym].get(day)
            if bar is None:
                continue
            p["sessions"] += 1
            exit_price = None
            sl = p["sl"]
            if bar["open"] <= sl:
                exit_price = bar["open"]          # gap through stop
            elif bar["low"] <= sl:
                exit_price = sl
            elif p["sessions"] >= TIME_STOP_SESSIONS:
                exit_price = bar["close"]
            if exit_price is not None:
                pnl = (exit_price - p["entry"]) * p["qty"]
                cost = (p["entry"] + exit_price) * p["qty"] * COST_RT / 2
                equity += pnl - cost
                closed.append({**p, "symbol": sym, "exit": exit_price,
                               "exit_date": day, "pnl": pnl - cost,
                               "r": (exit_price - p["entry"]) / p["risk_ps"]})
                del open_pos[sym]
                continue
            # trail the stop on the close
            p["hi_close"] = max(p["hi_close"], bar["close"])
            p["sl"] = max(p["sl"], p["hi_close"] * (1 - TRAIL_PCT))

        # ---- 2) try to fill yesterday's signals today ----
        fills = 0
        for sig in sorted(pending, key=lambda s: -s["score"]):
            if fills >= MAX_NEW_PER_DAY or len(open_pos) >= MAX_POSITIONS:
                break
            sym = sig["symbol"]
            if sym in open_pos or sym not in bars:
                continue
            bar = bars[sym].get(day)
            if bar is None:
                continue
            trigger = sig["trigger"]
            if bar["high"] < trigger:
                continue                          # never triggered today
            entry = max(bar["open"], trigger)
            sl = sig["sl"]
            risk_ps = entry - sl
            if risk_ps <= 0:
                continue
            heat = sum((q["entry"] - q["sl"]) * q["qty"]
                       for q in open_pos.values() if q["sl"] < q["entry"])
            if (heat + RISK_PCT * equity) / equity > MAX_HEAT_PCT:
                continue
            qty = int((equity * RISK_PCT) / risk_ps)
            qty = min(qty, int(equity * MAX_POS_VALUE_PCT / entry))
            if qty <= 0:
                continue
            open_pos[sym] = {"entry": entry, "sl": sl, "qty": qty,
                             "risk_ps": risk_ps, "hi_close": entry,
                             "sessions": 0, "entry_date": day}
            fills += 1
        pending = []

        # ---- 3) generate today's signals (for tomorrow's open) ----
        day_hist = hist[hist["date"] <= day]
        shortlist = scan(day_hist)
        if not shortlist.empty:
            for _, row in shortlist.head(5).iterrows():
                sym = row["symbol"]
                sig_bar = bars.get(sym, {}).get(day)
                if sig_bar is None:
                    continue
                pending.append({"symbol": sym, "score": row["score"],
                                "trigger": sig_bar["high"] * 1.001,
                                "sl": row["suggested_SL"]})

        # ---- 4) mark-to-market equity curve ----
        mtm = equity
        for sym, p in open_pos.items():
            bar = bars[sym].get(day)
            if bar is not None:
                mtm += (bar["close"] - p["entry"]) * p["qty"]
        equity_curve.append((day, mtm))

        if i % 20 == 0:
            print(f"{day}  equity={mtm:,.0f}  open={len(open_pos)}  "
                  f"closed={len(closed)}")

    # ---- report ----
    curve = pd.Series({d: v for d, v in equity_curve})
    peak = curve.cummax()
    max_dd = ((curve - peak) / peak).min() * 100
    total_ret = (curve.iloc[-1] / START_EQUITY - 1) * 100
    n = len(closed)
    wins = [t for t in closed if t["pnl"] > 0]
    losses = [t for t in closed if t["pnl"] <= 0]
    win_rate = len(wins) / n * 100 if n else 0
    avg_win = sum(t["pnl"] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t["pnl"] for t in losses) / len(losses) if losses else 0
    avg_r = sum(t["r"] for t in closed) / n if n else 0

    lines = [
        f"Period: {sessions[WARMUP_SESSIONS]} to {sessions[-1]} "
        f"({len(sessions)-WARMUP_SESSIONS} sessions)",
        f"Start equity: Rs {START_EQUITY:,.0f}",
        f"End equity (incl. open MTM): Rs {curve.iloc[-1]:,.0f}",
        f"TOTAL RETURN: {total_ret:+.1f}%",
        f"Max drawdown: {max_dd:.1f}%",
        f"Closed trades: {n}  |  Win rate: {win_rate:.0f}%",
        f"Avg win: Rs {avg_win:,.0f}  |  Avg loss: Rs {avg_loss:,.0f}",
        f"Avg R per trade: {avg_r:+.2f}",
        f"Still open at end: {len(open_pos)}",
    ]
    report = "\n".join(lines)
    print("\n" + "=" * 60 + "\nBACKTEST RESULT\n" + "=" * 60 + "\n" + report)

    top = sorted(closed, key=lambda t: -t["pnl"])[:5]
    bottom = sorted(closed, key=lambda t: t["pnl"])[:5]
    html = ("<h3>Backtest result (mechanical system, costs included)</h3><pre>"
            + report + "</pre><p><b>Best 5:</b> "
            + ", ".join(f"{t['symbol']} {t['pnl']:+,.0f}" for t in top)
            + "<br><b>Worst 5:</b> "
            + ", ".join(f"{t['symbol']} {t['pnl']:+,.0f}" for t in bottom)
            + "</p><p style='color:#888'>One year, one regime. This tests the "
            "mechanical rules only - the morning discretionary filter is not "
            "simulated. Treat as evidence, not prophecy.</p>")
    send_notice(f"Backtest: {total_ret:+.1f}% | DD {max_dd:.1f}% | "
                f"{n} trades | WR {win_rate:.0f}%", html)


if __name__ == "__main__":
    run()
