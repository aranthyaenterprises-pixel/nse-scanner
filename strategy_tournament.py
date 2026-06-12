"""
Strategy tournament: five pre-registered strategy families tested in ONE
batch over the full dataset, each judged independently on an in-sample
window (to 2020-12-31) and out-of-sample window (2021 onward). Multiple-
testing protection: nothing is tuned after results; pass requires BOTH
windows; any pass goes to paper trading, never directly to capital.

Strategies (all long-only, liquid universe: price>=50, 60d avg turnover
>= 25 Cr, no ETFs; costs 0.25% RT on replaced fraction):
  S1 MEANREV  - weekly: 8 biggest 5-day losers that remain above their
                200-day average (dip in an uptrend), hold 1 week
  S2 TREND    - composite > its 200-day MA: hold equal-weight liquid
                universe; below: cash
  S3 HI52     - monthly: 8 stocks closest to their 52-week high
  S4 LOWVOL   - monthly: 8 lowest 252-day volatility stocks
  S5 STMOM    - weekly: 8 best 4-week returns, hold 1 week

Pre-committed pass bar PER STRATEGY:
  positive CAGR in BOTH windows, AND OOS CAGR >= 12%, AND OOS CAGR >=
  benchmark OOS CAGR, AND OOS max drawdown >= -30%.

Run: python strategy_tournament.py
"""

import glob

import numpy as np
import pandas as pd

from scanner import send_notice, looks_like_etf

TOP_N = 8
MIN_TURNOVER = 25e7
MIN_PRICE = 50.0
COST_RT = 0.0025
SPLIT_DATE = "2020-12-31"


def load_closes():
    paths = sorted(glob.glob("data/hist_*.csv.gz")) + ["data/history.csv.gz"]
    frames = []
    for p in paths:
        try:
            frames.append(pd.read_csv(p, compression="gzip",
                                      usecols=["symbol", "series", "date",
                                               "close", "turnover"]))
        except FileNotFoundError:
            continue
    df = pd.concat(frames, ignore_index=True)
    df = df[df["series"] == "EQ"]
    df = df.drop_duplicates(subset=["symbol", "date"], keep="last")
    df = df[~df["symbol"].apply(looks_like_etf)]
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["turnover"] = pd.to_numeric(df["turnover"], errors="coerce")
    df = df.dropna(subset=["close"])
    closes = df.pivot_table(index="date", columns="symbol", values="close",
                            aggfunc="last").sort_index().astype("float32")
    tos = df.pivot_table(index="date", columns="symbol", values="turnover",
                         aggfunc="last").sort_index().astype("float32")
    return closes, tos


def simulate(picks_at, rebal_idx, rets, dates):
    """Generic portfolio simulator. picks_at: dict i -> list of symbols."""
    equity, held = 1.0, []
    vals = []
    rebal_set = set(rebal_idx)
    for i in range(rebal_idx[0], len(dates)):
        if i in rebal_set:
            new = picks_at.get(i, [])
            changed = len(set(held) ^ set(new))
            slots = max(len(held), len(new), 1)
            equity *= 1 - COST_RT * changed / (2 * slots)
            held = new
        if held:
            r = rets.iloc[i][held].fillna(0).mean()
            equity *= 1 + float(r)
        vals.append((dates[i], equity))
    return pd.Series([v for _, v in vals],
                     index=pd.to_datetime([d for d, _ in vals]))


def window_stats(curve, split):
    out = {}
    for name, seg in (("IS", curve[curve.index <= split]),
                      ("OOS", curve[curve.index > split])):
        if len(seg) < 50:
            out[name] = (np.nan, np.nan)
            continue
        seg = seg / seg.iloc[0]
        yrs = len(seg) / 250
        cagr = (seg.iloc[-1] ** (1 / yrs) - 1) * 100
        dd = ((seg - seg.cummax()) / seg.cummax()).min() * 100
        out[name] = (cagr, dd)
    return out


def main():
    closes, tos = load_closes()
    dates = closes.index.tolist()
    rets = closes.pct_change()
    print(f"Matrix: {len(dates)} x {closes.shape[1]}")

    avg_to = tos.rolling(60, min_periods=40).mean()
    elig = (closes >= MIN_PRICE) & (avg_to >= MIN_TURNOVER)
    ma200 = closes.rolling(200, min_periods=150).mean()
    composite = (1 + rets.mean(axis=1, skipna=True).fillna(0)).cumprod()
    comp_ma200 = composite.rolling(200).mean()

    months = pd.Series(pd.to_datetime(dates)).dt.to_period("M")
    monthly_idx = [i for i in range(1, len(dates))
                   if months[i] != months[i - 1] and i > 270]
    weekly_idx = list(range(270, len(dates), 5))

    r5 = closes / closes.shift(5) - 1          # 5-day return
    r20 = closes / closes.shift(20) - 1        # 4-week return
    hi52 = closes.rolling(250, min_periods=120).max()
    prox = closes / hi52                       # closeness to 52w high
    vol252 = rets.rolling(252, min_periods=150).std()

    def rank_picks(idx_list, score_df, ascending, extra_mask=None):
        picks = {}
        for i in idx_list:
            row = score_df.iloc[i - 1]
            m = elig.iloc[i - 1] & row.notna()
            if extra_mask is not None:
                m &= extra_mask.iloc[i - 1].fillna(False)
            s = row[m].sort_values(ascending=ascending)
            picks[i] = list(s.head(TOP_N).index)
        return picks

    above200 = closes > ma200

    strategies = {
        "S1_MEANREV": (weekly_idx,
                       rank_picks(weekly_idx, r5, True, above200)),
        "S3_HI52": (monthly_idx, rank_picks(monthly_idx, prox, False)),
        "S4_LOWVOL": (monthly_idx, rank_picks(monthly_idx, vol252, True)),
        "S5_STMOM": (weekly_idx, rank_picks(weekly_idx, r20, False)),
    }

    # S2 TREND: composite timing of the whole eligible universe
    trend_picks = {}
    for i in monthly_idx:
        if composite.iloc[i - 1] > comp_ma200.iloc[i - 1]:
            m = elig.iloc[i - 1]
            trend_picks[i] = list(m[m].index)
        else:
            trend_picks[i] = []
    strategies["S2_TREND"] = (monthly_idx, trend_picks)

    # benchmark: equal-weight eligible universe, always invested
    bench_picks = {i: list(elig.iloc[i - 1][elig.iloc[i - 1]].index)
                   for i in monthly_idx}

    split = pd.Timestamp(SPLIT_DATE)
    rows = []
    bench_curve = simulate(bench_picks, monthly_idx, rets, dates)
    bws = window_stats(bench_curve, split)
    rows.append({"strategy": "BENCHMARK", "IS_CAGR": bws["IS"][0],
                 "IS_DD": bws["IS"][1], "OOS_CAGR": bws["OOS"][0],
                 "OOS_DD": bws["OOS"][1], "verdict": "-"})
    print(f"BENCHMARK IS {bws['IS'][0]:+.1f}% OOS {bws['OOS'][0]:+.1f}%")

    for name, (idx, picks) in strategies.items():
        curve = simulate(picks, idx, rets, dates)
        ws = window_stats(curve, split)
        is_c, is_d = ws["IS"]
        oos_c, oos_d = ws["OOS"]
        passed = (is_c > 0 and oos_c > 0 and oos_c >= 12
                  and oos_c >= bws["OOS"][0] and oos_d >= -30)
        rows.append({"strategy": name, "IS_CAGR": is_c, "IS_DD": is_d,
                     "OOS_CAGR": oos_c, "OOS_DD": oos_d,
                     "verdict": "PASS" if passed else "fail"})
        print(f"{name}: IS {is_c:+.1f}%/{is_d:.0f}%  "
              f"OOS {oos_c:+.1f}%/{oos_d:.0f}%  "
              f"{'PASS' if passed else 'fail'}")

    rep = pd.DataFrame(rows).round(1)
    passes = [r["strategy"] for r in rows if r["verdict"] == "PASS"]
    overall = (f"{len(passes)} strategy(ies) passed: {', '.join(passes)} "
               "- eligible for PAPER TRADING only"
               if passes else
               "NO strategy passed both windows - search closes per "
               "agreement; capital proceeds to core/satellite plan")
    print("\n" + rep.to_string(index=False) + "\n\nOVERALL: " + overall)

    html = ("<h3>Strategy tournament - 5 pre-registered families</h3>"
            "<p>Judged independently in-sample (to 2020) and out-of-sample "
            "(2021+). Pass bar: positive both windows, OOS CAGR >= 12% and "
            ">= benchmark, OOS DD >= -30%. Costs included.</p>"
            + rep.to_html(index=False, border=0)
            + f"<p><b>OVERALL: {overall}</b></p>"
            "<p style='color:#888'>Five tests against one dataset means "
            "even a PASS carries multiple-testing risk; that is why any "
            "pass earns paper trading, never direct deployment.</p>")
    send_notice(f"Tournament: {len(passes)} of 5 passed", html)


if __name__ == "__main__":
    main()
