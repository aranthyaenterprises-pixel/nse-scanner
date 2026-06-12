"""
Rotation backtest: monthly relative-strength portfolio, tested over the
full multi-year dataset (data/hist_*.csv.gz + data/history.csv.gz).

Rules (pre-committed):
  - Universe: EQ series, no ETFs, price >= 50, 60-day avg turnover >= 25 Cr
  - On the first trading day of each month, rank eligible stocks by
    trailing return over LOOKBACK sessions (skipping the most recent 5
    sessions to avoid short-term reversal noise)
  - Hold the TOP_N equally weighted until next rebalance
  - Regime filter: equal-weight market composite must be above its rising
    50-day average at rebalance; otherwise the month is spent in CASH (0%)
  - Costs: 0.25% round trip applied to the portfolio fraction replaced
  - Variants: 63 / 126 / 252 session lookbacks (3 / 6 / 12 months)

Output: CAGR, max drawdown, yearly table per variant + equal-weight
universe benchmark for honest comparison. Emailed via scanner credentials.

Run: python rotation_backtest.py
"""

import glob

import numpy as np
import pandas as pd

from scanner import send_notice, looks_like_etf

TOP_N = 8
LOOKBACKS = [63, 126, 252]
SKIP = 5
MIN_TURNOVER = 25e7
MIN_PRICE = 50.0
COST_RT = 0.0025
START_EQUITY = 1_700_000


def load_all() -> pd.DataFrame:
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
    return df.dropna(subset=["close"])


def main() -> None:
    df = load_all()
    closes = df.pivot_table(index="date", columns="symbol", values="close",
                            aggfunc="last").sort_index()
    tos = df.pivot_table(index="date", columns="symbol", values="turnover",
                         aggfunc="last").sort_index()
    closes = closes.astype("float32")
    dates = closes.index.tolist()
    print(f"Matrix: {len(dates)} sessions x {closes.shape[1]} symbols")

    rets = closes.pct_change()
    # market regime from equal-weight mean daily return composite
    composite = (1 + rets.mean(axis=1, skipna=True).fillna(0)).cumprod()
    ma50 = composite.rolling(50).mean()
    regime_on = (composite > ma50) & (ma50 > ma50.shift(5))

    avg_to = tos.rolling(60, min_periods=40).mean()

    # benchmark: equal-weight universe buy & hold over evaluated window
    # rebalance dates: first session of each month, after enough warmup
    months = pd.Series(pd.to_datetime(dates)).dt.to_period("M")
    first_of_month = [i for i in range(1, len(dates))
                      if months[i] != months[i - 1]]
    first_of_month = [i for i in first_of_month if i > max(LOOKBACKS) + SKIP]
    if not first_of_month:
        print("Not enough data.")
        return
    start_i = first_of_month[0]

    bench = composite.iloc[start_i:]
    bench_years = len(bench) / 250
    bench_cagr = ((bench.iloc[-1] / bench.iloc[0]) ** (1 / bench_years) - 1) * 100

    summaries, yearly_all = [], {}
    for lb in LOOKBACKS:
        mom = closes.shift(SKIP) / closes.shift(SKIP + lb) - 1
        equity = 1.0
        held: list[str] = []
        curve_d, curve_v = [], []
        for k, i in enumerate(first_of_month):
            day = dates[i]
            nxt = first_of_month[k + 1] if k + 1 < len(first_of_month) \
                else len(dates)
            if bool(regime_on.iloc[i - 1]):
                row_mom = mom.iloc[i - 1]
                elig = (closes.iloc[i - 1] >= MIN_PRICE) \
                    & (avg_to.iloc[i - 1] >= MIN_TURNOVER) \
                    & row_mom.notna()
                ranked = row_mom[elig].sort_values(ascending=False)
                new = list(ranked.head(TOP_N).index)
            else:
                new = []
            changed = len(set(held) ^ set(new))
            slots = max(len(held), len(new), 1)
            equity *= 1 - COST_RT * changed / (2 * slots)
            held = new
            seg = rets.iloc[i:nxt][held] if held else None
            for j in range(i, nxt):
                if held:
                    r = rets.iloc[j][held].fillna(0).mean()
                    equity *= 1 + float(r)
                curve_d.append(dates[j])
                curve_v.append(equity)

        curve = pd.Series(curve_v, index=pd.to_datetime(curve_d))
        dd = ((curve - curve.cummax()) / curve.cummax()).min() * 100
        years = len(curve) / 250
        cagr = (curve.iloc[-1] ** (1 / years) - 1) * 100
        summaries.append({
            "lookback_m": lb // 21,
            "total_return_%": (curve.iloc[-1] - 1) * 100,
            "CAGR_%": cagr,
            "max_dd_%": dd,
        })
        yearly = curve.resample("YE").last()
        prev, by_year = 1.0, {}
        for ts, v in yearly.items():
            by_year[ts.year] = (v / prev - 1) * 100
            prev = v
        yearly_all[f"{lb//21}m"] = by_year
        print(f"lookback {lb//21}m: CAGR {cagr:+.1f}%, maxDD {dd:.1f}%")

    rep = pd.DataFrame(summaries).round(2)
    ytab = pd.DataFrame(yearly_all).round(1)
    print("\n" + rep.to_string(index=False))
    print(f"\nEqual-weight universe benchmark CAGR: {bench_cagr:+.1f}%")
    print("\nYear-by-year (%):\n" + ytab.to_string())

    verdict = ("PASS: all lookbacks clear 10% CAGR with max DD above -30% "
               "- proceed to paper trading"
               if all(s["CAGR_%"] >= 10 for s in summaries)
               and all(s["max_dd_%"] >= -30 for s in summaries)
               else "MIXED: judge against benchmark and drawdowns"
               if any(s["CAGR_%"] >= 8 for s in summaries)
               else "FAIL: does not beat the lazy alternatives")
    print("\nVERDICT:", verdict)

    html = ("<h3>Rotation backtest (monthly relative strength)</h3>"
            f"<p>Top {TOP_N} liquid stocks, monthly rebalance, regime-gated "
            "to cash. Costs included.</p>"
            + rep.to_html(index=False, border=0)
            + f"<p>Equal-weight universe benchmark CAGR: "
            f"<b>{bench_cagr:+.1f}%</b> (the do-nothing comparison)</p>"
            "<h4>Year-by-year returns (%)</h4>" + ytab.to_html(border=0)
            + f"<p><b>VERDICT: {verdict}</b></p>"
            "<p style='color:#888'>Cash months earn 0% here; in reality a "
            "liquid fund would add ~6% annualised on idle periods. STCG/LTCG "
            "tax not included. Past regimes, not future promises.</p>")
    best = max(s["CAGR_%"] for s in summaries)
    send_notice(f"Rotation backtest: best CAGR {best:+.1f}% | "
                f"{verdict.split(':')[0]}", html)


if __name__ == "__main__":
    main()
