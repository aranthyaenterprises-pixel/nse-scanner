# NSE Full-Market Momentum Scanner

Scans **every NSE-listed stock** (~2,100) each evening using the official
bhavcopy and emails a ranked shortlist of momentum setups.

## What it looks for

Momentum-only, by design:

1. **BREAKOUT** — close above prior 20-day high after tight consolidation, on 2x+ volume
2. **52W-HIGH** — within 3% of 52-week high, 2x+ volume, strong close, EMAs aligned
3. **ACCUMULATION** — 3x+ volume with 60%+ delivery (institutional footprint), up 3–9%

Junk is filtered first: only EQ series, price ≥ ₹50, avg turnover ≥ ₹5 Cr.

## Setup (one time, ~10 minutes)

1. Create a **new private GitHub repo** and upload these files keeping the
   folder structure (`scanner.py`, `requirements.txt`, `.github/workflows/scan.yml`).
2. Add three **repository secrets** (Settings → Secrets and variables → Actions):
   - `GMAIL_USER` — your Gmail address
   - `GMAIL_APP_PASSWORD` — Gmail app password (same as your Sunsky checker)
   - `RECIPIENT` — where to send the report (can be same as GMAIL_USER)
3. Go to **Actions → NSE Daily Momentum Scan → Run workflow**, type `backfill`
   in the mode box, and run it once. This downloads ~260 days of history
   (takes ~10–15 min) and commits it to the repo.
4. Done. Every weekday at **7:15 PM IST** it fetches the day's data, scans,
   and emails the shortlist automatically.

## Daily routine

1. Evening: shortlist arrives by email.
2. Morning: paste the shortlist into Claude → full chart review via Kite →
   final 2–3 trade plans with entry / SL / target.
3. You place the orders. **Always place the SL with the entry.**

## Notes

- If NSE changes the bhavcopy URL format (they did in 2024), the download
  will fail loudly in the Actions log — paste the error to Claude for a fix.
- No setups on a given day = email says "sit on hands". That is a feature.
