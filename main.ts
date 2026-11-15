// btc-cent-forecast-01 entry (Node 22, standard library only): damped long-window
// log-linear trend. Forecasts the close 240 minutes after the checkpoint as:
//
//   guess = clamp(last * exp(slope * 240 * DAMP), 0.00001, 9999999.99999),
//   rounded to 5 decimals
//
// where `slope` is the ordinary-least-squares slope of log(price) over the last
// WINDOW_MIN minutes of the case's 1-minute closes, and `last` is the checkpoint
// close. The trend is damped (DAMP < 1) because a 4-hour move is mostly noise
// around the checkpoint price: undamped extrapolation and short-window momentum
// both lose to persistence on the practice cases.
//
// Backtest (~/workspace/make-fun/btc-cent-forecast-01-meta/backtest/):
//   - Practice cases p00..p04, exact actuals (each case's target is the next
//     case's checkpoint close; p05's target is outside the kit). Persistence
//     MAE over p00..p04: 47054.66 cents; the house baseline over all six
//     practice cases is 44521.576667 cents (RULES.txt).
//   - This model (48h, d=0.7): MAE 42833.95 cents over p00..p04, beats
//     persistence on 4 of 5 cases. The whole long-window family (36-72h,
//     d=0.3-0.7) beats persistence on average, so the choice is a stable
//     neighbourhood, not a knife-edge fit; leave-one-out over the 5 cases
//     picks 48h/d0.7 as the winner in all 5 folds. Short-window trends, EMA
//     momentum and mean-reversion all lose (EMA momentum: MAE >100k cents —
//     single-pool price spikes punish momentum badly).
//   - Independent large-sample check was attempted on Kraken XBTUSD 1m but
//     failed: Binance is blocked by this environment's egress proxy and
//     Kraken's OHLC pagination returned only the most recent 720 candles
//     per call (the fetch loop never advanced), so no out-of-sample venue
//     data was usable. Model selection therefore rests on the 5 practice
//     cases + leave-one-out stability (backtest/loo.mjs), not a large sample
//     — the final regime (Nov 15-17) may differ from practice (Nov 10-11),
//     and the edge is a modest ~9%, not a certainty.
//
// I/O, limits and failure modes follow the kit starter byte-for-byte in spirit:
// read every <id>.in in the input directory (grader runs one case at a time),
// write <id>.out as {"forecasts":[{"case":"<id>","price":<number>}]}, exactly one
// row, price finite in (0, 10,000,000), at most 1 MiB. Runs offline, no network,
// deterministic, well inside the 8s / 1 CPU / 1024 MB cap per case.
// If a case is malformed (no prices, non-finite last price) it throws, exactly
// like the starter — an invalid case scores as the checkpoint close anyway.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The smallest positive price on the grid: one unit of its 5 decimals. */
const MIN_PRICE = 0.00001;
/** The largest grid price below the scorer's exclusive 10,000,000-dollar cap. */
const MAX_PRICE = 9_999_999.99999;

/** Minutes of 1-minute closes the trend is fitted on (48h of the 72h window). */
const WINDOW_MIN = 2880;
/** Damping on the extrapolated 240-minute log move (see header). */
const DAMP = 0.7;
/** Horizon in minutes: the case's forecast target is `at` + 240. */
const HORIZON_MIN = 240;

interface CaseMarket {
  id: string;
  prices: number[];
}

interface Case {
  markets: CaseMarket[];
}

/** x kept on the grid's 5 decimals and inside the scorer's (0, 10,000,000). */
const price5 = (x: number): number => Math.round(Math.min(MAX_PRICE, Math.max(MIN_PRICE, x)) * 1e5) / 1e5;

/**
 * Damped log-linear trend forecast for one market: OLS slope of log(price) over
 * the last WINDOW_MIN closes, extrapolated HORIZON_MIN minutes, damped by DAMP,
 * anchored on the checkpoint close. Falls back to persistence when the window
 * is too short or the slope is not finite.
 */
function forecast(m: CaseMarket): number {
  if (!Array.isArray(m.prices) || m.prices.length === 0) throw new Error(`market ${m.id} has no prices`);
  const prices = m.prices;
  const last = prices[prices.length - 1] as number;
  if (typeof last !== 'number' || !Number.isFinite(last)) throw new Error(`market ${m.id}'s last price is not a finite number`);

  const n = Math.min(WINDOW_MIN, prices.length);
  if (n < 2) return price5(last);

  // OLS slope of log(price) against minute index over the window.
  let sx = 0;
  let sy = 0;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    const y = Math.log(prices[prices.length - n + i] as number);
    if (!Number.isFinite(y) || y <= 0) return price5(last);
    sx += i;
    sy += y;
    sxx += i * i;
    sxy += i * y;
  }
  const denom = n * sxx - sx * sx;
  if (denom === 0) return price5(last);
  const slope = (n * sxy - sx * sy) / denom;
  if (!Number.isFinite(slope)) return price5(last);

  return price5(Math.exp(Math.log(last) + slope * HORIZON_MIN * DAMP));
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (!Array.isArray(c.markets) || c.markets.length !== 1) throw new Error(`${file} does not hold exactly one market`);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ forecasts: [{ case: id, price: forecast(c.markets[0] as CaseMarket) }] })}\n`);
}
