# Feature library v1

The first real features, against the registry Phase 0 built. §20.5 sets the scope: what ORB on
Stocks-in-Play and beta-adjusted relative strength need, not a comprehensive library.

## What exists

| Module | Contents |
|---|---|
| `features/indicators.py` | `indicators` registry + `log_return`, `atr`, `realised_vol`, `rvol`, `ema`, `fracdiff_close`; helpers `true_range`, `fracdiff_weights` |
| `features/levels.py` | `OpeningRange`, `opening_range`, `session_vwap`, `vwap_distance` — plain functions, **not** registered |

`registry.py` was already built in Phase 0 and is unchanged: `FeatureSpec` declares a lookback,
`evaluate(history, as_of)` returns `None` while warming up and raises `LookaheadError` on any bar
stamped after the moment.

## The split, and why it exists

A registered feature gets a **fixed trailing window** — exactly `lookback` bars. A level anchored
to the *session open* has no fixed bar count: the distance from the open changes every minute.
Forcing those into the registry would mean lying about the window or sizing it for the worst case.
§5.3 already calls them "shared infrastructure feeding every other strategy", so `levels.py`
follows the spec rather than working around the registry.

**The cost is real and worth stating: the lookahead guard does not apply to `levels.py`.** Nothing
in those signatures carries an `as_of` to check. `session_vwap` takes an optional `up_to`; the rest
trust the caller. This is the only place in the feature library where lookahead is not
structurally prevented.

## The decisions

**ATR is a plain mean, not Wilder's smoothing. EMA is seeded from its window, not carried.** Both
for one reason: a value that depends on history outside the declared lookback is not reproducible
from that lookback, so the same bar gets a different number depending on how much history the
slice happened to include. The backtest and the live engine would then disagree about the same
bar — the §3.6 failure mode. Tests pin this by prepending 50 unrelated bars and asserting the
value is unchanged.

**`fracdiff_close` uses d=0.4 as a placeholder, and says so.** §8 wants "stationary without
destroying memory". The right `d` is the smallest that passes a stationarity test on this corpus,
and that test does not exist yet. The constant carries the caveat; the version string moves when
it is replaced.

**`rvol` returns 0.0 when the baseline window has no volume.** A real state for an illiquid name.
The alternatives are a division by zero or an infinity propagating into every model input.

**`opening_range` returns `None` for a short session rather than a partial range.** A 15-minute
range built from 4 bars is a different statistic wearing the same name, and a strategy trading it
would be trading a level nobody else is watching.

**A breakout is strictly above the high, not at it** — otherwise the signal fires on the bar that
set the level.

**`session_vwap` prefers the feed's own per-bar VWAP and falls back to `(h+l+c)/3`.** The feed's
figure is computed from every print inside the bar, so it is strictly better where it exists.

**Features return `float`; levels return `Price`.** An indicator is an estimate and feeds a model.
A breakout level becomes an order's limit price, and an order price is money.

## Measured on real bars

AAPL, 2026-09-15, 390 IBKR minute bars (a full session): `atr` 0.337, `realised_vol` 0.178,
`rvol` 4.71 (the closing minute, as expected), `ema` 330.81, 15-minute opening range
331.50 / 328.35, session VWAP 330.26 with the close 0.33% above it. All plausible; none verified
against an independent implementation, which is what the lab is for.

## Not done yet

- **No session-boundary handling in `indicators.py`.** A 20-bar EMA at 09:35 reaches into
  yesterday. For an intraday strategy that is usually wrong, and the fix is for the caller to pass
  a window starting at the session open. Nothing enforces it.
- **No cross-sectional features.** Beta-adjusted relative strength (§5.4) needs a regression
  against SPY plus a sector ETF, which is a different shape — many instruments at one instant,
  not one instrument over time. The registry has no vocabulary for it yet.
- **No prior-day levels** (prior high/low/close, pre-market range). They need the previous
  session's bars alongside today's, which is a second seam `levels.py` does not have.
- **Nothing consumes these.** No strategy, no label, no model. Next is triple-barrier labelling.
