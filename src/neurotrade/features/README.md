# features

Calculations over market data — moving averages, volatility, relative volume,
and everything else a strategy reads instead of raw prices.

**One implementation, used by both research and live.** This is the single most
important guarantee in the system. A calculation done one way in the backtest and
another way in production makes every backtest result a claim about software that
is not the software trading your money.

## Files

| File | What it does |
|---|---|
| `registry.py` | Registering a feature, and the rules every feature obeys |
| `indicators.py` | The registered features: log return, ATR, realised volatility, relative volume, EMA, fractionally differenced close |
| `resolver.py` | The rolling per-symbol window a feature is computed from |
| `levels.py` | Session-anchored reference levels — opening range, session VWAP, prior close, the gap. Plain functions plus the tracker that keeps them current |

**Why `levels.py` sits outside the registry.** A registered feature declares a
fixed lookback and is handed exactly that many bars. A level anchored to the
session open has no fixed bar count — the distance from the open changes every
minute — so there is nothing to declare. §5.3 already treats these as separate
"shared infrastructure", so the split follows the spec rather than working
around it.

That leaves the lookahead guard not applying to the plain functions: pass bars
up to the decision moment and no further. **`SessionLevelTracker` is the
answer to that**, and it is what the engine actually uses. Bars are folded in
one at a time as they close, so there is no window to pass and no way to pass
one reaching past the decision moment. The batch functions remain for tests and
one-off analysis; the two cannot disagree, because the rule for what a bar
contributes to a VWAP lives in `vwap_contribution` and both call it.

A tracked level resets at the session boundary. Two things survive it, and both
are *about* the boundary: the prior close, and the high-low range of the last
`RANGE_MEMORY` sessions.

**That range history is there to give a gap a scale.** Published gap thresholds
("wider than 1.2 ATR fills less than a tenth of the time") are quoted against
*daily* volatility. The corpus feeds one-minute bars, so the registered `atr` is
a one-minute statistic and the same gap expressed in it is larger by two orders
of magnitude — a number no published threshold applies to. `gap_in_ranges`
divides by the mean recent session range instead, and is `None` until the history
is full rather than averaging three days and calling it fourteen.

**Who keeps the history.** A `FeatureSpec` is handed a window and remembers
nothing, so something has to hold the bars. `resolver.py` does, per symbol, and
it is the only thing that does — which is what lets `StrategyContext` hand a
strategy values and no series at all. The window is continuous across sessions
on purpose: `max_lookback` exists so the engine can load history *before* a
session opens, and a buffer that reset at midnight would leave every
open-of-session strategy cold for its first `lookback` bars.

## The two rules a feature obeys

**It declares how much history it needs.** A 20-bar average computed from 8 bars
is not a 20-bar average — it is a different number that looks plausible. Until
the window is full, a feature returns "not yet" rather than a value. That is not
zero, and a strategy must not treat it as zero.

**It cannot see the future.** Every calculation is given the moment being
modelled and refuses any data stamped after it. Using tomorrow's price by
accident is the most productive way to build a backtest that works beautifully
and loses money live, and it is close to invisible on inspection — the code looks
right and the numbers look right. So it raises instead.
