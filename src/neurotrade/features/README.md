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
| `levels.py` | Session-anchored reference levels — opening range, session VWAP, distance from it. Plain functions, not registered |

**Why `levels.py` sits outside the registry.** A registered feature declares a
fixed lookback and is handed exactly that many bars. A level anchored to the
session open has no fixed bar count — the distance from the open changes every
minute — so there is nothing to declare. §5.3 already treats these as separate
"shared infrastructure", so the split follows the spec rather than working
around it. The cost is that the lookahead guard does not apply to them: pass
bars up to the decision moment and no further.

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
