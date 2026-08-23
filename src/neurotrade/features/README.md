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

No features are implemented yet. They arrive in Phase 1, once there is a lab
that can measure whether one is worth having.

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
