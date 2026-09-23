# strategies

The trading ideas. Each one answers a single question: *given what is knowable
right now, is there a trade here, which way, and where would I be wrong?*

A strategy does **not** decide how much to buy. It produces a proposal; the risk
engine decides size from the model's confidence. That split keeps direction
explainable and confines machine learning to the job it is actually good at —
filtering out the proposals that will not work.

## Files

| File | What it does |
|---|---|
| `base.py` | The contract every strategy implements, and the registry that holds them |
| `context.py` | Fills in what a strategy may see: venue phase, regime, features, session levels |

No strategies are implemented yet. The first two arrive in Phase 2: an opening-
range breakout, and one that trades a stock's movement relative to the market.

## What a strategy can and cannot see

It receives a small bundle: the instrument, the moment, the market phase, the
current market conditions, and its declared feature values. That is all.

No price history, no clock, no broker. Each of those is withheld deliberately —
history would let it look ahead, a clock would break replay, and a broker would
let it skip the risk engine. Making those impossible is stronger than asking
everyone to remember.

A strategy also cannot read a feature it did not declare. The engine only
prepares what was asked for, so an undeclared one would be missing or computed
over the wrong window, and neither is visible in the number itself.

The session anchors are the one thing nobody declares: where the session opened,
how far it has travelled, the volume-weighted average price, the opening ranges
that have completed, and yesterday's close. §5.3 calls them shared
infrastructure feeding every other strategy, they cost one fold per bar, and
there is no history to load for them — so there is nothing for a declaration to
tell the host. They arrive whether or not anything asked, and they are `None`
before the session's first bar rather than half-filled.

## Market conditions gate what may run

Trend days, choppy days and reversal days reward different tools. Each strategy
declares which conditions it is allowed to fire in, and the host enforces it — so
a breakout strategy and a fade-the-breakout strategy can never be live at the
same time.

**Only one of those conditions is classified today.** The model that tells trend
from chop is an HMM and arrives in Phase 5. Fitting a stand-in now would put an
unvalidated model in the decision path of every result Phase 2 produces, and its
thresholds would spend trial budget that the deflated Sharpe counts against
every strategy. So `context.py` classifies the one part of §5.7 that is a clock
fact rather than a model — the 12:00–14:00 ET liquidity lull, a default no-trade
window — and reports everything else as unclassified, which grants nothing.

That leaves research needing a way to measure a strategy before the classifier
exists. The backtest engine has one: it can treat *unclassified* as permissive,
and it stamps the result to say it did. A condition that really was classified
still gates, so the lull stays closed even then, and a number produced that way
cannot later be mistaken for one produced under the real gate.
