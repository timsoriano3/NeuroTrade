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

## Market conditions gate what may run

Trend days, choppy days and reversal days reward different tools. Each strategy
declares which conditions it is allowed to fire in, and the host enforces it — so
a breakout strategy and a fade-the-breakout strategy can never be live at the
same time.
