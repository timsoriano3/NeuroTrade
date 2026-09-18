# Triple-barrier labelling and the cost model

§8's labelling and cost-model rows, landed together because they are one idea: a label is what a
trade earned, and §3.3 forbids computing that gross and subtracting costs afterwards.

## What exists

| Module | Contents |
|---|---|
| `core/costs.py` | `SpreadSource` (port), `FlooredSpread`, `FeeSchedule`, `TradeCost`, `CostModel`, `tick_size` |
| `lab/labelling.py` | `Label`, `BarrierTouch`, `triple_barrier`, `label_barriers`, `concurrency`, `average_uniqueness` |

## Why the cost model is in `core/` and not `lab/`

The plan put it in `lab/`. That is wrong under the layering contract: `execution/` may import
`core` and `adapters`, never `lab`, so a cost model in `lab/` would have to be duplicated for live
— and §3.6 names research/live divergence as the project's primary failure mode. §3.3's "every
signal must beat its own cost" is a live decision as much as a backtest one, and §8 recalibrates
the model nightly from real fills. It belongs in the layer both sides can reach.

## The decisions that keep labels honest

**Barriers are tested against the price path, not the close.** A stop is hit when the low trades
through it. Testing closes understates stop frequency, always flatteringly.

**When both barriers fall inside one bar, the stop wins.** The order of touches is unknowable from
OHLC. This is not a guess about what happened; it is the conservative reading, and the only
alternative that does not inflate every result. `BarrierTouch.ambiguous` makes the share countable
rather than invisible.

**The entry bar cannot trigger a barrier.** A decision made on a bar's close cannot be filled
inside that same bar. Walking from the entry bar itself is a one-bar lookahead.

**No bars after the entry yields `None`, not a timeout.** An unlabelled decision and one that
timed out are different outcomes; conflating them puts fabricated results at the most recent
dates, which is exactly where a model is most likely to be trusted.

**Costs are charged on both legs.** A round trip crosses the spread twice and pays commission
twice. `is_win` is therefore deliberately *not* `label is Label.PROFIT` — a position can touch its
target and still lose money, which is the case a cost-blind label hides.

**Prices must be split-adjusted before they arrive.** Nothing here can detect that they were not.
`core/actions.py` is what fixes it.

## The cost model's shape

Three components reported separately, because they behave differently as size grows: commission is
near-linear, spread linear, slippage super-linear. A strategy that works at 100 shares and fails at
10,000 fails through the third term, and one total would hide which.

**Spread is a port.** The corpus has no quotes, so research estimates and live passes the real
quote; real `BID_ASK` history replaces the estimator if IBKR serves it. Estimates are **floored**
at `max(estimate, one tick, bucket minimum)` and negatives are impossible by construction — high/low
estimators produce negative spreads often, and a negative spread is a subsidy for trading. The bias
is deliberately toward over-stating cost (§17).

**A costless backtest is unrepresentable.** `FlooredSpread` floors at one tick whatever its
configuration says, because nothing trades tighter than a tick. Setting every fee to zero still
costs a tick crossed twice.

**Slippage is `impact * volatility * sqrt(participation)`** — the standard square-root impact
shape. Without an `average_volume` there is no participation and slippage is zero, which is
optimistic and stated rather than hidden. Zero average volume raises rather than returning zero,
or the most illiquid names would price as the cheapest to trade.

**A maker pays zero spread, never negative.** Modelling a rebate as a cost reduction invites a
strategy whose whole edge is the modelled rebate.

## Uniqueness weighting

Overlapping labels are nearly the same observation. `average_uniqueness` weights each by the mean
of `1 / concurrency` across its span. Without it, ten labels that are really one carry ten times
the weight in any statistic, and every Sharpe and p-value downstream is overstated (§8). CPCV's
purging will use the same spans.

## Measured end to end

AAPL, 2026-09-15, 390 IBKR minute bars. A null strategy — enter long every 5th bar, 0.3% barriers,
30-bar time limit, 1,000 shares — labelled 78 of 78 entries: **14 profit, 7 stop, 57 timeout, 0
ambiguous, 32 net wins, mean net return -0.00028**.

Two things worth keeping. A signal with no edge comes out **negative after costs** rather than
near zero, which is the check that costs are genuinely inside the label. And **mean uniqueness was
0.216** — 30-bar labels started every 5 bars overlap roughly sixfold, so the effective sample is
about a fifth of the nominal 78. That is the inflation the weighting exists to remove, and it is
large enough to change any significance test computed without it.

## Not done yet

- **Barrier distances are passed in, not derived.** §8 wants ATR-scaled barriers; `features/
  indicators.py` has the ATR and nothing wires the two together.
- **No meta-labelling.** §7.3's second stage — a model on top of the label deciding whether to
  take the trade — needs the CV harness first.
- **`FeeSchedule` is not in `config/`.** Defaults are IBKR's US fixed tier in code. Twelve-factor
  says they belong in a profile; the seam is a `CostSettings` block that does not exist yet.
- **Nothing recalibrates from fills.** §8 wants it nightly; there are no fills.
- **Uniqueness is computed but unused.** It weights nothing until there is a model to weight.
