# lab

Where a strategy is measured before it is believed.

§3.1 puts this ahead of any trading logic: no strategy is written until we can
honestly measure one. It is easy to build a backtest that looks profitable and
isn't, and everything else in the system is downstream of being able to tell the
difference.

## Files

| File | What it does |
|---|---|
| `drive.py` | The loop both runners share: clock, bus, run digest |
| `replay.py` | Replays a recorded session and proves the replay was faithful |
| `feed.py` | The corpus as one ts-ordered bar stream, merged across a universe |
| `engine.py` | Runs strategies over that stream and collects their intents |
| `labelling.py` | Triple-barrier labels, with costs applied inside, plus uniqueness weights for overlapping label spans |
| `cv.py` | CPCV — every combination of test blocks, purged and embargoed — and walk-forward as the secondary check |
| `significance.py` | Whether a result survives the search that found it: PSR, deflated Sharpe, PBO via CSCV |
| `trials.py` | The ledger the deflation counts against — every hypothesis tested, by anything |
| `controls.py` | Two strategies and two synthetic series whose verdicts are known before the lab sees them |
| `gate.py` | Gate G3 — runs both controls and fails if either verdict comes back wrong |

## Why three modules and not one

They answer three different questions, and the order matters.

`cv.py` produces a **distribution** instead of a number: `C(N, k)` splits
assemble into `C(N, k)·k/N` complete backtest paths, each trained on a
different combination of the sample. One backtest cannot tell a good strategy
from a lucky one; a spread across paths can.

`significance.py` asks what that distribution is worth **given how hard we
looked**. The best of 500 tries posts a fine Sharpe ratio even when all 500 are
worthless, so the deflated Sharpe sets the hurdle at what the search itself
would have produced from noise, and PBO asks the blunter question of whether
picking the in-sample winner beats picking at random.

`trials.py` supplies the number of tries. It has to be persistent and
append-only, because the count that matters includes the searches nobody kept —
and a count assembled from the results we chose to remember is exactly the
count that makes every result look significant.

**The cost model is not here** — it is `core/costs.py`. Costs are applied inside
the backtest (§3.3), but the live engine needs the same numbers to decide
whether a signal beats its own cost, and `execution/` cannot import `lab/`. One
implementation, in the layer both sides reach.

## Replay, and why the digest matters

Feed a session's recorded events back through the same bus the live system uses,
under a simulated clock, and you must get the identical result — every time, on
every machine.

The proof is a **run digest**: a rolling hash of every event dispatched, in
order. Two runs agree only if they saw the same events, in the same order, with
the same contents down to the last decimal.

```bash
make verify-replay          # replays twice, compares digests
```

## One loop, two sources

A replay and a backtest are the same machine fed from different places. The loop
lives in `drive.py`; `replay.py` points it at a recorded event log and
`engine.py` points it at the corpus through `feed.py`. Nothing else differs —
same bus, same `SimClock`, same digest.

That is §3.6's one-implementation rule turned on the lab itself. Two loops could
drift in the way that matters most: a difference in when the clock moves, or in
what gets hashed, would make a backtest result unreproducible by a replay of the
same session. `test_drive.py` pins it — the same bars through both engines must
produce the same digest.

Merging is ordered by `(ts_event, seq, symbol)` rather than time alone. At
one-minute bars every instrument closes on the same tick, so a time-only sort
would leave the order of a tie to whichever iterator the heap popped first, and
determinism would fail every minute of every session.

**Intents are published, not returned.** A strategy's proposals go back on the
bus, so they enter the digest next to the bars that caused them — a strategy
that changes its mind changes the digest even when the input data is identical.

**What does not happen here:** fills. An intent is a proposal (§5.1); what
becomes of it is `labelling.py`'s triple barrier, not a fill simulator. One
place decides how a position resolves, and it is the place the labels come from.

What a strategy sees is assembled elsewhere — `strategies/context.py` turns the
venue calendar and the feature library into one view per strategy per bar. The
engine's own default, `NullContext`, grants nothing: no features and no
classified regime. Since `Strategy.regimes` defaults to `()` and
`Regime.UNKNOWN` grants nothing, a real strategy does not fire under it at all.
That is the safe direction to fail, and it is why a run wired that way coming
back empty is correct rather than broken.

**Research runs before the regime classifier exists.** §5.7's HMM lands in Phase
5, so every regime a classifier would name is unclassified until then and
nothing declaring a real one would ever fire. `BacktestEngine(ungated=True)`
treats *unclassified* — and only unclassified — as permissive, and the result
carries `regime_gated=False` so the number cannot later be read as though the
gate had been on. A regime that really was classified still gates, which keeps
the midday liquidity lull a no-trade window even in a research run.

**Features are warm only if you warm them.** `run(start, end, warmup_ns=...)`
reads the span before `start` into the feature history without dispatching it,
so nothing in it reaches the bus or the digest. Without that, every feature is
cold for its first `lookback` bars and the open-of-session strategies of §5.2
are silent exactly where they are supposed to trade.

The useful part is that the digest covers **outputs as well as inputs**. Once
strategies exist, a strategy that starts deciding differently changes the digest
even though the recorded data is untouched. So "did that change alter
behaviour?" becomes a yes/no question instead of an afternoon of diffing logs.

## The exit gate, and why there are two controls

§13's exit gate for Phase 1 asks for one thing: *the lab correctly rejects a
deliberately overfit control strategy*. Taken literally that is passed by a lab
that rejects everything, and a harness that never says yes is indistinguishable
from a working one right up until it throws away a real strategy — which files
no complaint. So `gate.py` runs two controls, and the verdicts have to differ.

```bash
make verify-lab                 # both controls, the default seed
make verify-lab SEED=7          # any other
```

| Control | Data | Search | Must be |
|---|---|---|---|
| **overfit** | random walk, drift removed | 70 crossover variants, both directions | REJECTED |
| **honest** | planted persistent drift | 4 variants, declared a priori | ACCEPTED |

Both travel the identical path — `Strategy.on_bar` → intents → triple-barrier
labels with costs inside → trial ledger → CPCV → deflated Sharpe. Only the data
and the size of the search differ, which is what makes the comparison mean
something.

Two details in the construction carry most of the weight:

**The walk is demeaned, not merely zero-drift in expectation.** Any one
realisation finishes somewhere, and a directional rule aligned with where it
finished wins in every subsample of it. That is a *stable* ranking, so the
overfitting statistics report a clean result and are right to.

**The snooped grid holds each variant and its mirror.** Thirty-five window pairs
on one series are highly correlated, so the best of them is barely luckier than
the median. Pairing each with a "fade" variant whose returns are its negative
lets the search pick a *sign* by luck, which is the textbook snoop.

## PBO is reported here, and does not vote

The verdict rests on the deflated Sharpe alone. Over ten seeds, DSR ran
0.000–0.220 on the overfit control against 1.000 on the honest one. PBO over the
same seeds ran 0.000–0.886 and 0.057–0.529 — two distributions that almost
entirely overlap, so any threshold placed between them would be fitted to
whichever seed happened to be tried first.

That is CSCV behaving correctly, not a bug in it. PBO is a property of the
**search**, not of the winning strategy. The snooped grid carries a systematic
ranking that has nothing to do with luck — turnover differs by window pair, so
cost does too, and a cost-driven ranking is stable across every subsample. And
the honest control's four near-equivalent rules on one series are close to
arbitrary to rank, so PBO lands near 0.5 whatever the strategy is worth.

The number is still printed, because it is worth seeing and §8 asks for it. It
is simply not the instrument that separates these two controls.

## What the lab may depend on

Core, features, strategies — and **not** adapters. The lab talks to storage
through the interfaces in `core/ports.py`, so research runs against a file, a
fixture, or an in-memory list without changing. That rule is enforced by the
`lab-uses-ports-not-adapters` contract in `.importlinter`, and it is what moved
the event codec into `core`.
