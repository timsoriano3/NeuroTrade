# lab

Where a strategy is measured before it is believed.

§3.1 puts this ahead of any trading logic: no strategy is written until we can
honestly measure one. It is easy to build a backtest that looks profitable and
isn't, and everything else in the system is downstream of being able to tell the
difference.

## Files

| File | What it does |
|---|---|
| `replay.py` | Replays a recorded session and proves the replay was faithful |
| `labelling.py` | Triple-barrier labels, with costs applied inside, plus uniqueness weights for overlapping label spans |
| `cv.py` | CPCV — every combination of test blocks, purged and embargoed — and walk-forward as the secondary check |
| `significance.py` | Whether a result survives the search that found it: PSR, deflated Sharpe, PBO via CSCV |
| `trials.py` | The ledger the deflation counts against — every hypothesis tested, by anything |

Still ahead: the exit gate — a deliberately overfit control strategy the lab
has to reject.

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

The useful part is that the digest covers **outputs as well as inputs**. Once
strategies exist, a strategy that starts deciding differently changes the digest
even though the recorded data is untouched. So "did that change alter
behaviour?" becomes a yes/no question instead of an afternoon of diffing logs.

## What the lab may depend on

Core, features, strategies — and **not** adapters. The lab talks to storage
through the interfaces in `core/ports.py`, so research runs against a file, a
fixture, or an in-memory list without changing. That rule is enforced by the
`lab-uses-ports-not-adapters` contract in `.importlinter`, and it is what moved
the event codec into `core`.
