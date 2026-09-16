# adapters/universe

Where the list of tradable instruments comes from.

Everything else asks `UniversePort` and gets a `Universe` back, so the trading
core never learns whether that list was hand-written, derived from yfinance, or
ranked overnight by the Universe Selector.

## Files

| File | What it does |
|---|---|
| `universe_file.py` | `UniverseFile` — reads and validates `config/universe.yaml` into a `Universe` |
| `universe_history_parquet.py` | `UniverseHistoryStore` — reads and writes the point-in-time membership artifact |

## The history artifact

`universe_history_parquet.py` persists what `ingest/universe_history.py` computes:
one Parquet file per source under `derived/universe/`, holding which instruments
were eligible on each session date.

Two details are load-bearing. A session on which the screen admitted **nobody**
still gets a row, with a null ticker — drop it and an empty session becomes
indistinguishable from one never evaluated, and `as_of` would carry the previous
membership across it. And the digest recorded at write time is **recomputed on
read**: a truncated or hand-edited artifact would quietly change which names a
backtest was allowed to trade, and nothing downstream would notice.

## Why a file, for now

§12.1 starts the IBKR backfill crawler in Phase 0 week 1, but the yfinance
universe history that would populate it is stage 3 of the same table. A crawler
with no symbol list has no work queue, so the list is written down by hand and
replaced later. The port is what makes that replacement a new file here rather
than a change anywhere else.

Membership is data rather than configuration, and deliberately outside the
config hash. It grows from a few dozen names to a few thousand, and folding it
in would churn the fingerprint stamped on every trade record each time a ticker
was added. `Universe.digest` is what a run records instead.

## Every malformed file raises

A universe silently short by one venue produces a corpus silently short by one
venue, and nothing downstream can tell that apart from a venue that genuinely
had no data. The file is small, hand-edited and read once at startup, so
strictness is cheap and lands at the only moment anyone is looking at it.

Rejected: an unknown venue name, `SMART` (an order route, not a listing), a
ticker repeated within a venue, a ticker `Symbol` itself refuses, and a file
describing no symbols at all.

## The trap worth knowing

**YAML turns some real tickers into booleans.** `ON`, `NO` and `OFF` parse as
`True`/`False` — and `ON` is ON Semiconductor, a NASDAQ listing. The loader
rejects a non-string ticker and tells you to quote it, rather than calling
`str()` on it: `str(True)` is `'True'`, which looks like a ticker, would be
crawled forever, and would never resolve.
