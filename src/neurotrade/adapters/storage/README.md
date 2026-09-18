# adapters/storage

Where data lives on disk.

Two separate things share this folder, and it is worth keeping them apart:

**The corpus** — years of market data, written once and read in wide date
ranges. Columnar Parquet, queried with DuckDB.

**The event log** — everything that happened in one session, written a record at
a time and read start to finish. Newline-delimited JSON, appended to.

They look similar and are shaped by opposite needs. Appending one line to a
Parquet file means rewriting the file; scanning three years of JSON means
reading every byte of it.

## Files

| File | What it does |
|---|---|
| `schemas.py` | The on-disk column layout, and converting bars to rows and back |
| `parquet_store.py` | Reads and writes the corpus. Writes are idempotent |
| `duckdb_catalog.py` | Questions *about* the corpus: what is held, what is missing |
| `event_store.py` | The append-only session log |
| `actions_parquet.py` | `ActionStore` — the corporate-action set as one file under `derived/actions/` |

Events are turned into text by `core/codec.py`, not here. It lives in `core`
because the research lab needs it too, and a layer may not reach into an adapter
— see the `lab-uses-ports-not-adapters` contract in `.importlinter`.

## Two decisions worth knowing

**Prices are stored as exact decimals, not floats.** `100.12345678` written as a
float comes back as `100.123456779999997934…`. Every backtest computed from the
corpus would then disagree with a live run, for no reason anyone could find.

**Writes are idempotent.** The backfill crawler is resumable and re-fetches
ranges after every interruption. A duplicated bar does not look like corruption —
it looks like double the volume, which then ranks that stock top of the
watchlist.

## Layout on disk

```
raw/bars_1m/venue=NASDAQ/ticker=AAPL/session_date=2026-03-14/bars.parquet
```

Venue comes first because the same ticker trades on several venues: `TD` is
Toronto-Dominion on both TSE (in CAD) and NYSE (in USD), at different prices.
