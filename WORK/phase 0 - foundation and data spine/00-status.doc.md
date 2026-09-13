# Phase 0 — status

**Deliverable (TRADER_PLAN §13):** hexagonal skeleton, plugin registries, IBKR paper
connection, Parquet/DuckDB store, historical backfill, event-sourced replay harness.
**Exit gate:** replay a full historical session deterministically; paper order round-trips.

## Gates

| Gate | Claim | Command | Verified |
|---|---|---|---|
| G1 | Session replay is bit-for-bit | `make verify-replay` | Yes — digest `4f58fe2c99cd26dc7cbb8faf033a39d1`, stable across processes with differing `PYTHONHASHSEED` |
| G2 | Paper order round-trips | `make paper-smoke` | Yes — submitted, acknowledged and cancelled against paper account `DUT108414`; the order was written to the event log with its config hash, and that log replays deterministically |

Both gates are **cleared**. The phase deliverable is **not** complete.

## Built

```
src/neurotrade/
  core/        types clock events ids intent orders position ports registry codec calendar
               universe
  adapters/
    storage/   schemas parquet_store duckdb_catalog event_store
    calendar/  venue_calendar
    universe/  universe_file
    ibkr/      connection market_data broker pacing
  features/    registry            (registry only — no features defined yet)
  strategies/  base                (contract only — no strategies defined yet)
  lab/         replay
  ingest/      backfill crawler    (queue + fetch loop, run via `ibkr backfill` / `make backfill`)
  bus.py config.py logs.py cli.py
```
~16.2k lines across `src`, `tests`, `scripts`, `config`. Tests sit beside every module.

The **trading calendar** is built in both halves and is **not in the spec at all** — §12.1 is
silent on trading hours, holidays and half days, while `missing_sessions()` in `DuckDBCatalog`
takes an expected-session list it had no source for, and `core/ports.py`, `cli.py`,
`schemas.py` and `IbkrMarketData` all defer to "the venue calendar" in comments. Nothing wires
it in yet. See `09-venue-calendar.doc.md`.

## Not built

Spec order is §12.1's corpus-build table. Stages 1–3 are Phase 0; stage 5 is Phase 1.

- **IBKR backfill crawler** (§12.1 stage 1, "week 1") — pacing-aware, resumable, runs
  continuously. `ingest/backfill.py` walks the universe against the calendar and yields the
  sessions that are short; `ingest/crawler.py` drains that queue through `MarketDataPort` into
  `StoragePort` and reports what happened; `neurotrade ibkr backfill` / `make backfill` runs it
  against IBKR. The corpus is still empty: every live attempt so far (2026-09-12, weekend) hit
  IBKR's data-farm outage — first as a hang with no timeout, then (after adding
  `request_timeout_seconds`, see `08-gotchas.doc.md`) as a clean per-cell failure — see
  `10-backfill-crawler.doc.md`. What is left is a weekday crawl with the farms up.
- **Free sample seed data** (§12.1 stage 2) — FirstRateData / Kibot samples, so the lab has
  something to work with before the crawler has run.
- **yfinance daily bars and universe history** (§12.1 stage 3) — including `.TO` tickers.

Corpus target before Phase 5: 3–5 years of 1-minute bars across US + TSX, ~2,000 symbols,
well under 100 GB compressed.

**Not Phase 0:** the corpus quality gate (gap detection, split/dividend adjustment, halt
marking, duplicate prints, survivorship audit) is §12.1 stage 5, timed **Phase 1, ongoing**.

Nothing from Phase 1+ exists: no features, no strategies, no labelling, no CPCV, no cost
model, no trial ledger, no risk engine, no execution engine, no `api/`, no `ui/`.

## The one-line summary for a new session

The skeleton and both proofs are done; there is no data in it and nothing trades yet.
