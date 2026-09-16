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
    feeds/     seed_sources vendor_download firstrate kibot   (§12.1 stage 2 samples)
    ibkr/      connection market_data broker pacing
  features/    registry            (registry only — no features defined yet)
  strategies/  base                (contract only — no strategies defined yet)
  lab/         replay
  ingest/      backfill crawler    (queue + fetch loop, run via `ibkr backfill` / `make backfill`)
  bus.py config.py logs.py cli.py
```
~21.4k lines across `src`, `tests`, `scripts`, `config`. Tests sit beside every module.

The **trading calendar** is built in both halves and is **not in the spec at all** — §12.1 is
silent on trading hours, holidays and half days, while `missing_sessions()` in `DuckDBCatalog`
takes an expected-session list it had no source for, and `core/ports.py`, `cli.py`,
`schemas.py` and `IbkrMarketData` all defer to "the venue calendar" in comments. Nothing wires
it in yet. See `09-venue-calendar.doc.md`.

## Not built

Spec order is §12.1's corpus-build table. Stages 1–3 are Phase 0; stage 5 is Phase 1.

- **IBKR backfill crawler** (§12.1 stage 1, "week 1") — **crawling.** `ingest/backfill.py` walks
  the universe against the calendar and yields the sessions that are short; `ingest/crawler.py`
  drains that queue through `MarketDataPort` into `StoragePort`; `make backfill` runs it. The
  weekend data-farm outage that blocked every earlier attempt is gone: on 2026-09-15 (a weekday,
  farms up) the first live crawls filled **47,580 minute bars over 41 instruments, 2026-09-11 →
  2026-09-15**, in `raw/bars/`. The pacer throttled as designed — 55 requests, then a 600s wait.
  One cell failed on a VWAP outside its own bar, since fixed (`08-gotchas.doc.md`). What is left
  is depth: at 55 requests per 10 minutes, the §12.1 target is weeks of crawling, which is what
  the spec expects. See `10-backfill-crawler.doc.md`.
- ~~**Free sample seed data** (§12.1 stage 2)~~ — **done.** `make seed` fetches both vendors and
  ingests them through the same crawler; 1,019,421 bars are in `derived/seed/` as of 2026-09-15.
  FRD's sample is now known to be **unadjusted** (checked against yfinance, 2026-09-15); the
  Kibot-vs-IBKR overlap check still waits on a weekday IBKR crawl. See `12-seed-feeds.doc.md`.
- **yfinance daily bars** (§12.1 stage 3, first half) — **done.** `make daily` crawls Yahoo
  through the same crawler at `1d`; 53,879 unadjusted bars over 43 instruments, 2021-09-15 →
  2026-09-11, in `derived/daily/yfinance/`, `.TO` lines included. See `13-daily-bars.doc.md`.
- ~~**Universe history** (§12.1 stage 3, second half)~~ — **done.** `make universe` screens the
  daily corpus into point-in-time membership: 1,202 sessions, 2022-01-03 → 2026-09-11, 14–43
  members of 43, digest `ed83640bfe2ef898`, in `derived/universe/yfinance/`. The trailing window
  ends at `t-1`, floors are per-currency, and the artifact carries a survivorship-bias flag
  because yfinance lists only surviving names. See `14-universe-history.doc.md`.

**§12.1 stages 1–3 are all now delivered.** What Phase 0 still owes is corpus *depth*, which is
crawl time rather than code.

Corpus target before Phase 5: 3–5 years of 1-minute bars across US + TSX, ~2,000 symbols,
well under 100 GB compressed.

**Not Phase 0:** the corpus quality gate (gap detection, split/dividend adjustment, halt
marking, duplicate prints, survivorship audit) is §12.1 stage 5, timed **Phase 1, ongoing**.

Nothing from Phase 1+ exists: no features, no strategies, no labelling, no CPCV, no cost
model, no trial ledger, no risk engine, no execution engine, no `api/`, no `ui/`.

## The one-line summary for a new session

The skeleton and both proofs are done; the corpus holds a million vendor-sample bars, nothing from
IBKR yet, and nothing trades.
