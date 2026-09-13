# The backfill crawler

§12.1 stage 1: "pacing-aware, resumable, runs continuously". Universe, work queue, fetch loop,
CLI command and make target all exist. The corpus is still empty — the first live attempt
timed out (see Known limitation and `08-gotchas.doc.md`).

The spec defines neither the universe nor a calendar, and the crawler needs both. The calendar
is `09-venue-calendar.doc.md`; the universe and queue are here.

## The symbol axis

`core/universe.py` — `Universe`, a frozen tuple of `Symbol`, sorted and deduplicated at
construction, with `venues`, `by_venue` and a BLAKE2b `digest`.

- **Keyed on `Symbol`, never on ticker.** TD is Toronto-Dominion on both NYSE in USD and TSE in
  CAD. One entry per ticker would average two currencies into one series.
- **Empty is rejected.** Downstream, an empty universe reads as "nothing to do", so a crawler
  would report a complete corpus having done nothing.
- **`digest` exists because the universe is data, not configuration.** Folding it into the
  config hash would churn the fingerprint stamped on every trade record; a run records the
  digest instead.

`adapters/universe/universe_file.py` — `UniverseFile` implements `UniversePort`, parsing
`config/universe.yaml` eagerly so a typo fails startup rather than the first fetch. The
committed seed file holds 43 symbols across ARCA, NASDAQ, NYSE and TSX, digest
`2ea7e3533356185d`, **unverified against IBKR's contract database** — the first crawl is what
confirms them.

## The work queue

`ingest/backfill.py` — `plan_backfill(universe, calendar, catalog, start=, end=, interval=)`
yields `BackfillCell`s, each one instrument-session-interval, carrying the session itself so the
fetch window and expected bar count come from the day's real bounds.

**Resumable with no state of its own.** The queue is recomputed from the calendar and from what
is on disk — **the corpus is its own progress record.** Writes are idempotent, so re-fetching
covered ground costs a request and changes nothing.

**Held is not complete**, which is why `CatalogPort` returns counts rather than a set of dates:
an interrupted fetch leaves a session present and short, not absent.

**Order is recency-major** — every symbol's most recent session before any symbol's older ones,
so an interrupted crawl leaves the corpus shallow across the whole universe rather than deep for
one end of the alphabet.

## The fetch loop

`ingest/crawler.py` — `crawl(universe, calendar, catalog, feed, store, *, start, end, source,
interval=, limit=, max_consecutive_failures=, on_outcome=)` drains one plan through
`MarketDataPort` into `StoragePort` and returns a `CrawlReport` (per-cell `CellOutcome`s, plus
`requests`, `bars_written`, `completed`).

- **One call is one pass: plan, drain, return.** Continuous running is the caller looping
  passes, not a loop inside `crawl` — the plan is a snapshot, and a week-long crawl should
  re-read the corpus rather than trust a queue computed days ago.
- **`order_cells` sorts untouched sessions before short ones**, stably, so recency-major order
  survives within each group. A short session may be a halt the corpus can't recognise yet
  (see Known limitation below); re-requesting it likely returns the same shortfall, so spending
  budget on never-fetched sessions first grows the corpus faster.
- **No pacer in the loop.** `MarketDataPort` requires adapters to self-pace, and
  `IbkrMarketData` already awaits its own `HistoricalPacer`. A second limiter here would count
  the same requests twice and halve throughput for nothing.
- **Fetch window is `(session.open_ns + 1, session.close_ns + 1)`.** The port's range is
  `[start, end)` on `ts_event`, but a session's bars close in `(open, close]` — a bar closing
  exactly at the bell belongs to the previous session. Shifting both bounds by one nanosecond
  asks for exactly this session and nothing either side.
- **Re-trimmed with `session.holds_bar` after fetch**, not just requested with the right
  window: a bar an adapter files under the wrong session date is invisible to the plan (which
  counts by date) and would sit in the corpus looking like a complete day.
- **Failure handling is per-symbol, then per-pass.** A bar for the wrong symbol or interval is
  `ValueError` — a broken adapter, never a missing listing. A feed exception marks the cell
  `FAILED` and skips that symbol for the rest of the pass (an unresolvable listing fails
  identically every session, so retrying it per-date buys nothing). Five consecutive failures
  end the pass — a dead Gateway, not a bad listing. Store errors and `CancelledError` (a
  `BaseException`) both propagate rather than being recorded, since a local fault (full disk,
  killed process) would hit every later cell too.
- **`source: str`, not the storage `Source` enum** — that enum lives in the adapter, and
  `ingest/` may only import `core`.
- **No clock, no state.** Nothing here reads the time or remembers what it did; the next pass's
  plan observes whatever the store now holds.

## Known limitation, by design

A session that is *permanently* short — a symbol halted for the afternoon, or listed midway
through the day — never reaches its expected count and is re-offered on every pass. One request
per pass, so it blocks nothing, but it never resolves. The missing fact is halt marking, which
is §12.1 stage 5 and Phase 1. A completeness tolerance was rejected as the fix: it would
silently accept short data everywhere to paper over one case.

## The command: `neurotrade ibkr backfill` / `make backfill`

`cli.py`'s `ibkr_backfill` is the only place the concrete adapters (IBKR feed, `ParquetStore`,
`DuckDBCatalog`, `UniverseFile`, `VenueCalendar`) meet `ingest/`; `ingest/` itself stays
core-only. Resolved the doc's open question: **`--start` is required, no default** — a default
would silently decide how much history the corpus holds, and it isn't config since the window
is data, not settings. `--end` defaults to yesterday in UTC, via `LiveClock` — UTC yesterday is
never later than venue-local yesterday, so the default can never store a still-open session
short.

`--passes N` loops `crawl`, reusing one `VenueCalendar` across the loop (building one is the
expensive part), and stops early when a pass plans zero cells or gives up. Detecting an empty
plan costs one extra plan-only pass — no network requests — which was accepted as simpler than
threading a "would this pass be empty" check through `crawl`.

Connects once before the loop rather than lazily on first request: a lazy connect turns a dead
Gateway into five `FAILED` cells instead of the one `IbkrConnectionError` it is (exit 1).

Bars written (summed across all passes) go to stdout, for scripting; per-cell outcomes go to
stderr, one line per cell, so a crawl measured in hours can be watched live. Exit is non-zero if
the last pass did not complete. `backfill_pass_complete` is logged after every pass and carries
the universe digest, since a corpus is only explainable if each pass says which symbol list it
was filling.

```bash
make backfill START=2026-08-01                    # END defaults to yesterday, UTC
make backfill START=2026-08-01 LIMIT=50 PASSES=3
```

## Known limitation, live

The first live attempt (2026-09-12, weekend) found `ibkr check` healthy against `DUT108414` but
`backfill` hung for 5+ minutes with zero bars: `qualifyContractsAsync` timed out at the raw
`ib_async` layer. A healthy probe means the socket and login work, not that IBKR's historical
data farms are serving — see the gotcha. `IbkrMarketData` now has a per-request timeout (fixed,
see `08-gotchas.doc.md`); a second live attempt the same day (farms still down, timeout set to
10s) failed 5 cells cleanly with "resolving X went unanswered for 10.0s" and stopped the pass
(exit 1) instead of hanging. Corpus is still empty — the crawl still needs a weekday with the
farms up.

## Not done

- **The first successful live crawl** — every attempt so far has hit the weekend data-farm
  outage; the timeout now fails those cells cleanly instead of hanging, but no bars have landed.
- **Stages 2 and 3** — FirstRateData and Kibot samples, then yfinance daily bars and universe
  history. `Source` already has values for all three.

## Verified

`make check` PASS, 1014 tests; `invariant-auditor` found nothing; `docs-drift-auditor` found
nothing. **Not verified live:** the 43-symbol universe against IBKR's contract database — every
live attempt so far has hit the weekend data-farm outage before qualifying anything, so it
neither confirms nor refutes the seed file.
