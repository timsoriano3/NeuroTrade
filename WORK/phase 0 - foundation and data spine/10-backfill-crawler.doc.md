# The backfill crawler

§12.1 stage 1: "pacing-aware, resumable, runs continuously". **Two of the three halves exist —
what to fetch. Nothing fetches yet.** The loop, a CLI command and a make target are open.

The spec defines neither the universe nor a calendar, and the crawler needs both. The calendar
is `09-venue-calendar.doc.md`; the universe is here.

## The symbol axis

`core/universe.py` — `Universe`, a frozen tuple of `Symbol`, sorted and deduplicated at
construction, with `venues`, `by_venue` and a BLAKE2b `digest`.

- **Keyed on `Symbol`, never on ticker.** TD is Toronto-Dominion on both NYSE in USD and TSE in
  CAD. One entry per ticker would average two currencies into one series.
- **Empty is rejected.** Downstream, an empty universe reads as "nothing to do", so a crawler
  would report a complete corpus having done nothing. Same refusal as the calendar's for an
  out-of-horizon date, for the same reason.
- **`digest` exists because the universe is data, not configuration.** It grows from tens of
  names to thousands; folding it into the config hash would churn the fingerprint stamped on
  every trade record. A run records the digest instead. `sorted(set(...))` at construction is
  what makes it reproducible.

`adapters/universe/universe_file.py` — `UniverseFile` implements `UniversePort`, parsing
`config/universe.yaml` eagerly so a typo fails startup rather than the first fetch. Every
malformed file raises: unknown venue, `SMART`, a repeated ticker, a ticker `Symbol` refuses, a
wrong version, no symbols at all.

The committed seed file holds **43 symbols across ARCA, NASDAQ, NYSE and TSX**, digest
`2ea7e3533356185d`. Its venues are unverified against IBKR's contract database; the first crawl
is what confirms them. `UniversePort` exists rather than a bare loader because the *source*
changes by phase — a file now, yfinance history at stage 3, §5's Universe Selector later —
while the value returned does not.

## The work queue

`ingest/backfill.py` — `plan_backfill(universe, calendar, catalog, start=, end=, interval=)`
yields `BackfillCell`s, each one instrument-session-interval, carrying the session itself so the
fetch window and the expected bar count come from the day's real bounds.

**Resumable with no state of its own.** The queue is recomputed from the calendar and from what
is on disk, so **the corpus is its own progress record**. Writes are idempotent, so re-fetching
covered ground costs a request and changes nothing, and a process killed mid-session resumes by
noticing the session is short. A checkpoint file that disagreed with the data would be worse
than none.

**Held is not complete**, which is why `CatalogPort` returns counts. `DuckDBCatalog.bar_counts`
is one grouped query per symbol, not per session. `missing_sessions` is deliberately *not* used:
it answers which expected dates are absent, and an interrupted fetch leaves a session present
and short.

**Order is recency-major** — every symbol's most recent session before any symbol's older ones.
An interrupted crawl then leaves a corpus shallow across the whole universe rather than deep for
one end of the alphabet, and a cross-sectional study can use the first and not the second.

`ingest/` depends on **core alone**, through ports, with its own `.importlinter` contract. The
crawler is specified to run for weeks; if its loop could only run against a logged-in Gateway it
would be the least tested code in the repo.

## Known limitation, by design

A session that is *permanently* short — a symbol halted for the afternoon, or listed midway
through the day — never reaches its expected count and is re-offered on every pass. One request
per pass, so it blocks nothing, but it never resolves. The missing fact is halt marking, which
is §12.1 stage 5 and Phase 1. A completeness tolerance was rejected as the fix: it would
silently accept short data everywhere to paper over one case.

## Not done

- **The loop.** Drain the queue through `HistoricalPacer` into `ParquetStore`, via
  `MarketDataPort` and `StoragePort`. `BackfillCell.is_untouched` is there so the loop can
  prefer never-fetched sessions over merely short ones.
- **CLI command, make target, and the history window.** Open question: a declared value in
  `base.yaml`, or a required argument with no default. A default silently decides how much data
  the corpus has.
- **Stages 2 and 3** — FirstRateData and Kibot samples, then yfinance daily bars and universe
  history. `Source` already has values for all three.

## Verified

`make check` 961 tests, `lint-imports` 7 contracts kept, `make docs-check` 33 documents.
Traps found building this are in `08-gotchas.doc.md` under Tooling.
