# ingest

Building the corpus: deciding what to fetch, then fetching it.

The training corpus is a first-class asset with its own build plan, separate
from live market data. This package is that plan.

## Files

| File | What it does |
|---|---|
| `backfill.py` | `plan_backfill` — which instrument-sessions are missing, most recent first. `BackfillCell` is one unit of that work |
| `crawler.py` | `crawl` — one pass: plan, fetch each missing session from the feed, write it to the store. `CrawlReport` says what the pass did |
| `universe_history.py` | `screen_universe` — who was tradable on each past session, from the daily corpus. `ScreenRules` and `LiquidityFloor` are the thresholds |
| `actions.py` | `fetch_actions` — collect every instrument's splits and dividends. `scan_gaps` — audit the daily corpus for overnight moves those actions do not explain |
| `quality.py` | `audit_corpus` — the §12.1 stage 5 gate: missing and short sessions, intra-session holes, duplicate prints, halted-looking sessions, survivorship. Reports; never repairs |

Wired to real adapters — Parquet store, DuckDB catalog, venue calendar,
universe file, and a feed — in `cli.py`, which is the only place those
concretes meet this package. **Three commands drive the same crawl**, differing
only in which feed, which interval and which dataset root they hand it:

```bash
make backfill START=2026-08-01             # IBKR history -> raw/bars, 1m
make backfill START=2026-08-01 LIMIT=50 PASSES=3
make seed                                  # vendor samples -> derived/seed/<source>, 1m
make daily START=2021-09-15                # Yahoo -> derived/daily/yfinance, 1d
```

`universe_history.py` is the exception: it consumes the corpus rather than
filling it. `make universe START=2022-01-03` reads `derived/daily/yfinance` and
writes point-in-time membership to `derived/universe/yfinance`. Its one rule
worth restating here is that the trailing window ends at **t-1** — a bar is
stamped at its session close, so the session being decided cannot vote on
itself without leaking the future into the decision.

That the seed and Yahoo feeds reuse this crawler rather than parsing into the
corpus themselves is the point: one calendar trim, one resumability rule, one outcome
report, whatever the data's origin (§3.6).

## Why it depends on core alone

Everything here goes through the interfaces in `core/ports.py`: the calendar,
the catalog, the feed, the store. Nothing imports the IBKR adapter or the
Parquet writer, and `.importlinter` enforces it.

The reason is testability, and it is not abstract. The crawler is specified to
run for weeks. If its loop could only be exercised against a logged-in Gateway,
it would be the least tested code in the repo and the code with the most
opportunity to fail quietly.

## Resumability without state

There is no checkpoint file. The queue is recomputed from two facts that already
exist: which days each venue traded, and how many bars are on disk for each of
them. **The corpus is its own progress record.**

That works because writes are idempotent, so re-fetching ground already covered
costs a request and changes nothing. A crawler killed mid-session resumes by
noticing that the session is short. A checkpoint that disagreed with the data on
disk would be worse than having none.

## Held is not complete

An interrupted fetch leaves a session holding some of its bars. `CatalogPort`
returns bar *counts* rather than a set of dates for exactly this reason, and the
comparison against `TradingSession.expected_bars` is what turns a count into a
decision. A crawler that treated presence as completeness would leave every
interrupted session permanently short.

Known limitation: a session that is *permanently* short — a symbol halted for
the afternoon, or one that listed midway through the day — never reaches its
expected count, so it is re-offered on every pass. One request per pass, so it
blocks nothing, but it never resolves either. Halt marking is part of the corpus
quality gate in Phase 1, and that is the missing fact.

## Order is recency-major

Sessions are offered newest first, and every symbol's most recent session comes
before any symbol's older ones. An interrupted crawl then leaves a corpus that is
shallow across the whole universe rather than deep for the alphabetically early
part of it — a cross-sectional study can use the former and cannot use the
latter.

## One pass, then re-plan

`crawl` plans once, drains that plan and returns. Continuous running is a loop
of passes, because the plan is a snapshot of the corpus and a crawl measured in
weeks should re-read it rather than trust a queue computed days earlier.

Within a pass, never-fetched sessions go ahead of short ones: a short session
may be a halt, and re-requesting it mostly returns the same shortfall.

Pacing is the feed's job, not the loop's — `MarketDataPort` requires adapters
to pace themselves, and the IBKR adapter waits on its own limiter. A symbol
whose request fails is skipped for the rest of the pass, and a run of
consecutive failures ends the pass, since that pattern is a dead connection
rather than a bad listing.
