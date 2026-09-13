# ingest

Building the corpus: deciding what to fetch. The fetching itself arrives with
the crawler loop.

The training corpus is a first-class asset with its own build plan, separate
from live market data. This package is that plan.

## Files

| File | What it does |
|---|---|
| `backfill.py` | `plan_backfill` — which instrument-sessions are missing, most recent first. `BackfillCell` is one unit of that work |

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
