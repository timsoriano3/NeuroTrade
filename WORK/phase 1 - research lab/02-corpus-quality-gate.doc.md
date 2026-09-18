# The corpus quality gate

§12.1 stage 5, minus the adjustment half (that is `01-corporate-actions.doc.md`). Six checks over
the corpus, run from one command, reporting only.

## What exists

| Module | Responsibility |
|---|---|
| `core/quality.py` | `Coverage`, `Gap`, `Duplicate`, `SuspectSession` — the vocabulary for describing a corpus |
| `core/ports.py` | `CorpusQualityPort` — `coverage`, `gaps`, `duplicate_timestamps`, `suspect_sessions` |
| `adapters/storage/duckdb_catalog.py` | Satisfies it. `suspect_sessions` is new; `duplicate_timestamps` now returns `Duplicate` rather than raw tuples |
| `ingest/quality.py` | `audit_corpus` — composes the six checks. `MissingSession`, `ShortSession`, `SurvivorshipAudit`, `CorpusReport`, `summarise` |
| `cli.py` | `neurotrade corpus check`; `make corpus-check` |

The checks: sessions the calendar held but the corpus lacks; sessions held with too few bars;
holes inside an otherwise-present session; duplicate prints; sessions that do not look like
trading; and the survivorship audit.

## The decisions

**`Coverage` and `Gap` moved from the adapter into `core`.** They are domain concepts and a port
returns them, so a concrete catalog cannot own them. `core/calendar.py` had carried a comment
about exactly this misplacement since `expected_bars` was written.

**A new `CorpusQualityPort` rather than widening `CatalogPort`.** The crawler needs one method —
bar counts per session — and every fake in its tests implements that and nothing else. Bolting
four audit queries onto the port it depends on would make each of those fakes grow methods the
crawler never calls. `DuckDBCatalog` satisfies both, structurally.

**It reports; it never repairs.** Every fault has at least two causes the corpus cannot separate:
a missing session is a failed fetch or a day the listing did not trade; a hole is a halt or a
dropped request; a flat session is a halt or an illiquid name. Writing a plausible number over a
known unknown is worse than the hole, because the hole is visible and the guess is not.

**Halt marking is done by inference, not from a halt feed.** We have none. `suspect_sessions`
reports two shapes — zero volume across a session, and a single distinct close over more than one
bar. Both read downstream as a calm, liquid instrument, which is the dangerous direction:
realised volatility collapses toward zero and anything sized off it takes an unbounded position.

**Survivorship is reported as unmeasurable, deliberately.** The universe source lists what still
trades, so a name delisted in 2023 is absent from the corpus and no scan of that corpus can find
what is not there. `SurvivorshipAudit.measurable` is False whenever the source is survivor-only,
and only the visible half is counted: names whose history starts inside the range. A floor on the
bias, never the bias.

**Survivorship never turns the gate red.** It is a standing property of a free data source, not
something a re-fetch fixes. A permanently red gate is an ignored one.

**Intra-session gap detection is skipped for daily bars.** Vacuous — a daily session holds one
bar, so there is no inside — and expensive: `gaps` is one query per instrument-session, so five
years of 43 names is ~52,000 DuckDB queries to prove nothing. The first real run exceeded ten
minutes before being killed.

## Measured

- Daily corpus (`--interval 1d --source yfinance-daily`, 2021-09-15 → 2026-09-11):
  **0 missing, 0 short, 0 gaps, 0 duplicates, 0 suspect**; 43/43 names have history.
- IBKR minute corpus (`--interval 1m --source ibkr`, 2026-09-11 → 2026-09-15):
  **7 missing sessions, 0 short, 0 gaps, 0 duplicates, 0 suspect**; 42/43 names have history. The
  missing sessions are JPM and UNH, which the first crawl did not reach — crawl progress, not a
  fault.

## Not done yet

- **No halt feed.** Inference is all there is until one exists; a `TradingHalt` event type is in
  `core/events.py` with nothing producing it.
- **`coverage` is on the port but unused by the gate.** It is there for the report to grow a
  per-source breakdown, which `Coverage.is_mixed_source` exists to answer.
- **Nothing consumes a `CorpusReport` programmatically** — the CLI prints it and exits. Whether
  the gate should block a research run is a Phase 1 decision that has not been made.
