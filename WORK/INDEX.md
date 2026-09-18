# WORK — what has been built, and where it is written down

Session-context ledger. Read this first; then open **only** the `.doc.md` you need.
Injected automatically at session start, so it must stay short.

## Project status

| Phase | Title | State |
|---|---|---|
| 0 | Foundation & data spine | **Gates cleared, deliverables incomplete** |
| 1 | Research Lab | **In progress** — stage 5, features, labelling + costs |
| 2–8 | — | Not started |

Phase 0 exit gates, both verified:
- **G1** — a session replays bit-for-bit. `make verify-replay` → `4f58fe2c99cd26dc7cbb8faf033a39d1`
- **G2** — a paper order round-trips against account `DUT108414`. `make paper-smoke`

Corporate actions: 43 symbols, 18 splits, 1,588 dividends (`derived/actions/yfinance/`); the
daily corpus audits to 4 gaps, all real news moves. **§12.1 stages 1–3 delivered** (2026-09-15). What Phase 0 still owes is corpus *depth* —
crawl time, not code. Corpus now: IBKR minute bars 47,580 over 41 names (`raw/bars/`, the weekday
crawl works; 55 requests per 10 min means weeks to target); seed samples 1,019,421
(`derived/seed/`); daily bars 53,879 over 43 names, 2021-09-15 → 2026-09-11; universe history
1,202 sessions, digest `ed83640bfe2ef898`, survivorship-flagged. Quality gate is **Phase 1**.

## Phase 0 — foundation and data spine

| Doc | Covers | Read it when |
|---|---|---|
| `00-status.doc.md` | Gates, what exists, what does not | Starting any session |
| `01-domain-model.doc.md` | `core/` — types, clock, events, ids, intent, orders, position, universe, ports | Touching domain code |
| `02-config-logging-cli.doc.md` | `config.py`, `logs.py`, `cli.py`, profiles, config hash | Adding a setting or a command |
| `03-plugin-registries.doc.md` | `core/registry`, `features/registry`, `strategies/base` | Adding a feature or strategy |
| `04-storage-and-corpus.doc.md` | `schemas`, `parquet_store`, `duckdb_catalog`, `event_store`, `codec` | Touching data on disk |
| `05-bus-and-replay.doc.md` | `bus.py`, `lab/replay.py`, how G1 is proven | Anything determinism-related |
| `06-ibkr-adapter.doc.md` | `connection`, `market_data`, `broker`, `pacing`, how G2 is proven | Touching the broker |
| `07-tooling-ci-docs.doc.md` | Makefile, pre-commit, CI, import-linter, `docs-check` | Changing the build |
| `09-venue-calendar.doc.md` | `core/calendar.py`, `adapters/calendar/`, why the spec has no calendar | Touching sessions or the crawler |
| `10-backfill-crawler.doc.md` | `core/universe.py`, `adapters/universe/`, `ingest/backfill.py` | Building the corpus |
| `11-seed-data.plan.md` | **Plan** for §12.1 stage 2 — vendor facts, where files live, commit sequence | Starting seed-data work |
| `12-seed-feeds.doc.md` | `adapters/feeds/` + `neurotrade seed` — the sample feeds, the commands, what is in the corpus | Touching seed data |
| `13-daily-bars.doc.md` | `adapters/feeds/yfinance_daily.py` + `neurotrade daily backfill` — daily bars from Yahoo, and why they are unadjusted | Touching daily bars |
| `14-universe-history.doc.md` | `ingest/universe_history.py` + `neurotrade universe build` — point-in-time membership, the PIT rule, the bias flag | Touching the universe or anything point-in-time |
| `08-gotchas.doc.md` | **Bugs already paid for.** Non-obvious traps with their fixes | **Before writing code — highest value per token** |

## Phase 1 — Research Lab

| Doc | Covers | Read it when |
|---|---|---|
| `00-phase-1.plan.md` | **Plan** — the eight commits, the spread decision, why adjustment came first | Starting any Phase 1 work |
| `01-corporate-actions.doc.md` | `core/actions.py`, `adapters/feeds/yfinance_actions.py`, `ActionStore`, `ingest/actions.py`, the two CLI commands | Touching adjustment, splits or dividends |
| `02-corpus-quality-gate.doc.md` | `core/quality.py`, `CorpusQualityPort`, `ingest/quality.py`, `neurotrade corpus check` | Auditing the corpus, or adding a check |
| `03-feature-library.doc.md` | `features/indicators.py` and `features/levels.py` — the first features, and why levels sit outside the registry | Adding a feature |
| `04-labelling-and-costs.doc.md` | `lab/labelling.py` and `core/costs.py` — triple barriers, uniqueness weights, spread/commission/slippage | Labelling, or anything about what a trade costs |
| `08-gotchas.doc.md` | **Bugs already paid for.** Yahoo's hidden split adjustment, Decimal notation traps | **Before writing code** |

## Why the working agreements exist

`cost-and-delegation.doc.md` — the measurements behind the batching, model-tier and clearing rules
in `CLAUDE.md`, the hooks that enforce them, and why subagents were retired. Read it before
changing any of them.

## Conventions

- `TRADER_PLAN.md` (gitignored) is the spec. **Grep for the section, `sed` that range — never read the whole file.**
- `CLAUDE.md` holds the working agreements and invariants.
- These docs describe **what is true now**, not a changelog. Rewrite them in place; git holds the history.
