# WORK — what has been built, and where it is written down

Session-context ledger. Read this first; then open **only** the `.doc.md` you need.
Injected automatically at session start, so it must stay short.

## Project status

| Phase | Title | State |
|---|---|---|
| 0 | Foundation & data spine | **Gates cleared, deliverables incomplete** |
| 1 | Research Lab | Not started |
| 2–8 | — | Not started |

Phase 0 exit gates, both verified:
- **G1** — a session replays bit-for-bit. `make verify-replay` → `4f58fe2c99cd26dc7cbb8faf033a39d1`
- **G2** — a paper order round-trips against account `DUT108414`. `make paper-smoke`

Phase 0 work still open (§12.1): the backfill crawler's **CLI command and make target**, free
sample seed data, and yfinance daily bars plus universe history. Universe, calendar, work queue
and fetch loop all exist — none of it in the spec — but nothing has run against a live Gateway,
so the corpus is still empty. See `10-backfill-crawler.doc.md`. The corpus quality gate is
**Phase 1**, not Phase 0.

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
| `08-gotchas.doc.md` | **Bugs already paid for.** Non-obvious traps with their fixes | **Before writing code — highest value per token** |

## Why the working agreements exist

`cost-and-delegation.doc.md` — the measurements behind the delegation, batching, model-tier and
clearing rules in `CLAUDE.md`, and the hooks that enforce them. Read it before changing any of them.

## Conventions

- `TRADER_PLAN.md` (gitignored) is the spec. **Grep for the section, `sed` that range — never read the whole file.**
- `CLAUDE.md` holds the working agreements and invariants.
- These docs describe **what is true now**, not a changelog. Rewrite them in place; git holds the history.
