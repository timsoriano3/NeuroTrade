# NeuroTrade

An automated day-trading system for US and Canadian stocks.

The short version of how it works: **rules decide direction, machine learning
decides conviction, and hard limits decide what is allowed.** A set of trading
strategies each propose trades. A model scores how likely each proposal is to
work. A risk engine turns that score into a position size — and can veto
anything, because risk rules are structural and no model may override them.

It runs on one machine, trades through Interactive Brokers, and learns mainly
from replaying years of historical data rather than from its own small number of
live trades.

**Status: Phase 0 of 8 — foundations.** Nothing trades yet. There is no
strategy, no model, and no broker connection. See [What exists today](#what-exists-today).

Not financial advice.

## Why it is built this way

Three ideas drive most of the design decisions:

**Prove the measurement before trusting the result.** It is easy to build a
backtest that looks profitable and isn't. So the validation tooling comes first,
and a strategy is only written once we can honestly measure one.

**The same code runs in backtest and live.** One definition of every
calculation, shared by both. When research and production drift apart, backtest
results stop being claims about the system that actually trades.

**A session must replay exactly.** Feed yesterday's recorded data back in and the
system must make the identical decisions, down to the byte. That is the only way
to tell whether a change improved things or just moved them.

## What exists today

The foundations — the vocabulary the rest of the system is written in, and the
storage it runs on. 645 tests.

| Area | What it does |
|---|---|
| **Domain model** | Prices, quantities and money as exact values; market events; orders, fills and positions; the trade proposals strategies produce |
| **Clock** | One source of time, swappable between real and simulated — the thing that makes replay possible |
| **Identifiers** | Every record's id is derived from its contents, so replaying a session produces the same ids rather than new random ones |
| **Configuration** | Three environments (research, paper, live) with a fingerprint stamped on every decision, so any trade can be traced to the exact settings that produced it |
| **Plugin registries** | Strategies and calculations register themselves by name and version, so two versions can run side by side for comparison |
| **Corpus storage** | Market data on disk as Parquet, queried with DuckDB, with tools to find what is missing |
| **Event log** | An append-only record of everything that happened, which a session can be replayed from |

Deliberately not built yet: strategies, models, the risk engine, the broker
connection, and the dashboard. Those are Phases 2 onward.

## Getting started

```bash
brew install uv
make setup
make check          # lint, typecheck, tests
```

`make help` lists everything. `make show-config PROFILE=paper` prints the
resolved settings and their fingerprint.

## Layout

One repository, several languages. Python is the trading system; Go and
TypeScript arrive in Phase 3 for the dashboard.

```
src/neurotrade/
  core/         the domain model — depends on nothing else
  adapters/     storage, and later the broker and data feeds
  features/     calculations shared by research and live
  strategies/   one module per strategy
  config.py     environment profiles
config/         profile files
```

Layers may only depend downward — `core` knows nothing about storage, brokers or
strategies. This is checked automatically on every commit, not left to
discipline: `make lint` fails if any layer reaches somewhere it should not.

## Documentation

- **[CLAUDE.md](CLAUDE.md)** — conventions, the rules code here must follow, and
  how the pieces fit together.
- **Trader Plan** (in Notion) — the full specification: strategies, the ML
  stack, validation methodology, data sources, and the phase-by-phase roadmap.
  A local copy lives at `TRADER_PLAN.md`, which is gitignored.
