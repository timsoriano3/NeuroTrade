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

**Status: Phase 0 of 8 — foundations.** Nothing trades yet: there is no strategy
and no model. The broker connection is built and a paper order round-trips, but
there is nothing yet deciding what to send it. See [What exists today](#what-exists-today).

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
to tell whether a change improved things or just moved them. This one already
works — `make verify-replay` replays a recorded session twice and compares.

## What exists today

The foundations — the vocabulary the rest of the system is written in, the
storage it runs on, the machinery that proves a session replays exactly, and the
broker connection it trades through. 810 tests.

| Area | What it does |
|---|---|
| **Domain model** | Prices, quantities and money as exact values; market events; orders, fills and positions; the trade proposals strategies produce |
| **Clock** | One source of time, swappable between real and simulated — the thing that makes replay possible |
| **Identifiers** | Every record's id is derived from its contents, so replaying a session produces the same ids rather than new random ones |
| **Configuration** | Three environments (research, paper, live) with a fingerprint stamped on every decision, so any trade can be traced to the exact settings that produced it |
| **Plugin registries** | Strategies and calculations register themselves by name and version, so two versions can run side by side for comparison |
| **Corpus storage** | Market data on disk as Parquet, queried with DuckDB, with tools to find what is missing |
| **Event log** | An append-only record of everything that happened, which a session can be replayed from |
| **Event bus** | Delivers events to whatever is listening, in a fixed order — the reason two runs behave identically |
| **Replay** | Re-runs a recorded session and proves it behaved the same, by hashing everything that happened |
| **Broker** | Connects to Interactive Brokers, pulls historical bars within their rate limits, and places orders — with a structural guard that refuses real ones outside the live profile |

Still to finish in Phase 0: a command that runs the backfill crawler, a free
seed data feed, and yfinance daily bars with universe history. The crawler's
fetch loop exists but has not been run, so the corpus is currently empty.

Deliberately not built yet: strategies, models, the risk engine and the
dashboard. Those are Phases 2 onward.

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
  adapters/     storage, the IBKR broker, the venue calendar, the universe
  features/     calculations shared by research and live
  strategies/   one module per strategy
  lab/          measuring a strategy honestly; replay lives here
  ingest/       building the corpus — which sessions are still missing
  bus.py        delivers events to subscribers
  config.py     environment profiles
  logs.py       structured logging
  cli.py        the `neurotrade` command
config/         profile files, and the seed universe
scripts/        one-off tools, not part of the package
tests/fixtures/ a recorded session, replayed by CI
```

Each folder has its own README explaining what is in it and the rules that apply
there.

Layers may only depend downward — `core` knows nothing about storage, brokers or
strategies. This is checked automatically on every commit, not left to
discipline: `make lint` fails if any layer reaches somewhere it should not.

## Documentation

- **[CLAUDE.md](CLAUDE.md)** — conventions, the rules code here must follow, and
  how the pieces fit together.
- **Trader Plan** (in Notion) — the full specification: strategies, the ML
  stack, validation methodology, data sources, and the phase-by-phase roadmap.
  A local copy lives at `TRADER_PLAN.md`, which is gitignored.
