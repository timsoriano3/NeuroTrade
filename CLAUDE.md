# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

NeuroTrade — an autonomous day-trading system for US and Canadian equities. Rules produce trade
direction; ML produces conviction and size; hard risk limits are structural and unlearnable.

**`TRADER_PLAN.md` (gitignored, local copy of the Notion source of truth) is the spec.** It holds the
strategy arsenal, ML stack, validation methodology, data plan, phased roadmap and success metrics.
Read the relevant section before implementing anything in that area. This file covers only what
`TRADER_PLAN.md` does not: how to work in the repo.

**Status: Phase 0 (foundation & data spine).** Much of what follows describes the contract Phase 0
establishes, not code that exists yet.

## Commands

`make` is the single entry point across all three languages.

Go and TypeScript targets exist but report SKIP until `api/` and `ui/` have code (Phase 3).

```
make doctor       # report which toolchains are present
make setup        # install dependencies for every present toolchain
make check        # lint + typecheck + tests — run before declaring work done
make test / lint / typecheck / fmt
make up / down    # docker compose stack (postgres+timescale, redis, crawler)

make crawl                 # start the IBKR backfill crawler
make crawl-status          # coverage %, ETA, pacing headroom
make replay SESSION=<date> # deterministic session replay; prints run digest
make data-audit            # corpus quality gate
make ibkr-check            # IBKR connection health probe
make paper-smoke           # paper order round-trip
```

Single test: `uv run pytest tests/path/test_x.py::test_name -x`

## Architecture

**Hexagonal + event-sourced.** The trading core knows nothing about IBKR, Postgres or Parquet — all
external systems sit behind ports in `core/ports.py`. Every market event, signal, intent, order and
fill is an append-only record, so any session replays bit-for-bit.

**Layering is enforced by import-linter in CI, not convention.** Adding an import that violates this
fails the build:

```
core/        depends on nothing
features/    core
strategies/  core, features
risk/        core
ml/          core, features
lab/         core, features, strategies, ml
execution/   core, adapters
discovery/   lab            — and NOTHING imports discovery/
promotion/   core, ml, lab
api/    (Go) reads Redis Streams only; imports no Python package
ui/     (TS) generated OpenAPI types only
```

`discovery/` being import-isolated is what physically prevents experimental logic from reaching live
capital. Never add an import into it.

**Polyglot boundaries.** Python is the trading system (single `neurotrade` package under `src/`).
Go is `api/` — the REST + WebSocket gateway, its own module, talks to Python only via Redis Streams.
TypeScript is `ui/` — React dashboard, consumes generated OpenAPI types only. Each keeps its native
toolchain; CI runs only the affected targets per commit.

## Invariants

These change how code must be written. Violating one is a defect even if tests pass.

- **One implementation shared by research and live.** A feature or strategy has exactly one
  definition, imported by both the backtest and the live engine. Divergence here is the project's
  primary failure mode.
- **No wall-clock outside `LiveClock`.** Everything takes time from the `Clock` port. `datetime.now()`
  in domain code breaks replay determinism.
- **Money and prices are `Decimal`; derived features are `float`.** Anything that becomes a P&L, a
  position size or an audit record uses the `Decimal`-backed value objects in `neurotrade.core.types`.
  Indicators and model inputs stay `float`. Never construct a `Price`/`Quantity`/`Money` from a
  `float` — it raises. At a feed boundary use `from_float`, which routes via `repr`.
- **Cross-currency arithmetic raises.** US and Canadian names trade simultaneously, so USD and CAD
  are both live. Converting needs an explicit rate and is not the domain layer's job.
- **Determinism is testable.** Two replays of the same session must produce identical digests. Seed
  every RNG through the registry; never depend on dict/set iteration order.
- **Raw data is immutable.** `raw/` is never mutated; `derived/` is always recomputable from it.
  Features are recomputed from raw, never cached where they can silently diverge.
- **Costs live inside the backtest.** Spread, fees and modelled slippage are applied during
  simulation, never subtracted from results afterwards.
- **Every trade record carries the config hash, model versions and a PIT feature snapshot.** Any
  trade must be reconstructable months later.
- **No model may override a hard risk limit.** Risk rules are structural.
- **Every hypothesis tested increments the trial ledger** — including automated discovery runs.
  Significance is deflated against the true search space.
- **Strategies and features are versioned plugins.** Adding one is a new file plus config, never a
  change to core.

## Working agreements

### Commit workflow — plan execution

When executing a plan, do **not** batch the work into large commits and do not commit anything
yourself.

1. Break the plan into a sequence of **small** commits, each one logically complete and
   understandable on its own.
2. Implement **one** commit's worth of changes, then stop.
3. Write the commit handoff described below.
4. **Wait.** Do not begin the next commit until the user says they have committed and to proceed.

The user commits. Claude never runs `git commit` unless explicitly asked.

### Commit handoff — what to write when a commit is ready

Explain the commit fully. The user reads this instead of the diff, so it must stand alone.

1. **What it contains** — every file added or changed and what each one is for. Not a file list; say
   what the code does.
2. **Why these changes belong together** — the single idea that makes it one commit.
3. **Key call chains**, in this format, one line per chain followed by what each hop does:

   ```
   make check → uv run mypy → src/neurotrade/core/clock.py
     make check      aggregate gate; fails the build on any lint/type/test error
     uv run mypy     strict typecheck across src and tests
     clock.py        the module under test — no wall-clock outside LiveClock
   ```

   Trace from the entry point a reader would actually start from (a make target, a CLI command, a
   test, an inbound event) through to the code this commit adds. Cover every entry point the commit
   introduces. If the commit adds no executable code, say so explicitly rather than inventing a chain.
4. **Decisions and trade-offs** — anything chosen over an obvious alternative, and why.
5. **What was verified** — the commands actually run and their real results. Never claim a check that
   was not run.
6. **What this does not do yet** — the seams left open for the next commit.

### How to group commits

The test of a good commit is whether someone reading it in isolation can follow what happened and
why. Breadth-first scaffolding fails that test badly.

- **Build depth-first, bottom-up — one piece at a time.** Finish a module and its tests before
  starting the next. Follow the dependency order: things that depend on nothing come first, then
  their dependents.
- **Never scaffold breadth-first.** Do not create the whole directory tree, empty `__init__.py`
  files, or placeholder modules for work that lands later. A package appears in the commit that puts
  real code in it, not before.
- **One language at a time.** Python is the system; Go (`api/`) and TypeScript (`ui/`) do not arrive
  until Phase 3 when there is real code for them. Build tooling should be *structured* to accept
  them later — the `Makefile` and CI keep per-language targets that skip cleanly when a toolchain or
  module is absent — but do not install, configure, or stub a language before its code exists.
- **Prefer many small commits over few large ones.** If a summary needs "and" more than once, split it.
- A commit that only adds config or only adds one module plus its tests is the right size. A commit
  that touches every layer is not.

### Conventions

- Be terse in commit messages and prose; sacrifice grammar for concision.
- Twelve-factor config: no hardcoded paths or settings outside `config/` profiles and env vars. This
  is what keeps the eventual move off this Mac a deployment change rather than a port.
- No GPU/CUDA dependency in the base install — PyTorch lives in the optional `gpu` dependency group.
