# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

NeuroTrade — an autonomous day-trading system for US and Canadian equities. Rules produce trade
direction; ML produces conviction and size; hard risk limits are structural and unlearnable.

**`TRADER_PLAN.md` (gitignored, local copy of the Notion source of truth) is the spec.** It holds the
strategy arsenal, ML stack, validation methodology, data plan, phased roadmap and success metrics.
Read the relevant section before implementing anything in that area. This file covers only what
`TRADER_PLAN.md` does not: how to work in the repo.

**Status: Phase 0 (foundation & data spine) — both exit gates cleared, deliverables incomplete.**
Built: the domain model, configuration and logging, the plugin registries, Parquet/DuckDB corpus
storage, the event log, the event bus, the replay engine (**gate G1**), and the IBKR adapter —
connection, historical bars, pacing and broker (**gate G2**). Still open in Phase 0: the backfill
crawler, the yfinance seed feed, the trading calendar, and the corpus quality gate — the corpus is
empty. `features/` and `strategies/` hold their registry and contract but no actual features or
strategies. Not built at all: models, the risk engine, the dashboard. Sections below describing
those state the contract they will meet, not code that exists.

`WORK/INDEX.md` carries the current status; keep the two in step.

## Session context — read this before exploring

`WORK/` is the project's context ledger. It exists so a session can learn what has been built
by reading ~40 lines instead of re-deriving it from 15k lines of source.

- `WORK/INDEX.md` — the map. Injected automatically at session start by a hook.
- `WORK/phase <#> - <title>/<subtitle>.doc.md` — one coherent subsystem per file.

**Open only the doc the task needs.** `08-gotchas.doc.md` in each phase folder is the highest
value per token in the repo — it is the list of bugs already paid for, and reading it before
writing code is cheaper than rediscovering any one of them.

`TRADER_PLAN.md` is the spec and is large. **Never read it whole.** Use the `plan-section`
agent, or `grep -n` for the section heading and `sed -n 'A,Bp'` that range.

### Checkpointing work

When a coherent body of work is finished — a phase, a subsystem, a gate — say so, propose the
exact `WORK/phase <#> - <title>/<subtitle>.doc.md` path, and recommend writing it and clearing
context before continuing. Then wait; the user decides. A `Stop` hook raises this mechanically
once enough has accumulated, but noticing earlier is better than being told.

Use the `work-journal` skill to write it. Catching this *before* a compaction is the point —
a compaction loses detail a doc would have preserved cheaply.

## Commands

`make` is the single entry point across all three languages.

Go and TypeScript targets exist but report SKIP until `api/` and `ui/` have code (Phase 3).

Only targets that exist are listed. Operational commands arrive with their
subsystems — do not document one before it works.

```
make doctor         # which toolchains are present
make setup          # install dependencies
make check          # lint + typecheck + tests — run before declaring work done
make test / lint / typecheck / fmt
make docs-check     # the docs still describe the code; part of `make lint`
make show-config    # resolved settings and their hash. PROFILE=research|paper|live
make replay         # replay a session, print its digest. LOG= or SESSION=
make verify-replay  # gate G1: replay twice, compare digests
make ibkr-check     # probe IB Gateway: reachable, and the account we expect
make paper-smoke    # gate G2: submit a paper order, acknowledge, cancel
```

Single test: `uv run pytest tests/path/test_x.py::test_name -x`

## Architecture

**Hexagonal + event-sourced.** The trading core knows nothing about IBKR, Postgres or Parquet — all
external systems sit behind ports in `core/ports.py`. Every market event, signal, intent, order and
fill is an append-only record, so any session replays bit-for-bit.

**Layering is enforced by import-linter, not convention.** The contracts live in `.importlinter`
and run as part of `make lint`, so an import that violates this fails before it reaches CI:

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

Modules at the package root — `bus.py`, `config.py`, `logs.py`, `cli.py` — are infrastructure rather
than domain layers. They may depend on `core`; `bus.py` depends on nothing else, because everything
above it publishes to it.

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

### Delegation, batching and model tiers

Agent use is **authorised standing policy for this project** — the user asked for it explicitly.
Do not wait to be told again.

#### Why this is a rule and not a preference

Measured on this project: cache reads outnumbered output tokens **571 : 1**, making re-read of
context roughly **69%** of spend and generated output about **6%**. Cost per request is
proportional to context size, and **every tool call is a request**. So session cost is
approximately `context size × number of requests`.

Two consequences, both counter-intuitive:

- A tool call that returns a lot is charged **twice** — once for the call, and again on every
  later request, because its output now sits in context forever.
- A subagent's context is discarded. Only its return value arrives here. Delegation is
  therefore a **compression** mechanism, not merely a parallelism one.

One session ran 744 main-loop tool calls and 0 delegated ones, and accounted for essentially a
whole day's spend. That is the failure this section exists to prevent.

#### Hard triggers — delegate, do not read

| Trigger | Agent | Model |
|---|---|---|
| Any question about what the spec says | `plan-section` | haiku |
| Any sweep for where code lives or what its surface is | `codebase-locator` | haiku |
| Any test suite, lint or full build gate | `check-runner` | haiku |
| Any question about how a library/SDK actually works | `library-researcher` | sonnet |
| Writing tests for a module | `test-author` | sonnet |
| Auditing a diff against the invariants above | `invariant-auditor` | sonnet |
| Auditing a diff for documentation that went false | `docs-drift-auditor` | sonnet |
| Reviewing validation methodology or statistics | `quant-methodology-reviewer` | **opus** |

**Never read `TRADER_PLAN.md` inline.** It is large and consulted constantly, which makes it the
single most expensive habit available. Use `plan-section`.

**Never run `make check` (or `pytest` over a directory) inline.** Passing output is pure noise
and failing output is long; both land in context permanently. Use `check-runner`.

Run `invariant-auditor` and `docs-drift-auditor` before writing a commit handoff. They catch the
two defect classes `make check` structurally cannot: a violated invariant passes tests, and a
false sentence passes `docs-check`.

Keep it inline when the answer is in a file already open, or when delegating would cost more
round-trips than it saves. Don't delegate trivia.

#### Batching

**Independent tool calls go in one message.** N sequential calls are N full-context requests;
the same N issued together are one. Before making a call, ask what else could be answered in the
same breath — and issue those together.

Sequencing is only justified when a later call genuinely needs an earlier result.

#### Model tiers

| Tier | For |
|---|---|
| haiku | mechanical, high-input/low-output, no judgement |
| sonnet | bounded judgement against a clear rubric |
| opus | reasoning where being wrong is expensive and silent |

**The main loop is a tier choice too, and it dominates the bill.** Cache read is ~69% of spend,
so the model running the conversation sets the rate on nearly all of it.

- **Sonnet for execution sessions** — writing modules and docs, wiring config, running gates,
  applying a plan that already exists. Roughly a fifth the cost at no useful loss.
- **Opus for design, ambiguous debugging, and Phase 1 validation work** — CPCV, triple-barrier
  labelling, deflated Sharpe, PBO, the trial ledger. §17 names backtest overfitting as the
  primary risk and its failure mode is a *clean equity curve that is wrong*; that is worth the
  expensive tier. Locating a file is not.

Say which mode a session is in when it is ambiguous, and suggest `/model sonnet` when a stretch
of work is plainly execution.

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

### Documentation standard

Readers of this codebase will not all have trading experience. Domain jargon —
microprice, R-multiple, LULD halt, maker/taker, triple barrier — gets a plain-English gloss the
first time a module uses it. Assume the reader knows Python and does not know markets.

**Required:**

- **Every public function and method** gets a Google-style docstring: a one-line summary, then
  `Args:` / `Returns:` / `Raises:` where any of them is non-obvious, then an `Example:` of 1–3 lines.
- **Examples are doctests** (`>>>`), and pytest runs them. An example that drifts out of date fails
  the build, which is the only way examples stay true.
- **Every dataclass field** gets a one-line trailing comment giving its meaning and unit —
  especially units, since `Nanos`, R-multiples and per-share versus total costs are easy to confuse.
- **Non-obvious constants and module-level values** get a comment explaining the choice, not the value.
- **Any code whose correctness is not self-evident** gets a comment on *why* — the failure it
  prevents, the alternative rejected, the spec section it implements.

**Not wanted** — these make the codebase harder to read, not easier:

- Restating the signature in prose (`"""Returns the price."""` above `def price() -> Price`).
- `Args:`/`Returns:` blocks on a single-argument accessor whose types already say everything.
- Docstrings on `__str__`, `__repr__`, `__eq__` and similar dunder methods with standard semantics.
- Line-by-line narration of code that reads plainly.
- Repeating in a field comment what the field's type already states — say what it *means*, not what it is.

The test: a comment earns its place if it tells the reader something the code cannot. Prefer one
good sentence about why over three restating what.

**Keep the READMEs in step.** Moving or renaming a module means updating the README that lists it,
and adding a package means giving it one. `make docs-check` catches the mechanical cases — a named
file that no longer exists, a documented `make` target that was never added, a package with no
README, a broken link — and runs as part of `make lint`. It cannot check whether a sentence is still
true, so that part is on you.

### Conventions

- Be terse in commit messages and prose; sacrifice grammar for concision.
- Twelve-factor config: no hardcoded paths or settings outside `config/` profiles and env vars. This
  is what keeps the eventual move off this Mac a deployment change rather than a port.
- No GPU/CUDA dependency in the base install — PyTorch lives in the optional `gpu` dependency group.
- `.claude/settings.json` disables the `huggingface-skills` and `postman` plugins for this repo.
  They are unused here and cost ~5.6k tokens of skill descriptions on *every* request — about 4%
  of a day's cache reads. They stay enabled at user scope for other projects. JSON cannot carry a
  comment, which is why the reason is recorded here.
