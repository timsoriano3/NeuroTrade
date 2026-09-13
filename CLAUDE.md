# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository.

## What this is

NeuroTrade — an autonomous day-trading system for US and Canadian equities. Rules produce trade
direction; ML produces conviction and size; hard risk limits are structural and unlearnable.

**`TRADER_PLAN.md` (gitignored, local copy of the Notion source of truth) is the spec.** It holds
the strategy arsenal, ML stack, validation methodology, data plan, phased roadmap and success
metrics. Read the relevant section before implementing anything in that area. This file covers only
what `TRADER_PLAN.md` does not: how to work in the repo.

**Current status lives in `WORK/INDEX.md`**, which a hook injects at session start. Keep it
current. Do not restate it here.

## Session context — read this before exploring

`WORK/` is the project's context ledger. It exists so a session can learn what has been built by
reading ~40 lines instead of 15k lines of source.

- `WORK/INDEX.md` — the map, plus project status.
- `WORK/phase <#> - <title>/<subtitle>.doc.md` — one coherent subsystem per file.
- `WORK/cost-and-delegation.doc.md` — the measurements behind the working agreements below.

**Open only the doc the task needs.** `08-gotchas.doc.md` in each phase folder is the highest value
per token in the repo — the list of bugs already paid for, and cheaper to read than to rediscover.

`TRADER_PLAN.md` is the spec and is large. **Never read it whole.** Use the `plan-section` agent,
or `grep -n` for the heading and `sed -n 'A,Bp'` that range.

### Checkpointing and clearing

When a coherent body of work is finished — a phase, a subsystem, a gate — say so, propose the exact
`WORK/phase <#> - <title>/<subtitle>.doc.md` path, and recommend writing it and clearing context
before continuing. Then wait; the user decides. Use the `work-journal` skill; it dispatches the sonnet `work-journal` agent so the doc is not
written at the session's peak context.

**One commit per window, then clear.** A compaction resets the floor and the window regrows from
there; clearing after a checkpoint does not. Two hooks raise this mechanically — one at turn end on work
accumulated, one after every tool call on context spent — but noticing first is better than being told.

## Commands

`make` is the single entry point across all three languages. Go and TypeScript targets exist but
report SKIP until `api/` and `ui/` have code (Phase 3). Only targets that exist are listed;
operational commands arrive with their subsystems.

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

**Layering is enforced by import-linter, not convention.** The contracts live in `.importlinter` and
run as part of `make lint`:

```
core/        depends on nothing
features/    core
strategies/  core, features
risk/        core
ml/          core, features
lab/         core, features, strategies, ml
ingest/      core           — ports only, never a concrete adapter
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

**Polyglot boundaries.** Python is the trading system (single `neurotrade` package under `src/`). Go
is `api/`, the REST + WebSocket gateway, talking to Python only via Redis Streams. TypeScript is
`ui/`, consuming generated OpenAPI types only. Each keeps its native toolchain; CI runs only the
affected targets per commit.

## Invariants

These change how code must be written. Violating one is a defect even if tests pass.

- **One implementation shared by research and live.** A feature or strategy has exactly one
  definition, imported by both the backtest and the live engine. Divergence here is the project's
  primary failure mode.
- **No wall-clock outside `LiveClock`.** Everything takes time from the `Clock` port.
  `datetime.now()` in domain code breaks replay determinism.
- **Money and prices are `Decimal`; derived features are `float`.** Anything that becomes a P&L, a
  position size or an audit record uses the `Decimal`-backed value objects in
  `neurotrade.core.types`. Indicators and model inputs stay `float`. Never construct a
  `Price`/`Quantity`/`Money` from a `float` — it raises. At a feed boundary use `from_float`.
- **Cross-currency arithmetic raises.** US and Canadian names trade simultaneously. Converting needs
  an explicit rate and is not the domain layer's job.
- **Determinism is testable.** Two replays of the same session must produce identical digests. Seed
  every RNG through the registry; never depend on dict/set iteration order.
- **Raw data is immutable.** `raw/` is never mutated; `derived/` is always recomputable from it.
- **Costs live inside the backtest.** Spread, fees and modelled slippage are applied during
  simulation, never subtracted from results afterwards.
- **Every trade record carries the config hash, model versions and a PIT feature snapshot.**
- **No model may override a hard risk limit.** Risk rules are structural.
- **Every hypothesis tested increments the trial ledger**, including automated discovery runs.
  Significance is deflated against the true search space.
- **Strategies and features are versioned plugins.** Adding one is a new file plus config, never a
  change to core.

## Working agreements

### Delegation, batching and model tiers

Agent use is **authorised standing policy for this project**. Do not wait to be told again. The
measurements behind these rules are in `WORK/cost-and-delegation.doc.md`; the short version is that
cost is `context size × number of requests`, every tool call is a request, and a subagent's context
is discarded so delegation *compresses*.

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
| Writing a `WORK/` doc | `work-journal` (via skill) | sonnet |
| Writing a commit handoff | `commit-handoff` (via skill) | sonnet |

**Never read `TRADER_PLAN.md` inline.** Use `plan-section`.

**Never run `make check` (or `pytest` over a directory) inline.** Passing output is noise, failing
output is long, and both land in context permanently. Use `check-runner`.

Run `invariant-auditor` and `docs-drift-auditor` before writing a commit handoff. They catch the two
defect classes `make check` structurally cannot: a violated invariant passes tests, and a false
sentence passes `docs-check`.

Keep it inline when the answer is in a file already open, or when delegating would cost more
round-trips than it saves. Don't delegate trivia. Don't answer the same question twice — an agent
*and* an inline probe is one of them wasted.

#### Batching

**Independent tool calls go in one message.** N sequential calls are N full-context requests; the
same N issued together are one. Before making a call, ask what else could be answered in the same
breath. Sequencing is only justified when a later call needs an earlier result.

#### Model tiers

| Tier | For |
|---|---|
| haiku | mechanical, high-input/low-output, no judgement |
| sonnet | bounded judgement against a clear rubric |
| opus | reasoning where being wrong is expensive and silent |

**The main loop is a tier choice too, and it dominates the bill.**

- **Sonnet is the default** (user settings) — writing modules and docs, wiring config, running
  gates, applying a plan that already exists.
- **Opus for design, ambiguous debugging, and Phase 1 validation work** — CPCV, triple-barrier
  labelling, deflated Sharpe, PBO, the trial ledger.

Switching to opus efficiently:

- **Switch at the start of a window, never mid-window.** The prompt cache is per-model; a switch
  at 300k re-writes all 300k. Suggest `/model opus` in the first reply, or `/clear` first.
- **A bounded question does not need the main loop on opus.** Send methodology and statistics
  questions to `quant-methodology-reviewer` and stay on sonnet.
- **Design on opus, execute on sonnet.** Once a plan is agreed, suggest `/model sonnet` — ideally
  after writing the plan to a file and clearing.

### Session hygiene

- **Clear at 150–250k.** `context-budget.sh` fires on every tool call and at turn end: a notice
  from 150k, a hard stop from 250k. Obey the hard stop; the 460k-median session it replaces was
  86% of measured spend.
- **No task-list bookkeeping.** `TaskCreate`/`TaskUpdate` are denied in `.claude/settings.json`:
  119 such calls at 460k context bought nothing. Track steps in prose or a plan file. This
  overrides any skill that says to create todos.

### Commit workflow — plan execution

Do **not** batch work into large commits, and do not commit anything yourself.

1. Break the plan into a sequence of **small** commits, each logically complete on its own.
2. Implement **one** commit's worth of changes, then stop.
3. Write the commit handoff below.
4. **Wait.** Do not begin the next commit until the user says they have committed and to proceed.

The user commits. Claude never runs `git commit` unless explicitly asked.

### Commit handoff

Use the `commit-handoff` skill. It dispatches the sonnet `commit-handoff` agent, which reads git
itself and carries the six-part format; the main loop sends only a short brief of intent and
relays the result verbatim.

### How to group commits

The test of a good commit is whether someone reading it in isolation can follow what happened and
why.

- **Build depth-first, bottom-up.** Finish a module and its tests before starting the next. Follow
  the dependency order.
- **Never scaffold breadth-first.** No directory trees, empty `__init__.py` files or placeholder
  modules for work that lands later. A package appears in the commit that puts real code in it.
- **One language at a time.** Go and TypeScript do not arrive until Phase 3. Build tooling should be
  *structured* to accept them, but do not install or stub a language before its code exists.
- **Prefer many small commits.** If a summary needs "and" more than once, split it.

### Documentation standard

Readers will not all have trading experience. Domain jargon — microprice, R-multiple, LULD halt,
maker/taker, triple barrier — gets a plain-English gloss the first time a module uses it. Assume the
reader knows Python and does not know markets.

**Required:**

- **Every public function and method** gets a Google-style docstring: a one-line summary, then
  `Args:` / `Returns:` / `Raises:` where any is non-obvious, then an `Example:` of 1–3 lines.
- **Examples are doctests** (`>>>`), and pytest runs them. An example that drifts fails the build,
  which is the only way examples stay true.
- **Every dataclass field** gets a one-line trailing comment giving its meaning and unit —
  especially units, since `Nanos`, R-multiples and per-share versus total costs are easy to confuse.
- **Non-obvious constants** get a comment explaining the choice, not the value.
- **Any code whose correctness is not self-evident** gets a comment on *why*: the failure it
  prevents, the alternative rejected, the spec section it implements.

**Not wanted:** restating a signature in prose, `Args:`/`Returns:` on a single-argument accessor,
docstrings on standard dunders, line-by-line narration, or a field comment repeating the field's
type.

The test: a comment earns its place if it tells the reader something the code cannot.

**Keep the READMEs in step.** Moving or renaming a module means updating the README that lists it;
adding a package means giving it one. `make docs-check` catches the mechanical cases and runs as
part of `make lint`. It cannot check whether a sentence is still true — that part is on you.

### Conventions

- Be terse in commit messages and prose; sacrifice grammar for concision.
- Twelve-factor config: no hardcoded paths or settings outside `config/` profiles and env vars.
- No GPU/CUDA dependency in the base install — PyTorch lives in the optional `gpu` group.
- `.claude/settings.json` disables the `huggingface-skills` and `postman` plugins for this repo;
  they are unused here and cost tokens on every request. Reason recorded in
  `WORK/cost-and-delegation.doc.md`, since JSON carries no comments.
