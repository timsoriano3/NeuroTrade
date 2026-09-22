# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository.

## What this is

NeuroTrade — an autonomous day-trading system for US and Canadian equities. Rules produce trade
direction; ML produces conviction and size; hard risk limits are structural and unlearnable.

**`TRADER_PLAN.md` (gitignored, local copy of the Notion source of truth) is the spec.** It holds
the strategy arsenal, ML stack, validation methodology, data plan, phased roadmap and success
metrics. Read the relevant section before implementing anything in that area. This file covers only
what `TRADER_PLAN.md` does not: how to work in the repo.

**Current status lives in `WORK/INDEX.mem.md`**, which a hook injects at session start. Keep it
current. Do not restate it here.

## Session context — read this before exploring

`WORK/` is the project's context ledger. It exists so a session can learn what has been built by
reading ~40 lines instead of 15k lines of source.

**`WORK/` is gitignored and local-only**, like `TRADER_PLAN.md`. It grows without bound and none
of it belongs in the repo's history, so it never appears in a commit and is never staged. The cost
is that it does not survive a fresh clone — treat it as this machine's memory, not as a deliverable.

- `WORK/INDEX.mem.md` — the map, plus project status. **Cap 45 lines**; injected every session.
- `WORK/phase <#> - <title>/<subtitle>.doc.md` — one coherent subsystem per file. **Cap ~100.**
- `WORK/phase <#> - <title>/08-gotchas.mem.md` — the traps for that phase.
- `WORK/cost-and-delegation.doc.md` — the measurements behind the working agreements below.

**The suffix says whether writing it needs permission** (global rules, Working files): `.mem.md` is
state, `.doc.md` and `.plan.md` are reasoning, and both are written only at the commit checkpoint
on an explicit yes. Gotchas are a `.mem.md` precisely so a trap that would otherwise be lost can be
recorded without waiting for the gate.

**Open only the file the task needs.** `08-gotchas.mem.md` in each phase folder is the highest value
per token in the repo — the list of bugs already paid for, and cheaper to read than to rediscover.

`TRADER_PLAN.md` is the spec and is large. **Never read it whole.** `grep -n` for the heading and
`sed -n 'A,Bp'` that range — see Working procedures.

### Checkpointing and clearing

**Journalling is gated — always ask first.** Once a commit's work passes its gate and audits, put
the journal-worthy findings in chat, then ask, verbatim:

> **"Have you reviewed the code and want to update the working files?"**

Only an explicit yes licenses a write; silence, a question, or a new instruction does not. On yes,
run the `work-journal` skill for everything accumulated since the last write — not just the last
exchange. Pick the path yourself, rewriting the existing subsystem doc or naming a new
`WORK/phase <#> - <title>/<subtitle>.doc.md`, and say which you wrote.

Journalling happens at the session's peak context, so write from git and slices rather than
re-reading source you no longer hold. **WORK/ is gitignored**, so the write is never part of the
commit and never staged — it lands on disk only.

**The exemption:** a trap that would be lost before the next checkpoint goes straight into the
phase's `08-gotchas.mem.md` without asking, as does an index update when the window is about to be
cleared. Say in one line that you did it and why.

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
make backfill       # fill the bar corpus from IBKR. START=YYYY-MM-DD [END= LIMIT= PASSES=]
make seed           # seed the corpus from free vendor samples: fetch then ingest
make seed-fetch / seed-ingest   # either half alone. SOURCE=firstrate|kibot [SNAPSHOT=]
make daily          # fill the daily-bar corpus from Yahoo. START=YYYY-MM-DD [END= LIMIT=]
make universe       # build point-in-time universe membership. START=YYYY-MM-DD [END=]
make actions        # fetch splits and dividends from Yahoo. START=YYYY-MM-DD [END=]
make actions-check  # audit the daily corpus for gaps no action explains. START=YYYY-MM-DD [END= THRESHOLD=]
make corpus-check   # audit the corpus for faults. START=YYYY-MM-DD [END= INTERVAL= SOURCE= LIMIT=]
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

### Working procedures — everything runs in the main thread

**No subagents.** The `Agent` tool is not used in this project: a fresh agent re-derives context
this session already holds, and the measured effect was more spend, not less. The procedures below
carry the logic the retired agents held; run them yourself.

Cost is still `context size × number of requests`, so the goal each procedure serves is the same:
read little, keep long output out of the window.

#### Keep tool output out of context

- **Locate, then slice.** `grep -n` to find, `sed -n 'A,Bp'` to read. Never read a whole file to
  answer "where is X" or "what does the spec say". Read a file whole only when you will edit most
  of it.
- **Long-output commands go to a log file, never to the window:**
  ```bash
  make check > "$TMPDIR/gate.log" 2>&1; echo "exit=$?"
  grep -nE 'FAIL|failed|error:|Error' "$TMPDIR/gate.log" | head -20   # only on failure
  ```
  Same for a full `pytest` run. Passing output is noise; failing output is long. Report the
  outcome and the actionable lines, never the transcript.
- **Batch.** Independent tool calls go in one message; N sequential calls are N full-context
  requests. Sequence only when a later call needs an earlier result.

#### Reading the spec

`TRADER_PLAN.md` is large and gitignored. **Never read it whole.**

```bash
grep -nE '^#{1,4} ' TRADER_PLAN.md     # table of contents
grep -n -i '<topic>' TRADER_PLAN.md    # candidates
sed -n '<start>,<end>p' TRADER_PLAN.md # the section only
```

Quote the section verbatim with its line range; say `spec is silent on this` rather than filling
a gap. Stable anchors: §3.6 one shared feature implementation, §10.1 profiles, §12.1 corpus build,
§13 phased roadmap, §18 repo strategy.

#### Researching a library

1. **context7 first** (`resolve-library-id`, then `query-docs`) — model recall is stale by
   definition. Web search only when context7 has no answer.
2. **Check the package is still the maintained one.** Renames and successor packages are the most
   common source of confidently wrong answers (`ib_insync` → `ib_async`).
3. **Verify against the installed version:** `uv pip show <pkg> | head -3`, and read the source in
   `.venv` when the docs are ambiguous — that is how the `reqHistoricalDataAsync` timeout
   behaviour was settled.
4. Say **unverified** rather than stating a plausible signature.

#### Writing tests

- **Heavier on rejection than on happy paths.** Section comments (`# ── Price ───`) group a file;
  `pytest.mark.parametrize` for families of bad input.
- **`pytest.raises(match=...)` takes a REGEX** — escape `(`, `)`, `.`, `$` or ruff RUF043 fails
  the build. Hit four times.
- `typing.assert_type` for type-level guarantees, never an `isinstance` assertion mypy can prove.
- Doctests in `src/` run as tests. Any `>>>` output must be **executed, not predicted** — likewise
  any hash, digest or count.
- `filterwarnings = ["error"]`: a warning fails the suite; fix the leak.
- IBKR tests get `@pytest.mark.ibkr` and are excluded by default.
- **Never weaken a domain invariant to make a test pass.** Three times a fixture had `close`
  outside `high`/`low`; the validation was right every time.
- While iterating, run **only the file you are writing** (`uv run pytest <file> -x -q | tail -30`),
  then the full gate once at the end.

#### Auditing a diff before the handoff

`make check` structurally cannot catch two defect classes, so run both passes yourself on
`git diff HEAD` plus untracked files, before writing the commit handoff.

**Invariants** — hold the diff against the Invariants section above, not against general good
practice. Useful sweeps:
```bash
grep -rn 'datetime.now\|time.time()\|utcnow' src        # wall-clock leakage
grep -rn 'float(' src/neurotrade/core src/neurotrade/features   # precision boundary
grep -rn 'for .* in \(set(\|\.items()\)' src            # iteration-order dependence
grep -rn 'random\.\|np.random' src | grep -v seed       # unseeded RNG
```
Report violations only, as `VIOLATION — "<invariant>"` with file:line and the failure it causes.

**Docs drift** — read every README and `CLAUDE.md` the change set touches on and ask only: would a
reader be **misled**? A module described in the wrong package, a contract that changed shape, a
command documented with behaviour it no longer has, a list gone stale. Typos are not drift.
Mechanical cases (missing file, dead link, absent make target) are `make docs-check`'s job.

#### Reviewing methodology and statistics (Phase 1 onward)

§17 names backtest overfitting as the project's primary risk: the failure is not a crash, it is a
clean equity curve that does not survive live. Tests cannot catch it. Before trusting any
validation result, hunt:

- **Leakage** — any path from `t+1` into a decision at `t`. Feature windows closing after the
  label opens, normalisation fitted on the full sample, bar timestamps taken at the open,
  survivorship in the universe, corporate actions applied retroactively.
- **CV that leaks through overlapping labels** — triple-barrier labels span time. Check purging
  and embargo, and that CPCV is combinatorial rather than a renamed k-fold.
- **Multiple testing** — every hypothesis, including every automated discovery run, inflates the
  best observed Sharpe. Deflate (DSR, PBO) against the true search space, not the results kept.
- **Cost modelling** — costs inside the simulation, not subtracted after; no mid fills; spread on
  exit; slippage that scales with size and volatility; maker/taker distinguished.
- **Regime and stationarity** — check the distribution of outcomes, not the aggregate.

Say plainly when a result is not trustworthy. Being agreeable here is worse than saying nothing.

#### Model choice

The main loop is the whole bill now.

- **Sonnet is the default** — writing modules and docs, wiring config, running gates, applying an
  agreed plan.
- **Opus for design, ambiguous debugging, and Phase 1 validation work** — CPCV, triple-barrier
  labelling, deflated Sharpe, PBO, the trial ledger.
- **Switch at the start of a window, never mid-window.** The prompt cache is per-model; switching
  at 300k rewrites all 300k. Suggest `/model opus` in the first reply, or `/clear` first, and
  `/model sonnet` once a plan is agreed.

### Session hygiene


- **Clear at 150–250k.** `context-budget.sh` fires on every tool call and at turn end: a notice
  from 150k, a hard stop from 250k. Obey the hard stop; the 460k-median session it replaces was
  86% of measured spend.
- **No task-list bookkeeping.** `TaskCreate`/`TaskUpdate` are denied in `.claude/settings.json`:
  119 such calls at 460k context bought nothing. Track steps in prose or a plan file. This
  overrides any skill that says to create todos.

### Commit workflow — plan execution

Do **not** mix unrelated scopes in one commit, and do not commit anything yourself.

1. Break the plan into commits **grouped by scope** (see below), each logically complete on its own.
2. Implement **one** commit's worth of changes, then stop.
3. Audit the diff (invariants, then docs drift) — see Working procedures.
4. Write the commit handoff below. It covers the code only — WORK/ is gitignored.
5. Ask the working-files question (see Checkpointing) and journal only on a yes.
6. **Wait.** Do not begin the next commit until the user says they have committed and to proceed.

The user commits. Claude never runs `git commit` unless explicitly asked.

### Commit handoff

Use the `commit-handoff` skill; it carries the six-part format. Write it from `git status --short`,
`git diff HEAD --stat` and the hunks, plus the gate results already run this session. Never claim a
check that was not run.

### How to group commits

The test of a good commit is whether someone reading it in isolation can follow what happened and
why.

- **Build depth-first, bottom-up.** Finish a module and its tests before starting the next. Follow
  the dependency order.
- **Never scaffold breadth-first.** No directory trees, empty `__init__.py` files or placeholder
  modules for work that lands later. A package appears in the commit that puts real code in it.
- **One language at a time.** Go and TypeScript do not arrive until Phase 3. Build tooling should be
  *structured* to accept them, but do not install or stub a language before its code exists.
- **Group by scope, not by file.** One commit per coherent scope: a subsystem with its tests,
  wiring and docs lands together. Pieces that share a scope (sibling adapters behind one port,
  a command and its make target, a fix and the doc it invalidates) are one commit, not a
  sequence. Split only when scopes differ — a reader should not need three commits to follow
  one idea. Phase 0 shipped too many slivers; aim for roughly one commit per plan section.

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
