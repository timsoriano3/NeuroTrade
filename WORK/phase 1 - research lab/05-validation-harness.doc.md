# Validation harness — CPCV, significance, trial ledger

§8's primary gate and the two things that decide whether a result means anything. Plan commits 6
and 7, landed together: the deflation is useless without a trial count, and the trial count is
pointless without something to deflate.

## What exists

| Module | Responsibility |
|---|---|
| `lab/cv.py` | `CombinatorialPurgedCV`, `WalkForward`, `Split`, `purge_and_embargo` |
| `lab/significance.py` | `sharpe_ratio`, `moments`, `probabilistic_sharpe_ratio`, `expected_max_sharpe`, `deflated_sharpe_ratio`, `probability_of_backtest_overfitting` |
| `lab/trials.py` | `TrialLedger` — `record`, `count`, `sharpes`, `hurdle`, `deflate` |
| `core/trials.py` | `Trial`, `TrialSource` — the record, in `core` because the port names it |
| `core/ports.py` | `TrialLedgerPort` — `append`, `trials(family)` |
| `adapters/storage/trial_ledger.py` | `TrialLedgerStore` — append-only JSONL, `CorruptTrialLedger` |
| `config.py` | `StorageSettings.trial_ledger` → `<data_root>/trials/ledger.jsonl` |

Pure stdlib — `statistics.NormalDist` supplies the normal CDF and its inverse, so no numpy or
scipy entered the dependency set for this.

## CPCV

`split(spans)` takes **label spans**, not indices: `(start, end)` bar indices per observation,
both inclusive. Purging cannot be done without them — an index says when a position opened, and
the leak is about when it closed.

- `C(N, k)` splits, `C(N, k)·k/N` paths. `paths()` returns `(split index, group)` pairs per path,
  one entry per group in time order, so per-split predictions re-deal into complete backtest
  paths. That distribution is what `significance.py` consumes.
- `is_combinatorial` is False at `n_test_groups=1`, which is purged k-fold wearing the name. A
  quick check, not the gate.
- `WalkForward` is the §8 secondary check: block 0 is training-only, each later block tested in
  turn, expanding or rolling. It still purges — "training came first" is about opens, not closes.

**Purging is per contiguous run of test groups, not against their hull.** See gotchas; this is
the one decision in the module that is not cosmetic.

## Significance

Everything is **per-period**. `annualized()` exists for reporting and must never feed
`deflated_sharpe_ratio` — a Sharpe scaled by `sqrt(252)` and a sample size of 252 describe
different experiments.

- `probabilistic_sharpe_ratio` charges for short samples, negative skew and fat tails.
  `moments()` returns **non-excess** kurtosis (3 is normal) because the PSR variance term expects
  it.
- `expected_max_sharpe(n_trials, trial_variance)` is the hurdle: what the best of N worthless
  tries would post. Zero at one trial or zero variance — one look is no search.
- `deflated_sharpe_ratio` is the PSR with that hurdle as its benchmark.
- `probability_of_backtest_overfitting` is CSCV: `n_blocks` blocks, every way of halving them,
  best in-sample strategy's out-of-sample rank logit-transformed; PBO is the share of non-positive
  logits. `is_overfit` at `>= 0.5` — selection no better than a coin flip.

**CSCV recombines blocks out of time order, deliberately.** It asks whether a ranking is stable
across subsamples, not whether anything forecasts forward. That is the only place in the codebase
where order is discarded; anything that trains a model uses `cv.py`, where order is the mechanism.
Default `n_blocks=10` (252 splits) rather than López de Prado's 16 (12,870) — pure-Python cost,
raise it when a result matters.

## Trial ledger

Deflation is computed **within a family** — a family is a search space, and the expected best of
500 opening-range variants says nothing about a mean-reversion idea tried once.

Intended order is run → `record` → `deflate`, so a candidate counts itself. Recording only after
a result proves interesting reproduces exactly the bias the ledger removes.

`recorded_ns` comes from the injected `Clock`, so a `SimClock`-driven research script derives the
same `TrialId` on every run and a rebuilt ledger gains no phantom trials. `TrialId` is derived
from `(hypothesis, config_hash, recorded_ns)`.

**Storage choices are all about not shrinking.** JSONL, one file for the project's lifetime (the
event log rotates per session; this does not, because today's candidate deflates against last
quarter's search). Every append opens, writes one line, closes. A line that will not decode raises
`CorruptTrialLedger` rather than being skipped — a skipped record lowers the count, which lowers
the hurdle, which approves what should have been rejected. `trial_ledger` sits beside `events/`,
not under `derived/`: it is not recomputable, because abandoned searches are gone with it.

## Deliberately not done

- **No CLI command and no make target.** These are libraries; the entry point is plan commit 8's
  `verify-lab`, which runs the deliberately overfit control strategy through them.
- **Nothing calls them yet.** No strategy exists to validate (Phase 2), so the numbers here have
  been exercised against constructed series only — no claim is made about the corpus.
- **Promotion thresholds are unset.** §8 lists five promotion gates; the DSR threshold is a
  config decision that belongs with the promotion flow (§10.2), not here.
- **`n_paths` is recorded on a `Trial` but nothing enforces it.** A trial claiming 5 paths is
  taken at its word.
