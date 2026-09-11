# Plugin registries

Invariant: **adding a strategy or feature is a new file plus config, never a change to core.**

## `core/registry.py` — the generic registry

`Registry[T]`, keyed by `(name, Version)` where `Version` is a parsed semver triple.

- `add()` / `register()` decorator — duplicate `(name, version)` raises `DuplicateRegistration`.
- `get(name, version=None)` — `None` means latest.
- `freeze()` — after freezing, registration raises `RegistryFrozen`. Called once the process
  has started trading, so nothing can be swapped underneath a running session.
- `snapshot()` — the sorted `name@version` tuple that goes into a run record. Sorted, never
  set-ordered: iteration order must not vary between runs.

## `features/registry.py` — features, with look-ahead enforcement

`FeatureSpec` wraps a pure function plus `lookback`, `interval`, `version`.
`FeatureRegistry.feature(...)` is the registration decorator.

`evaluate(history, as_of)` **raises `LookaheadError` if any bar in `history` closes after
`as_of`.** This is point-in-time correctness made structural rather than remembered: a feature
physically cannot read the future, in research or in live.

`max_lookback(interval)` tells the engine how much history to hold in the ring buffer.

No features are defined yet — this is the mechanism, awaiting Phase 1's library.

## `strategies/base.py` — the strategy contract

- `Regime` — the volatility/market states a strategy may declare eligibility for.
- `FeatureRef` — a `(name, version|None)` reference. **Deliberately not `order=True`**: sorting
  a mix of pinned and unpinned versions compares `str` to `None` and raises. Sorted by `str()`.
- `StrategyContext` — what a strategy is handed. `context.feature(name)` raises
  `UndeclaredFeature` for anything the strategy did not declare in `requires`, so the feature
  dependency graph is honest and computable ahead of time.
- `Strategy(ABC)` — `on_bar` / `on_quote` / `on_trade`, each returning a sequence of `Intent`.
  Strategies produce *direction*; conviction and size are not theirs to decide.
- `StrategyRegistry` — `eligible(regime)`, `required_features()` (the union across registered
  strategies — what the engine must compute), `snapshot()`, `freeze()`.

No strategies are defined yet — Phase 2.
