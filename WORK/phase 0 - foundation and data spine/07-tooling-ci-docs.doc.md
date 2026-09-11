# Build tooling, CI, and keeping docs honest

## `make` — one entry point for three languages

Python is real; `go-*` and `ts-*` targets exist and print **SKIP** until `api/` and `ui/` have
code (Phase 3). SKIP always means "nothing to run here", never "the command failed". The entry
point does not change shape when those languages land.

```
make doctor  setup  check(=lint typecheck test)  fmt  lint  typecheck  test
make docs-check  show-config  replay  verify-replay  ibkr-check  paper-smoke  clean
```

## Python toolchain

`uv` for everything. Ruff (lint + format, line length 100), mypy `strict` with the pydantic
plugin, pytest.

Notable pytest config:
- `testpaths = ["tests", "src"]` with `--doctest-modules` — **every `>>>` example in a
  docstring runs as a test.** An example nobody executes drifts and then actively misleads.
  This is also what has caught fabricated values four separate times.
- `filterwarnings = ["error"]` — a warning here is usually a correctness signal.
- `--strict-markers --strict-config`.
- `-m 'not ibkr'` by default.

## `.importlinter` — layering enforced, not suggested

Six `forbidden` contracts, run as part of `make lint`:
`core-is-the-bottom`, `adapters-do-not-reach-upward`, `features-depend-only-on-core`,
`strategies-depend-on-core-and-features`, `lab-uses-ports-not-adapters`,
`the-bus-is-infrastructure`.

Written as separate `forbidden` contracts rather than one `layers` contract on purpose: a
failure then names the specific rule broken and carries the reason it exists.

Layers not yet built (`risk`, `ml`, `execution`, `discovery`, `promotion`) get contracts when
they get code. **`discovery/` being import-isolated is what will physically prevent
experimental logic from reaching live capital — never add an import into it.**

## `.pre-commit-config.yaml`

Deliberately **not** a mirror of `make check`: mypy and pytest are too slow to sit in front of
every commit, and a hook slow enough to be annoying gets bypassed with `--no-verify`, which
catches nothing. Those run in CI, which cannot be bypassed.

Runs: whitespace/EOF/YAML/TOML/merge-conflict/case-conflict checks, `detect-private-key`,
large-file guard (512 KB), `ruff-check --fix` then `ruff-format` (ruff's recommended order),
and a local `no-market-data` hook blocking `.parquet/.duckdb/.arrow/.feather/.h5/.npy/.pkl`
outside `tests/fixtures/`.

## CI — `.github/workflows/ci.yml`

Runs per-language targets only for what changed. Note: `astral-sh/setup-uv` has no bare major
tag past `v7` — pin a full version (`@v9.0.0`). Check `git/matching-refs/tags`, not
`releases/latest`.

## `scripts/check_docs.py` — `make docs-check`

Docs drift because nothing fails when they go stale. This fails. Four checks:

1. Every file a doc names actually exists.
2. Every `make` target a doc claims actually exists.
3. Every package containing code has a README.
4. Every internal link resolves.

Restricted to fenced and inline code spans, so ordinary prose does not trip it.

**What it cannot check is whether a sentence is still true.** That gap is real, and is what the
`docs-drift-auditor` subagent exists to cover.
