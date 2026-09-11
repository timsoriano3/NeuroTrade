# Gotchas — bugs already paid for

Each of these cost real time. They are recorded so they cost it once.

## Domain

**`repr()` of a float subclass is not necessarily a number.**
`repr(np.float64(252.11))` is `'np.float64(252.11)'`, which `Decimal` cannot parse. `np.float64`
passes `isinstance(x, float)`, so it reaches `from_float` looking ordinary and then fails deep
in ingestion. Every feed goes through pandas, so every feed hits this.
Fix: `cls(repr(float(value)))` — normalise through `float()` first.

**Float arithmetic in `to_nanos` drifted ±100 ns on 2862 of 2900 samples.**
`timestamp() * 1e9` is not exact. Rewritten with pure integer arithmetic on the `timedelta`
components. Tests exist that fail against the old implementation.

**`super().__post_init__()` raises under `@dataclass(slots=True)`.**
The decorator returns a *new* class, so the `super()` cell captured at compile time refers to
the pre-decoration class. Call the parent explicitly: `MarketEvent.__post_init__(self)`.
A test asserts subclasses still validate.

**`order=True` on a dataclass with optional fields.**
`FeatureRef` sorting compared `str` to `None` and raised. Dropped `order=True`; sort by `str()`.

**`hash()` is salted per process.** Never derive an identifier from it — it breaks replay
across runs. Everything id-like goes through BLAKE2b in `core/ids.py`.

## Storage

**Hive partition types are inferred, and inference disagrees with the file.**
`session_date` inferred as string, stored as `date32`. Fix: pass an explicit
`SESSION_PARTITIONING` rather than relying on inference.

**`filterwarnings = ["error"]` turns a leaked file handle into a test failure.**
Correct behaviour. `EventStore` gained a `__del__` with `contextlib.suppress`.

**JSON numbers are IEEE doubles.** Encoding a `Decimal` as a JSON number silently destroys the
precision the entire invariant exists to protect. Decimals are encoded as **strings**.

## IBKR

**IBKR timestamps a bar at its OPEN. Our `Bar.ts_event` is its CLOSE.**
The adapter adds the interval. Getting this backwards is a silent look-ahead bug — the bar
appears available before it has finished forming.

**An oversized historical request hangs instead of erroring.**
A 146-year duration string produced a 60-second timeout, not a rejection. `_MAX_DAYS` caps
duration per interval, checked *before* the pacer.

**IBKR's exchange names are not the venue's own.** TSX is `"TSE"`, TSX-V is `"VENTURE"`.

**`Venue.SMART` is an order router, not a listing venue.** `Symbol` rejects it.

**`ib_insync` is archived; `ib_async` is the successor.** Training data and search results are
full of `ib_insync` answers. Use context7 for this API rather than recall.

**Gateway ≠ TWS.** Gateway has no "Enable ActiveX and Socket Clients" checkbox; its socket is
always on. Gateway's API mode must be **IB API**, not FIX CTCI.

## Logging

**structlog caches its logger factory on first use.**
`PrintLoggerFactory(sys.stderr)` binds the stream at import, so anything that later replaces
`sys.stderr` (pytest's capture, a CLI runner) logs to a dead stream. Fix: a `_CurrentStderr`
proxy that resolves `sys.stderr` per write. Needs `__slots__ = ("__weakref__",)`.

## Process — the expensive ones

**Fabricated values, four times.** Hash digests and example outputs written from plausibility
rather than from a command that was run. Doctests caught every one, but only after a wasted
cycle. Rule: never state a hash, digest, count, version or test result not produced by a
command run in this session.

**Silent documentation drift.** Moving the event codec into `core` left the README describing
the old location. This is not preventable by care — it needs `make docs-check` (mechanical
cases) plus an explicit prose audit (everything else).

**`git reset` to reorder commits dropped in-flight work.**
`structlog`, `typer`, `[project.scripts]`, `RunId`, `core/registry.py` and `features/` were
lost, and CI stayed red across three pushes before anyone looked. Diagnosed with
`gh run view --log-failed`. A `git-guard` hook now blocks destructive git.

**`pytest.raises(match=...)` takes a REGEX.** Unescaped `(`, `)`, `.` and `$` in an expected
message tripped ruff's RUF043 four separate times.

**Test fixtures that violate domain invariants are wrong, three times out of three.**
Every time a fixture had `close` outside `high`/`low`, the validation was right and the fixture
was lazy. Do not weaken domain validation to make a test pass.

**mypy `strict` rejects statically-impossible assertions.**
`assert not isinstance(delta, Price)` where the type is already `Decimal` is unreachable code.
Use `typing.assert_type` — checked by mypy, a stronger guarantee than a runtime check that can
never fail.

## Factual

**`TD` on NYSE is Toronto-Dominion, not Tandem Diabetes** (that is `TNDM`). The wrong claim
propagated into seven files before being verified against IBKR. The correct framing for the
same-ticker-two-venues example is dual listing across two currencies.
