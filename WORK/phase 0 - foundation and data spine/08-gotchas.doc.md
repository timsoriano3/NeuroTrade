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

**A bar stamped at the session open belongs to the PREVIOUS session.**
Because `ts_event` is the bar's close, a bar stamped 09:30 closed *at* the bell and covers the
minute before it. The first bar of a US session closes at 09:31; the last closes at exactly
16:00. Hence `TradingSession.holds_bar` is open-exclusive and close-inclusive, while
`contains` (was the venue open at this instant?) is inclusive at both ends. Using one for the
other shifts every session by a bar and corrupts the opening range.

**An exchange calendar's default bounds move with today's date.**
`exchange_calendars.get_calendar("XNYS")` covers twenty years back and one year forward *counted
from now*, so the same query answers differently next week and a date beyond the horizon raises
`DateOutOfBounds` rather than returning nothing. Always pass explicit `start`/`end`. Anything
deriving a work queue from a moving horizon silently changes what "complete" means.

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

**A healthy `ibkr check` does not mean the data farms are up.** The socket, login and account
check can all pass while historical requests hang — observed on a weekend against account
`DUT108414`. Connect also logs noisy "request timed out" lines for positions/orders/executions
during the same outage — harmless, ignore them.

**Fixed:** `IbkrMarketData` now has a `request_timeout_seconds` config (default 60s, `ib_async`'s
own default). Qualify uses `asyncio.wait_for`; history passes `timeout=` into
`reqHistoricalDataAsync` and reclassifies an empty-but-elapsed result as `MarketDataError` — see
`06-ibkr-adapter.doc.md`. Verified live 2026-09-12 (Sat, farms down) with the timeout set to 10s:
`ibkr backfill` failed 5 cells cleanly with "resolving X went unanswered for 10.0s" and stopped
the pass (exit 1) instead of hanging.

**IBKR's VWAP can fall outside its own bar, and it used to cost the whole session.**
Live, 2026-09-15: JPM came back with `average=344.576` against a high of `344.57`. `Bar` rejected
it — correctly — and because `fetch_bars` builds every bar in one comprehension, the entire
390-bar session was lost over one auxiliary field. `average` carries more decimals than the
tick-rounded high and low it ships beside, so it can land thousandths outside the range.

**Fixed:** `_checked_vwap` drops the VWAP and keeps the bar. Dropped, not clamped — a clamped
price is a number no trade printed, and `vwap` is already optional. Counts and the worst excess
per symbol are exposed as `IbkrMarketData.vwap_drops()` and printed by `ibkr backfill`, so a
rounding artefact stays distinguishable from a feed that is actually wrong.

**`ib_insync` is archived; `ib_async` is the successor.** Training data and search results are
full of `ib_insync` answers. Use context7 for this API rather than recall.

**Gateway ≠ TWS.** Gateway has no "Enable ActiveX and Socket Clients" checkbox; its socket is
always on. Gateway's API mode must be **IB API**, not FIX CTCI.

## Universe history

**A corpus that runs past `--end` must not fail the build.** The screen maps each bar to its
session by the calendar, so reading to the end of the corpus made every bar after `--end` match no
session and raise. The read is now bounded at both ends by the window's own sessions. Caught by a
test, not by the live run, because the live run happened to end where the corpus did.

**Eligibility has to be gated on the symbol's own venue.** Without that check the first real build
returned 43 of 43 members on *every* date — including the TSX names on days the TSX was shut. A
date is a decision point if any venue traded; an instrument is only eligible if its own venue did.

**`Price` is already positive by construction**, so a second "must be positive" check on a price
floor is dead code with its own error message to keep in step. `Money` is not — it may be
negative, since losses are money — so the dollar-volume floor does need its own guard.

## Vendor sample files

**python.org's macOS Python ships with no CA certificates.** `urllib.request.urlopen` on HTTPS
fails with `CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate` until the
interpreter's `Install Certificates.command` is run by hand — which a fresh clone on another
machine will not have done. Fix: build the context explicitly,
`ssl.create_default_context(cafile=certifi.where())`. Found only because a live fetch was tried;
every unit test passed with a fake opener.

**Both vendors stamp a bar at its OPEN, in Eastern time.** Verified from the files, not the
docs: FRD runs 04:00–19:59 with no 20:00 bar, Kibot 09:30–15:59 with no 16:00. Same shift as
IBKR — add one interval to reach the close — and the same silent lookahead if skipped.

**Kibot's free sample is a rolling window, not a fixed snapshot.** The file holds the last ~3
months and changes under you; two fetches are different data, so every fetch is stored in its
own dated folder.

**Kibot download links carry no filename.** The page lists opaque `api.kibot.com/?get=<code>`
links and the real name arrives only on the `Content-Disposition` header, so the wanted file has
to be identified by opening each link. The codes can rotate; a fetch that matches nothing must
fail loudly rather than silently seeding an empty corpus.

**FRD's sample is unadjusted** — settled 2026-09-15 against yfinance, see `13-daily-bars.doc.md`.
For dividends only: AAPL's window holds no split.

**Kibot's default files are split- and dividend-adjusted.** The `_unadjusted` variants differ —
IBM's first open on 2026-06-15 is 272.00 unadjusted against 270.06 adjusted. Raw must stay raw:
take the `_unadjusted` files, since adjustment is the stage 5 quality gate's job.

**Dating a bar by its UTC close puts it on the wrong day.** A 19:59 ET bar closes at 00:59 UTC the
*next* day, so deriving a file's session range from `ts_event` stretches it a day past what the
file holds and makes `seed ingest` crawl a session with no rows. `covered_dates` undoes the
open→close shift first and takes the **ET open's** date.

**FRD's sample has absent no-trade minutes, and not evenly.** After a full live ingest the six
NASDAQ names were 251/251 complete sessions, while DIA was 141/251 and VXX 124/251 — short by one
or two bars on the rest. Nothing is wrong with the ingest: the files have no row for a minute that
did not trade. A Phase 1 completeness gate must not read a short seed session as a failed fetch.

**`seed fetch` dates its folder in UTC.** A fetch run late on the 15th local lands in
`2026-09-16/`. Intended — `Clock` is UTC everywhere — but `--snapshot` takes the folder's name,
not the local day it felt like.

**RUF043 again (fifth time): `pytest.raises(match=...)` is a regex.** `match="no manifest.json"`
fails the build over the unescaped `.`. Use a raw string with the dot escaped.

## Yahoo / yfinance

**Yahoo's JSON floats overflow the corpus's Decimal column.** A $169.34 close arrives as
`169.33999633789062`; `Price.from_float` keeps the shortest round-tripping form — fourteen
decimal places — and `pyarrow` then refuses the write with `ArrowInvalid: Rescaling Decimal value
would cause data loss`. Every unit test passed; the first live run died on the first symbol.
Fix: `round(value, 4)` before `from_float` — four places is the finest tick a North American
equity quotes in, and it recovers the price that actually traded instead of preserving float
noise that would never compare equal to IBKR's `169.34`.

**`expected_bars` floored a daily session to ZERO bars.** `duration_ns // interval.nanos` for a
6.5-hour session and a 24-hour bar is 0, and `Coverage.is_complete(0)` is true for any count — so
every daily cell looked filled, the plan came back empty, and a daily backfill would have fetched
nothing while reporting success. Fixed in `TradingSession.expected_bars`: an interval at least as
long as the session gives 1.

**yfinance 1.7 deprecated `raise_errors=`.** It is `yf.config.debug.hide_exceptions = False` now,
global to the package. Passing the old argument still works but emits a `DeprecationWarning`, and
`filterwarnings = ["error"]` turns that into a failed test. The default (`True`) answers an empty
frame instead of raising, which is indistinguishable from "this symbol has no data here" — a 404
and a genuine gap would both enter the corpus as silence.

**yfinance traces every request step at DEBUG** — cookie fetches, crumb lookups, the SQL of its
own cache. The research profile runs at `DEBUG`, so a crawl buries its own progress in vendor
tracing. The adapter raises `logging.getLogger("yfinance")` to `WARNING`.

**Yahoo's daily index is tz-aware in the EXCHANGE's timezone**, stamped at local midnight —
`America/New_York` for US lines, `America/Toronto` for `.TO`. Converting to UTC before taking
`.date()` moves a Toronto row onto the next day.

## Logging

**structlog caches its logger factory on first use.**
`PrintLoggerFactory(sys.stderr)` binds the stream at import, so anything that later replaces
`sys.stderr` (pytest's capture, a CLI runner) logs to a dead stream. Fix: a `_CurrentStderr`
proxy that resolves `sys.stderr` per write. Needs `__slots__ = ("__weakref__",)`.

## Tooling

**ruff's ANN401 bans `typing.Any` in any signature.**
Wrapping an untyped third-party object costs a real annotation, not `Any`. Name the library's own
class (`xcals.ExchangeCalendar`, `pd.Timestamp`) — mypy still sees `Any` through
`ignore_missing_imports`, but the signature documents what it is. Untyped packages also need an
override block in `pyproject.toml` or strict mypy rejects the import.

**YAML turns real tickers into booleans.** `yaml.safe_load` reads `ON`, `NO`, `OFF`, `TRUE`
and `FALSE` as `bool`, in any case. `ON` is ON Semiconductor on NASDAQ — a live listing that
arrives as `True`. Verified in-session; `Y` and `N` survive as strings. Quote such tickers in
the file, and reject a non-string rather than calling `str()` on it: `str(True)` is `'True'`,
which looks like a ticker, would be crawled forever and would never resolve.

**import-linter cannot see an import inside a docstring.** Contracts are checked against the
AST, and a doctest is a string literal. `ingest/backfill.py` had a doctest importing a concrete
calendar adapter while its layer contract said core-only; `lint-imports` passed, and pytest ran
the import anyway. The contract was not enforcing the thing its comment claimed. Doctests in a
port-only layer use fakes; a real adapter belongs in the test file, where the dependency is
visible.

**The `ruff-on-edit` hook auto-fixes on an intermediate state.**
Adding a name to `__all__` and its import in one edit, then defining the class in the next,
loses the import: at the moment of the first edit nothing used it, so `ruff check --fix`
deleted it as F401. Write a module's imports, `__all__` and definitions in **one** write, or
add the imports last. The hook is right; the two-step edit is the mistake.

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
