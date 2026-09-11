# The venue calendar

Two halves. `core/calendar.py` holds the shape of a trading day;
`adapters/calendar/venue_calendar.py` holds the facts. See `01-domain-model.doc.md`
for the domain half.

**Not in the spec.** §12.1 goes from "build the store" to "start the backfill crawler"
and never defines trading hours, holidays or half days. Five modules defer to "the venue
calendar" in comments and `DuckDBCatalog.missing_sessions` takes an expected-session list
with no source. This is that missing piece, and the crawler is blocked on it: a crawler
cannot know what to fetch without knowing which days should have data.

## `VenueCalendar` — the adapter

Implements `CalendarPort` structurally. Backed by `exchange_calendars`, the package most
of the Python backtesting ecosystem uses.

`_MIC_BY_VENUE` maps each `Venue` to an ISO 10383 market identifier code:

| Venue | MIC | Note |
|---|---|---|
| NASDAQ | XNAS | alias of XNYS in the library |
| NYSE | XNYS | canonical US equity calendar |
| AMEX | XASE | alias |
| ARCA | ARCX | alias |
| BATS | BATS | alias |
| TSX | XTSE | canonical Canadian calendar |
| TSXV | XTSX | alias of XTSE |
| SMART | — | absent; raises. An order route, not a listing venue |

US venues genuinely share holidays and hours, so the aliasing is correct rather than a
shortcut. Writing each venue out anyway costs five lines and makes a future divergence a
one-line change.

## The two decisions that matter

**The date range is a fixed constant, not the library default.** Left alone,
`get_calendar` builds twenty years back and one year forward *counted from today* —
verified: on 2026-09-10 a default XNYS calendar reported `last_session` 2027-09-10. Two
runs on different days would then disagree, which `CalendarPort` forbids. The adapter
always passes `_HISTORY_START` (2000-01-01) and `_HISTORY_END` (2035-12-31).

**A date outside that horizon raises rather than returning nothing.** An empty answer is a
claim the adapter is not entitled to make: a crawler told "no sessions" would mark the
whole span complete and never fetch it. Building the full 36-year span costs about 0.17s
per venue and is cached per instance.

## Why the tests assert exact timestamps

They are the pin. `uv.lock` fixes the version, but an upgrade that corrects a historical
holiday changes what "complete" means for data already on disk. Asserting exact opens,
closes and session counts makes that upgrade fail the build instead of quietly reshaping
the corpus.

Facts the tests encode, all verified against the installed library:

- NASDAQ 2024-07-03 — 13:30 to 17:00 UTC, early close, 210 one-minute bars.
- NASDAQ 2024-07-08 — 13:30 to 20:00 UTC, 390 bars.
- TSX 2024-11-28 — open through US Thanksgiving, 14:30 to 21:00 UTC.
- NYSE 2024-11-28 — closed.
- TSX 2024-07-01 — closed for Canada Day. NYSE traded.
- TSX 2024-12-24 — 14:30 to 18:00 UTC, early close.
- NYSE, NASDAQ, TSX and TSXV each had 252 sessions in 2024.

The 13:30 versus 14:30 opens are daylight saving, not a venue difference.

## Dependency

`exchange-calendars>=4.13.2`, exact version fixed by `uv.lock`. It pulls in pandas, which
was not previously a direct dependency. pandas appears in this codebase only as the type of
the timestamps the library hands back; nothing computes with it. Neither package ships a
`py.typed` marker, so both carry a mypy override in `pyproject.toml`.

## Not done yet

Nothing calls this. `sessions()` is the crawler's work queue once differenced against
`missing_sessions` — but a work queue needs two axes, and the other one is the symbol list.
`core/universe.py` is that second axis and landed first; the wiring itself, plus a CLI command
and a make target, follows. Note the seam: `missing_sessions` takes `list[date]` while the port returns
`tuple[date, ...]`, so one of the two needs widening when they are joined.
