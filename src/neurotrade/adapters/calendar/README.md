# adapters/calendar

Which days each venue traded, and between which times.

This is the only place in the system that knows a venue's holidays. Everything
else asks `CalendarPort` and gets `TradingSession` objects back, so the trading
core never learns that a third-party package is involved.

## Files

| File | What it does |
|---|---|
| `venue_calendar.py` | `VenueCalendar` — maps each `Venue` to its exchange calendar and answers both port methods |

## Why a library and not a table of dates

A hand-written holiday table looks small and is not. Each venue closes about ten
times a year, closes early a few more times, and occasionally closes for
something nobody planned — a state funeral, a hurricane. The rules also change:
Juneteenth became a US market holiday in 2022, and anything computed from
"weekday and not in this list" is wrong for every year before someone remembers
to edit the list.

`exchange_calendars` encodes all of it, tracks the venues' published schedules,
and is the same package most of the Python backtesting ecosystem uses.

## Two decisions worth knowing

**The date range is fixed, not relative to today.** Left to itself the library
builds a calendar covering twenty years back and one year forward, counted from
the current date, which means the same query answers differently next week.
`CalendarPort` requires determinism, so the adapter always passes an explicit
range and refuses any date outside it. Refusing matters more than it sounds:
a crawler told "no sessions" for a year the calendar cannot see would conclude
the venue never traded and mark the whole year complete.

**Every US venue maps to its own market identifier code even though they share
a schedule.** NYSE, NASDAQ, AMEX, ARCA and BATS keep identical holidays and
hours, and the library treats the codes as aliases. Writing the mapping out in
full costs five lines and makes a future divergence a one-line change rather
than an archaeology exercise. The same applies to TSX Venture, which currently
follows the Toronto Stock Exchange exactly.

`Venue.SMART` has no entry. It is IBKR's order router, not a listing venue, so
asking when it opened is a caller bug and raises.

## Testing

The tests assert exact opens, closes and session counts for specific 2024 dates.
That is deliberate. A library upgrade that corrects a historical holiday changes
what "complete" means for data already on disk, so it has to fail the build
rather than quietly reshape the corpus.
