"""Tests for `BackfillCell` and `plan_backfill`.

The plan has to be testable without an IBKR connection or a Parquet corpus, so
most cases here use an in-memory calendar and catalog rather than the real
adapters — that testability is itself the point of the port abstraction, and
one test (the cross-venue holiday case) uses the real `VenueCalendar` to prove
the fakes agree with it.

Heavier on rejection and on the ordering guarantees than on the happy path:
`plan_backfill`'s entire reason to exist is the recency-major property — every
symbol's newest gap before any symbol's older one — and a subtle ordering bug
would still "work" while quietly reintroducing the alphabetically-early-only
corpus the module exists to prevent.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from neurotrade.adapters.calendar.venue_calendar import VenueCalendar
from neurotrade.core.calendar import TradingSession
from neurotrade.core.clock import to_nanos
from neurotrade.core.events import BarInterval
from neurotrade.core.types import Symbol, Venue
from neurotrade.core.universe import Universe
from neurotrade.ingest.backfill import BackfillCell, plan_backfill

AAPL = Symbol("AAPL", Venue.NASDAQ)
MSFT = Symbol("MSFT", Venue.NASDAQ)
SHOP = Symbol("SHOP", Venue.TSX)
JPM = Symbol("JPM", Venue.NYSE)

MON = date(2024, 3, 4)
TUE = date(2024, 3, 5)
WED = date(2024, 3, 6)


def _session(venue: Venue, day: date, *, minutes: int = 390) -> TradingSession:
    """A full (by default) session for `venue`/`day`, sized in whole minutes."""
    open_ns = to_nanos(datetime(day.year, day.month, day.day, 13, 30, tzinfo=UTC))
    return TradingSession(
        venue=venue,
        session_date=day,
        open_ns=open_ns,
        close_ns=open_ns + minutes * BarInterval.MIN_1.nanos,
        is_early_close=False,
    )


class FakeCalendar:
    """An in-memory `CalendarPort`, keyed by venue then date — no exchange_calendars.

    None of these fakes imports or subclasses anything from core.ports — the
    dependency arrow runs from adapters to core, never back.
    """

    def __init__(self, sessions: dict[Venue, dict[date, TradingSession]]) -> None:
        self._sessions = sessions

    def sessions(self, venue: Venue, start: date, end: date) -> tuple[date, ...]:
        by_date = self._sessions.get(venue, {})
        return tuple(sorted(d for d in by_date if start <= d <= end))

    def session(self, venue: Venue, session_date: date) -> TradingSession | None:
        return self._sessions.get(venue, {}).get(session_date)


class LyingCalendar:
    """Lists a date as a session, then denies it — the discrepancy the plan must
    refuse to paper over rather than fetch as a zero-length window."""

    def __init__(self, venue: Venue, dates: tuple[date, ...]) -> None:
        self._venue = venue
        self._dates = dates

    def sessions(self, venue: Venue, start: date, end: date) -> tuple[date, ...]:
        return self._dates if venue is self._venue else ()

    def session(self, venue: Venue, session_date: date) -> TradingSession | None:
        return None


class FakeCatalog:
    """An in-memory `CatalogPort` that records which symbols it was asked about."""

    def __init__(self, counts: dict[Symbol, dict[date, int]] | None = None) -> None:
        self._counts = counts or {}
        self.calls: list[Symbol] = []

    def bar_counts(self, symbol: Symbol, interval: BarInterval) -> dict[date, int]:
        self.calls.append(symbol)
        return self._counts.get(symbol, {})


# ── BackfillCell ─────────────────────────────────────────────


def test_rejects_a_symbol_not_listed_on_the_sessions_venue() -> None:
    """AAPL is a NASDAQ listing; a TSX session is not its session."""
    with pytest.raises(ValueError, match="is not listed on"):
        BackfillCell(
            symbol=AAPL,
            session=_session(Venue.TSX, MON),
            interval=BarInterval.MIN_1,
            held_bars=0,
        )


def test_rejects_a_negative_held_count() -> None:
    with pytest.raises(ValueError, match="held_bars must not be negative"):
        BackfillCell(
            symbol=AAPL,
            session=_session(Venue.NASDAQ, MON),
            interval=BarInterval.MIN_1,
            held_bars=-1,
        )


def test_expected_bars_comes_from_the_session() -> None:
    cell = BackfillCell(
        symbol=AAPL,
        session=_session(Venue.NASDAQ, MON, minutes=210),
        interval=BarInterval.MIN_1,
        held_bars=0,
    )
    assert cell.expected_bars == 210


def test_missing_bars_is_the_shortfall() -> None:
    cell = BackfillCell(
        symbol=AAPL,
        session=_session(Venue.NASDAQ, MON),
        interval=BarInterval.MIN_1,
        held_bars=389,
    )
    assert cell.missing_bars == 1


def test_missing_bars_is_floored_at_zero_when_held_exceeds_expected() -> None:
    """Extended-hours bars can push held past the regular-session count; that
    is not a negative shortfall."""
    cell = BackfillCell(
        symbol=AAPL,
        session=_session(Venue.NASDAQ, MON),
        interval=BarInterval.MIN_1,
        held_bars=400,
    )
    assert cell.missing_bars == 0


def test_is_untouched_true_when_nothing_is_held() -> None:
    cell = BackfillCell(
        symbol=AAPL,
        session=_session(Venue.NASDAQ, MON),
        interval=BarInterval.MIN_1,
        held_bars=0,
    )
    assert cell.is_untouched


def test_is_untouched_false_once_anything_is_held() -> None:
    cell = BackfillCell(
        symbol=AAPL,
        session=_session(Venue.NASDAQ, MON),
        interval=BarInterval.MIN_1,
        held_bars=1,
    )
    assert not cell.is_untouched


def test_start_ns_and_end_ns_map_to_the_session_bounds() -> None:
    session = _session(Venue.NASDAQ, MON)
    cell = BackfillCell(symbol=AAPL, session=session, interval=BarInterval.MIN_1, held_bars=0)
    assert cell.start_ns == session.open_ns
    assert cell.end_ns == session.close_ns


def test_session_date_matches_the_session() -> None:
    session = _session(Venue.NASDAQ, MON)
    cell = BackfillCell(symbol=AAPL, session=session, interval=BarInterval.MIN_1, held_bars=0)
    assert cell.session_date == MON


def test_str_shows_symbol_date_interval_and_held_over_expected() -> None:
    cell = BackfillCell(
        symbol=AAPL,
        session=_session(Venue.NASDAQ, MON),
        interval=BarInterval.MIN_1,
        held_bars=200,
    )
    assert str(cell) == "AAPL.NASDAQ 2024-03-04 1m 200/390"


# ── plan_backfill: ordering ──────────────────────────────────


def test_ordering_is_session_date_descending_then_universe_order() -> None:
    calendar = FakeCalendar(
        {Venue.NASDAQ: {day: _session(Venue.NASDAQ, day) for day in (MON, TUE, WED)}}
    )
    catalog = FakeCatalog()
    universe = Universe([MSFT, AAPL])  # sorts to (AAPL, MSFT)

    plan = list(plan_backfill(universe, calendar, catalog, start=MON, end=WED))

    assert [(cell.symbol, cell.session_date) for cell in plan] == [
        (AAPL, WED),
        (MSFT, WED),
        (AAPL, TUE),
        (MSFT, TUE),
        (AAPL, MON),
        (MSFT, MON),
    ]


def test_recency_major_beats_symbol_major() -> None:
    """Every symbol's newest gap is offered before any symbol's older one.

    AAPL is missing both Monday and Wednesday; MSFT is missing only Tuesday.
    A symbol-major queue would emit both AAPL sessions back to back; the
    recency-major property instead interleaves MSFT's single Tuesday gap
    between them — proving the ordering is by date first, not by symbol.
    """
    calendar = FakeCalendar(
        {Venue.NASDAQ: {day: _session(Venue.NASDAQ, day) for day in (MON, TUE, WED)}}
    )
    catalog = FakeCatalog({AAPL: {TUE: 390}, MSFT: {MON: 390, WED: 390}})
    universe = Universe([AAPL, MSFT])

    plan = list(plan_backfill(universe, calendar, catalog, start=MON, end=WED))

    assert [(cell.symbol, cell.session_date) for cell in plan] == [
        (AAPL, WED),
        (MSFT, TUE),
        (AAPL, MON),
    ]


# ── plan_backfill: completeness ──────────────────────────────


def test_a_complete_session_is_absent_from_the_plan() -> None:
    calendar = FakeCalendar({Venue.NASDAQ: {MON: _session(Venue.NASDAQ, MON)}})
    catalog = FakeCatalog({AAPL: {MON: 390}})
    plan = list(plan_backfill(Universe([AAPL]), calendar, catalog, start=MON, end=MON))
    assert plan == []


def test_a_session_one_bar_short_is_present_with_the_right_held_count() -> None:
    calendar = FakeCalendar({Venue.NASDAQ: {MON: _session(Venue.NASDAQ, MON)}})
    catalog = FakeCatalog({AAPL: {MON: 389}})
    plan = list(plan_backfill(Universe([AAPL]), calendar, catalog, start=MON, end=MON))
    assert len(plan) == 1
    assert plan[0].held_bars == 389
    assert plan[0].missing_bars == 1


def test_a_session_holding_more_than_expected_is_treated_as_complete() -> None:
    calendar = FakeCalendar({Venue.NASDAQ: {MON: _session(Venue.NASDAQ, MON)}})
    catalog = FakeCatalog({AAPL: {MON: 400}})
    plan = list(plan_backfill(Universe([AAPL]), calendar, catalog, start=MON, end=MON))
    assert plan == []


def test_an_empty_plan_when_the_corpus_is_complete() -> None:
    calendar = FakeCalendar(
        {Venue.NASDAQ: {day: _session(Venue.NASDAQ, day) for day in (MON, TUE, WED)}}
    )
    catalog = FakeCatalog(
        {AAPL: {MON: 390, TUE: 390, WED: 390}, MSFT: {MON: 390, TUE: 390, WED: 390}}
    )
    plan = list(plan_backfill(Universe([AAPL, MSFT]), calendar, catalog, start=MON, end=WED))
    assert plan == []


# ── plan_backfill: holidays ───────────────────────────────────


def test_a_holiday_is_absent_from_the_plan() -> None:
    """Tuesday is not a session at all — not present-and-short, just absent."""
    calendar = FakeCalendar(
        {Venue.NASDAQ: {MON: _session(Venue.NASDAQ, MON), WED: _session(Venue.NASDAQ, WED)}}
    )
    catalog = FakeCatalog()
    plan = list(plan_backfill(Universe([AAPL]), calendar, catalog, start=MON, end=WED))
    assert [cell.session_date for cell in plan] == [WED, MON]


def test_tsx_trades_through_us_thanksgiving_while_nyse_is_closed() -> None:
    """Canonical cross-venue disagreement: real `VenueCalendar`, not a fake.

    Each venue's calendar is genuinely independent — TSX offers its Thursday
    session while NYSE, closed for Thanksgiving, offers nothing at all.
    """
    calendar = VenueCalendar()
    catalog = FakeCatalog()
    universe = Universe([SHOP, JPM])
    thanksgiving = date(2024, 11, 28)

    plan = list(plan_backfill(universe, calendar, catalog, start=thanksgiving, end=thanksgiving))

    assert [cell.symbol for cell in plan] == [SHOP]


# ── plan_backfill: validation ─────────────────────────────────


def test_end_before_start_raises_value_error() -> None:
    calendar = FakeCalendar({})
    catalog = FakeCatalog()
    with pytest.raises(ValueError, match="range must move forwards"):
        list(plan_backfill(Universe([AAPL]), calendar, catalog, start=TUE, end=MON))


def test_a_calendar_that_lies_raises_value_error() -> None:
    """`sessions()` lists Monday; `session()` then denies it exists.

    That discrepancy must raise rather than be fetched as a zero-length
    window, which is what a silent `None` would otherwise produce.
    """
    calendar = LyingCalendar(Venue.NASDAQ, (MON,))
    catalog = FakeCatalog()
    with pytest.raises(ValueError, match="then reported it closed"):
        list(plan_backfill(Universe([AAPL]), calendar, catalog, start=MON, end=MON))


# ── plan_backfill: catalog access pattern ─────────────────────


def test_catalog_is_consulted_once_per_symbol_not_once_per_session() -> None:
    """A 2000-symbol universe would otherwise issue millions of queries."""
    calendar = FakeCalendar(
        {Venue.NASDAQ: {day: _session(Venue.NASDAQ, day) for day in (MON, TUE, WED)}}
    )
    catalog = FakeCatalog()
    universe = Universe([AAPL, MSFT])

    list(plan_backfill(universe, calendar, catalog, start=MON, end=WED))

    assert catalog.calls == [AAPL, MSFT]


# ── plan_backfill: determinism ────────────────────────────────


def test_ordering_is_deterministic_across_runs() -> None:
    """Regression guard for the `sorted(set(...))` in the module: nothing may
    depend on set or dict iteration order, which varies run to run."""
    calendar = FakeCalendar(
        {Venue.NASDAQ: {day: _session(Venue.NASDAQ, day) for day in (MON, TUE, WED)}}
    )
    catalog = FakeCatalog()
    universe = Universe([AAPL, MSFT])

    first = [
        (c.symbol, c.session_date)
        for c in plan_backfill(universe, calendar, catalog, start=MON, end=WED)
    ]
    second = [
        (c.symbol, c.session_date)
        for c in plan_backfill(universe, calendar, catalog, start=MON, end=WED)
    ]

    assert first == second
