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

from datetime import UTC, date, datetime, timedelta

import pytest

from neurotrade.adapters.calendar.venue_calendar import VenueCalendar
from neurotrade.core.calendar import TradingSession
from neurotrade.core.clock import to_nanos
from neurotrade.core.events import BarInterval
from neurotrade.core.types import Symbol, Venue
from neurotrade.core.universe import Universe
from neurotrade.ingest.backfill import (
    BackfillCell,
    BackfillWindow,
    plan_backfill,
    plan_windows,
)

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


# ── BackfillWindow: validation ────────────────────────────────


def _cell(symbol: Symbol, day: date, *, held: int = 0) -> BackfillCell:
    return BackfillCell(
        symbol=symbol,
        session=_session(symbol.venue, day),
        interval=BarInterval.MIN_1,
        held_bars=held,
    )


def _window(symbol: Symbol, *days: date) -> BackfillWindow:
    return BackfillWindow(
        symbol=symbol,
        interval=BarInterval.MIN_1,
        cells=tuple(_cell(symbol, day) for day in days),
    )


def test_a_window_covering_no_sessions_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"must cover at least one session"):
        BackfillWindow(symbol=AAPL, interval=BarInterval.MIN_1, cells=())


def test_a_window_holding_another_symbols_cell_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"does not belong in a"):
        BackfillWindow(
            symbol=AAPL,
            interval=BarInterval.MIN_1,
            cells=(_cell(AAPL, MON), _cell(MSFT, TUE)),
        )


def test_a_window_holding_another_intervals_cell_is_rejected() -> None:
    five_minute = BackfillCell(
        symbol=AAPL, session=_session(Venue.NASDAQ, MON), interval=BarInterval.MIN_5, held_bars=0
    )
    with pytest.raises(ValueError, match=r"does not belong in a"):
        BackfillWindow(symbol=AAPL, interval=BarInterval.MIN_1, cells=(five_minute,))


@pytest.mark.parametrize("days", [(TUE, MON), (MON, MON)])
def test_cells_out_of_ascending_order_or_repeated_are_rejected(days: tuple[date, ...]) -> None:
    """Order is load-bearing: the bars a request returns are split across the
    cells by walking both in step, so an unordered window would file bars
    under the wrong session."""
    with pytest.raises(ValueError, match=r"must ascend without repeats"):
        _window(AAPL, *days)


# ── BackfillWindow: bounds ────────────────────────────────────


def test_bounds_span_the_oldest_open_to_the_newest_close() -> None:
    window = _window(AAPL, MON, WED)

    assert window.start_ns == _session(Venue.NASDAQ, MON).open_ns
    assert window.end_ns == _session(Venue.NASDAQ, WED).close_ns
    assert (window.first_date, window.last_date) == (MON, WED)


def test_span_days_counts_calendar_days_inclusive_of_both_ends() -> None:
    """Calendar rather than trading days, because that is the unit a feed's
    duration limit is expressed in: MON..WED is three days, not two sessions."""
    assert _window(AAPL, MON, WED).span_days == 3
    assert _window(AAPL, MON).span_days == 1


def test_has_untouched_is_true_when_any_session_has_nothing_on_disk() -> None:
    mixed = BackfillWindow(
        symbol=AAPL,
        interval=BarInterval.MIN_1,
        cells=(_cell(AAPL, MON, held=200), _cell(AAPL, TUE, held=0)),
    )
    all_short = BackfillWindow(
        symbol=AAPL,
        interval=BarInterval.MIN_1,
        cells=(_cell(AAPL, MON, held=200), _cell(AAPL, TUE, held=1)),
    )

    assert mixed.has_untouched
    assert not all_short.has_untouched


# ── plan_windows: grouping ────────────────────────────────────


def test_consecutive_sessions_of_one_symbol_become_one_window() -> None:
    windows = list(plan_windows([_cell(AAPL, WED), _cell(AAPL, TUE), _cell(AAPL, MON)]))

    assert len(windows) == 1
    assert [cell.session_date for cell in windows[0].cells] == [MON, TUE, WED]


def test_each_symbol_gets_its_own_window_within_a_span() -> None:
    """A window is one request and a request is for one instrument."""
    windows = list(plan_windows([_cell(AAPL, MON), _cell(MSFT, MON)]))

    assert [window.symbol for window in windows] == [AAPL, MSFT]


def test_sessions_further_apart_than_the_cap_fall_into_separate_windows() -> None:
    far = date(2024, 5, 6)

    windows = list(plan_windows([_cell(AAPL, far), _cell(AAPL, MON)]))

    assert [(w.first_date, w.last_date) for w in windows] == [(far, far), (MON, MON)]


def test_the_cap_is_calendar_days_not_sessions() -> None:
    """Thirty calendar days is about twenty-two sessions, and the limit the
    feed enforces is the calendar one."""
    inside = date(2024, 4, 2)  # MON + 29 days
    outside = date(2024, 4, 3)  # MON + 30 days

    assert len(list(plan_windows([_cell(AAPL, inside), _cell(AAPL, MON)]))) == 1
    assert len(list(plan_windows([_cell(AAPL, outside), _cell(AAPL, MON)]))) == 2


def test_window_days_of_one_reproduces_a_request_per_session() -> None:
    windows = list(plan_windows([_cell(AAPL, TUE), _cell(AAPL, MON)], window_days=1))

    assert [w.first_date for w in windows] == [TUE, MON]
    assert all(len(w.cells) == 1 for w in windows)


def test_a_gap_inside_a_window_is_spanned_not_split() -> None:
    """The window holds only the *missing* sessions; the request covers the
    whole span anyway, and a session already complete simply has no cell."""
    windows = list(plan_windows([_cell(AAPL, WED), _cell(AAPL, MON)]))

    assert len(windows) == 1
    assert [cell.session_date for cell in windows[0].cells] == [MON, WED]


def test_an_empty_plan_yields_no_windows() -> None:
    assert list(plan_windows([])) == []


# ── plan_windows: ordering ────────────────────────────────────


def test_spans_are_newest_first_with_every_symbol_inside_one_before_the_next() -> None:
    """`plan_backfill`'s recency-major order at a coarser grain. Grouping
    symbol-major would fetch five years of AAPL before touching MSFT, and a
    cross-sectional study cannot use that corpus."""
    far = date(2024, 5, 6)
    cells = [
        _cell(AAPL, far),
        _cell(MSFT, far),
        _cell(AAPL, MON),
        _cell(MSFT, MON),
    ]

    windows = list(plan_windows(cells))

    assert [(w.symbol.ticker, w.first_date) for w in windows] == [
        ("AAPL", far),
        ("MSFT", far),
        ("AAPL", MON),
        ("MSFT", MON),
    ]


def test_spans_are_anchored_on_the_newest_date_so_the_partial_one_is_oldest() -> None:
    """A crawl killed early most needs the recent end whole, so the span that
    ends up shorter than the cap must be the oldest one."""
    days = [date(2024, 3, 4) + timedelta(days=n) for n in range(0, 40, 2)]

    windows = list(plan_windows([_cell(AAPL, day) for day in reversed(days)]))

    assert [w.span_days for w in windows] == [29, 9]


def test_grouping_is_deterministic_across_runs() -> None:
    cells = [_cell(symbol, day) for day in (WED, TUE, MON) for symbol in (AAPL, MSFT, JPM)]

    first = [(w.symbol, w.first_date, w.last_date) for w in plan_windows(cells)]
    second = [(w.symbol, w.first_date, w.last_date) for w in plan_windows(cells)]

    assert first == second


# ── plan_windows: validation ──────────────────────────────────


@pytest.mark.parametrize("window_days", [0, -1])
def test_a_non_positive_window_days_raises_value_error(window_days: int) -> None:
    with pytest.raises(ValueError, match=r"window_days must be at least 1"):
        list(plan_windows([_cell(AAPL, MON)], window_days=window_days))


def test_mixing_bar_sizes_in_one_plan_raises_value_error() -> None:
    """A window is one request, and one request asks for one bar size."""
    daily = BackfillCell(
        symbol=AAPL, session=_session(Venue.NASDAQ, MON), interval=BarInterval.DAY_1, held_bars=0
    )

    with pytest.raises(ValueError, match=r"one plan, one bar size"):
        list(plan_windows([_cell(AAPL, MON), daily]))
