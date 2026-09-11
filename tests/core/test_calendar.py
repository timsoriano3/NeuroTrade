"""Tests for `TradingSession` and `CalendarPort`.

Heavier on rejection cases than on happy paths: the point of `TradingSession`
is to make a class of bug unrepresentable, so what matters is that bad values
are actually refused. The `contains`/`holds_bar` asymmetry gets the most
weight of all — it is a documented bug class (`Bar.ts_event` is a close, not
an open), not a nicety.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from neurotrade.adapters.storage.duckdb_catalog import Coverage
from neurotrade.core.calendar import TradingSession
from neurotrade.core.clock import Nanos, to_nanos
from neurotrade.core.events import BarInterval
from neurotrade.core.ports import CalendarPort
from neurotrade.core.types import Symbol, Venue

# A full NYSE regular session: 13:30-20:00 UTC == 09:30-16:00 US/Eastern (EDT).
_FULL_OPEN = to_nanos(datetime(2024, 7, 8, 13, 30, tzinfo=UTC))
_FULL_CLOSE = to_nanos(datetime(2024, 7, 8, 20, 0, tzinfo=UTC))


def _session(
    *,
    venue: Venue = Venue.NYSE,
    session_date: date = date(2024, 7, 8),
    open_ns: Nanos = _FULL_OPEN,
    close_ns: Nanos = _FULL_CLOSE,
    is_early_close: bool = False,
) -> TradingSession:
    return TradingSession(
        venue=venue,
        session_date=session_date,
        open_ns=open_ns,
        close_ns=close_ns,
        is_early_close=is_early_close,
    )


# ── Construction / validation ────────────────────────────────


def test_rejects_smart_as_venue() -> None:
    """SMART is IBKR's order router, not a venue with trading hours."""
    with pytest.raises(ValueError, match="order route"):
        _session(venue=Venue.SMART)


@pytest.mark.parametrize(
    "close_ns",
    [_FULL_OPEN, _FULL_OPEN - 1],
    ids=["equal-to-open", "before-open"],
)
def test_rejects_close_not_after_open(close_ns: Nanos) -> None:
    with pytest.raises(ValueError, match="close after it opens"):
        _session(close_ns=close_ns)


def test_rejects_span_longer_than_24h() -> None:
    too_long_close = _FULL_OPEN + 24 * 60 * 60 * 1_000_000_000 + 1
    with pytest.raises(ValueError, match="more than a day"):
        _session(close_ns=too_long_close)


def test_accepts_a_span_of_exactly_24h() -> None:
    """The bound is loose on purpose — it catches nonsense, not real schedules."""
    exactly_24h_close = _FULL_OPEN + 24 * 60 * 60 * 1_000_000_000
    session = _session(close_ns=exactly_24h_close)
    assert session.duration_ns == 24 * 60 * 60 * 1_000_000_000


def test_is_frozen() -> None:
    session = _session()
    with pytest.raises(AttributeError):
        session.is_early_close = True  # type: ignore[misc]


# ── expected_bars ─────────────────────────────────────────────


def test_full_us_session_bar_counts() -> None:
    session = _session()
    assert session.expected_bars(BarInterval.MIN_1) == 390
    assert session.expected_bars(BarInterval.MIN_30) == 13


def test_early_close_session_bar_count() -> None:
    early = _session(
        session_date=date(2024, 7, 3),
        open_ns=to_nanos(datetime(2024, 7, 3, 13, 30, tzinfo=UTC)),
        close_ns=to_nanos(datetime(2024, 7, 3, 17, 0, tzinfo=UTC)),
        is_early_close=True,
    )
    assert early.expected_bars(BarInterval.MIN_1) == 210


def test_partial_trailing_bar_is_floored_not_counted() -> None:
    """A span that does not divide evenly must not round up to a phantom bar."""
    odd_span = _session(close_ns=_FULL_OPEN + 90 * 1_000_000_000)  # 1.5 minutes
    assert odd_span.duration_ns == 90 * 1_000_000_000
    assert odd_span.expected_bars(BarInterval.MIN_1) == 1  # not 2


def test_expected_bars_is_what_coverage_is_complete_wants() -> None:
    """`expected_bars` exists to feed `Coverage.is_complete` — tie them together.

    A corpus holding exactly the expected count of 1-minute bars must be
    reported complete; one short by a single bar must not.
    """
    session = _session()
    expected = session.expected_bars(BarInterval.MIN_1)
    symbol = Symbol("AAPL", Venue.NYSE)

    held = Coverage(
        symbol=symbol,
        session_date=session.session_date,
        interval=BarInterval.MIN_1,
        bar_count=expected,
        first_ts=session.open_ns + BarInterval.MIN_1.nanos,
        last_ts=session.close_ns,
        sources=("ibkr",),
    )
    assert held.is_complete(expected)

    short = Coverage(
        symbol=symbol,
        session_date=session.session_date,
        interval=BarInterval.MIN_1,
        bar_count=expected - 1,
        first_ts=session.open_ns + BarInterval.MIN_1.nanos,
        last_ts=session.close_ns,
        sources=("ibkr",),
    )
    assert not short.is_complete(expected)


# ── contains vs holds_bar ────────────────────────────────────
#
# `Bar.ts_event` is the bar's CLOSE. A bar stamped at the session open closed
# AT the bell, so it covers the minute *before* the session and belongs to the
# previous one. `contains` (inclusive both ends) answers "was the venue open
# at this instant"; `holds_bar` (open-exclusive, close-inclusive) answers
# "does this bar belong to this session". Conflating them shifts every
# session by one bar and corrupts the opening range.


def test_contains_is_inclusive_at_both_ends() -> None:
    session = _session()
    assert session.contains(session.open_ns) is True
    assert session.contains(session.close_ns) is True


def test_holds_bar_excludes_the_open() -> None:
    """A bar closing exactly at the open belongs to the *previous* session."""
    session = _session()
    assert session.holds_bar(session.open_ns) is False


def test_holds_bar_includes_the_close() -> None:
    """The final bar of the day closes exactly at the bell."""
    session = _session()
    assert session.holds_bar(session.close_ns) is True


def test_open_ns_differs_between_contains_and_holds_bar() -> None:
    """The asymmetry, stated as a single fact so it cannot be missed."""
    session = _session()
    assert session.contains(session.open_ns) != session.holds_bar(session.open_ns)


def test_timestamp_before_open_is_outside_both() -> None:
    session = _session()
    before = session.open_ns - 1
    assert session.contains(before) is False
    assert session.holds_bar(before) is False


def test_timestamp_after_close_is_outside_both() -> None:
    session = _session()
    after = session.close_ns + 1
    assert session.contains(after) is False
    assert session.holds_bar(after) is False


def test_first_real_bar_of_the_session_is_held() -> None:
    """The bar closing one minute after the open is the session's first bar."""
    session = _session()
    first_bar_close = session.open_ns + BarInterval.MIN_1.nanos
    assert session.holds_bar(first_bar_close) is True


# ── CalendarPort ──────────────────────────────────────────────
#
# None of these fakes imports or subclasses anything from core.ports — the
# dependency arrow runs from adapters to core, never back.


class FakeCalendar:
    def __init__(self, sessions_by_date: dict[date, TradingSession]) -> None:
        self._sessions_by_date = sessions_by_date

    def sessions(self, venue: Venue, start: date, end: date) -> tuple[date, ...]:
        return tuple(sorted(d for d in self._sessions_by_date if start <= d <= end))

    def session(self, venue: Venue, session_date: date) -> TradingSession | None:
        return self._sessions_by_date.get(session_date)


def test_fake_calendar_satisfies_the_port() -> None:
    """Conformance is structural — no inheritance, no registration."""
    fake = FakeCalendar({date(2024, 7, 8): _session()})
    assert isinstance(fake, CalendarPort)


def test_a_class_missing_a_method_does_not_satisfy_the_port() -> None:
    class Incomplete:
        def sessions(self, venue: Venue, start: date, end: date) -> tuple[date, ...]:
            return ()

    assert not isinstance(Incomplete(), CalendarPort)


def test_session_returns_none_for_a_closed_day() -> None:
    """A holiday is an ordinary negative answer, not an exception.

    The crawler asks this constantly; reserving exceptions for an unsupported
    venue is what keeps a holiday from looking like a fault.
    """
    fake = FakeCalendar({date(2024, 7, 8): _session()})
    assert fake.session(Venue.NYSE, date(2024, 7, 4)) is None  # Independence Day
