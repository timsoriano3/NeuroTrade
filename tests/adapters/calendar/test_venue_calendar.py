"""Tests for `VenueCalendar`, the `exchange_calendars`-backed `CalendarPort`.

**These tests are the pin.** `VenueCalendar`'s whole point is that a library
upgrade which silently rewrites holiday history fails the build instead of
quietly reshaping the corpus — see the module docstring. So the assertions
here are exact timestamps and exact counts against `exchange_calendars`
4.13.2, taken from running the installed library, never from plausibility.

Heavier on rejection than on happy paths otherwise: `Venue.SMART`, a range
outside the adapter's horizon and an inverted range must all raise, because a
silent wrong answer here corrupts a crawler's idea of what "complete" means.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from neurotrade.adapters.calendar.venue_calendar import VenueCalendar
from neurotrade.core.calendar import TradingSession
from neurotrade.core.clock import to_nanos
from neurotrade.core.events import BarInterval
from neurotrade.core.ports import CalendarPort
from neurotrade.core.types import Venue

# Every venue this adapter is expected to answer for, per its MIC mapping.
# SMART is excluded deliberately — it is an order route, not a listing venue.
_ALL_LISTING_VENUES = tuple(v for v in Venue if v is not Venue.SMART)


@pytest.fixture(scope="module")
def calendar() -> VenueCalendar:
    """One shared calendar over the adapter's default horizon.

    Building a calendar costs ~0.2s per venue, and these tests touch every
    venue in the mapping, so sharing one instance across the module keeps the
    file fast.
    """
    return VenueCalendar()


@pytest.fixture(scope="module")
def narrow_calendar() -> VenueCalendar:
    """A calendar whose horizon is exactly 2024, for the out-of-horizon cases."""
    return VenueCalendar(start=date(2024, 1, 1), end=date(2024, 12, 31))


# ── Exact session bounds ─────────────────────────────────────
#
# Built with `to_nanos` on an explicit `datetime(..., tzinfo=UTC)`, never a
# literal integer — the point is that the expected value is legible as a wall
# clock time, not a magic number that happens to match today.


def test_full_regular_session_bounds(calendar: VenueCalendar) -> None:
    """NYSE, 2024-07-01: an ordinary full day, 09:30-16:00 US/Eastern (EDT)."""
    session = calendar.session(Venue.NYSE, date(2024, 7, 1))
    assert session is not None
    assert session.open_ns == to_nanos(datetime(2024, 7, 1, 13, 30, tzinfo=UTC))
    assert session.close_ns == to_nanos(datetime(2024, 7, 1, 20, 0, tzinfo=UTC))
    assert session.is_early_close is False
    assert session.expected_bars(BarInterval.MIN_1) == 390


def test_half_day_session_bounds(calendar: VenueCalendar) -> None:
    """NASDAQ, 2024-07-03: the half day before Independence Day."""
    session = calendar.session(Venue.NASDAQ, date(2024, 7, 3))
    assert session is not None
    assert session.open_ns == to_nanos(datetime(2024, 7, 3, 13, 30, tzinfo=UTC))
    assert session.close_ns == to_nanos(datetime(2024, 7, 3, 17, 0, tzinfo=UTC))
    assert session.is_early_close is True
    assert session.expected_bars(BarInterval.MIN_1) == 210


def test_same_venue_opens_at_both_13_30_and_14_30_utc_across_the_year(
    calendar: VenueCalendar,
) -> None:
    """The 13:30 vs 14:30 open is daylight saving, not a venue difference.

    NASDAQ opens at 13:30 UTC in July (EDT) and 14:30 UTC in late November
    (EST) — same venue, same 09:30 local open, different UTC offset.
    """
    summer = calendar.session(Venue.NASDAQ, date(2024, 7, 8))
    autumn = calendar.session(Venue.NASDAQ, date(2024, 11, 29))
    assert summer is not None
    assert autumn is not None
    assert summer.open_ns == to_nanos(datetime(2024, 7, 8, 13, 30, tzinfo=UTC))
    assert autumn.open_ns == to_nanos(datetime(2024, 11, 29, 14, 30, tzinfo=UTC))


# ── expected_bars ─────────────────────────────────────────────


def test_expected_bars_for_a_full_day(calendar: VenueCalendar) -> None:
    session = calendar.session(Venue.NYSE, date(2024, 7, 8))
    assert session is not None
    assert session.expected_bars(BarInterval.MIN_1) == 390


def test_expected_bars_for_a_half_day(calendar: VenueCalendar) -> None:
    session = calendar.session(Venue.TSX, date(2024, 12, 24))
    assert session is not None
    assert session.expected_bars(BarInterval.MIN_1) == 210


# ── is_early_close ────────────────────────────────────────────


def test_is_early_close_true_on_a_us_venue(calendar: VenueCalendar) -> None:
    session = calendar.session(Venue.NASDAQ, date(2024, 11, 29))
    assert session is not None
    assert session.is_early_close is True


def test_is_early_close_false_on_a_us_venue(calendar: VenueCalendar) -> None:
    session = calendar.session(Venue.NASDAQ, date(2024, 7, 8))
    assert session is not None
    assert session.is_early_close is False


def test_is_early_close_true_on_a_canadian_venue(calendar: VenueCalendar) -> None:
    """TSX closes early on Christmas Eve."""
    session = calendar.session(Venue.TSX, date(2024, 12, 24))
    assert session is not None
    assert session.is_early_close is True


def test_is_early_close_false_on_a_canadian_venue(calendar: VenueCalendar) -> None:
    session = calendar.session(Venue.TSX, date(2024, 11, 28))
    assert session is not None
    assert session.is_early_close is False


# ── Closed days, including the cross-border cases ────────────
#
# TSX trades through US Thanksgiving; NYSE trades through Canada Day. A
# calendar that collapsed both venues onto one set of holidays would get one
# of these two dates wrong in each direction.


def test_us_holiday_closes_nyse(calendar: VenueCalendar) -> None:
    assert calendar.session(Venue.NYSE, date(2024, 7, 4)) is None  # Independence Day


def test_canadian_holiday_closes_tsx(calendar: VenueCalendar) -> None:
    assert calendar.session(Venue.TSX, date(2024, 7, 1)) is None  # Canada Day


def test_tsx_trades_through_us_thanksgiving(calendar: VenueCalendar) -> None:
    """NYSE is closed for Thanksgiving; TSX, unaffected, trades a full day."""
    assert calendar.session(Venue.NYSE, date(2024, 11, 28)) is None
    tsx_session = calendar.session(Venue.TSX, date(2024, 11, 28))
    assert tsx_session is not None
    assert tsx_session.is_early_close is False


def test_nyse_trades_through_canada_day(calendar: VenueCalendar) -> None:
    """TSX is closed for Canada Day; NYSE, unaffected, trades a full day."""
    assert calendar.session(Venue.TSX, date(2024, 7, 1)) is None
    nyse_session = calendar.session(Venue.NYSE, date(2024, 7, 1))
    assert nyse_session is not None
    assert nyse_session.is_early_close is False


# ── sessions(): counts and exact ranges ───────────────────────


@pytest.mark.parametrize("venue", [Venue.NYSE, Venue.NASDAQ, Venue.TSX, Venue.TSXV])
def test_2024_has_252_sessions(calendar: VenueCalendar, venue: Venue) -> None:
    sessions = calendar.sessions(venue, date(2024, 1, 1), date(2024, 12, 31))
    assert len(sessions) == 252


def test_tsxv_resolves_to_the_same_calendar_as_tsx(calendar: VenueCalendar) -> None:
    """TSXV is a distinct listing venue but shares TSX's underlying schedule."""
    tsx = calendar.sessions(Venue.TSX, date(2024, 1, 1), date(2024, 12, 31))
    tsxv = calendar.sessions(Venue.TSXV, date(2024, 1, 1), date(2024, 12, 31))
    assert tsx == tsxv


def test_nyse_sessions_across_the_july_4th_week(calendar: VenueCalendar) -> None:
    """The 1st through 8th: a holiday (4th) and a weekend (6th-7th) both absent."""
    sessions = calendar.sessions(Venue.NYSE, date(2024, 7, 1), date(2024, 7, 8))
    assert sessions == (
        date(2024, 7, 1),
        date(2024, 7, 2),
        date(2024, 7, 3),
        date(2024, 7, 5),
        date(2024, 7, 8),
    )


# ── Venue.SMART is not a listing venue ───────────────────────


def test_sessions_rejects_smart(calendar: VenueCalendar) -> None:
    with pytest.raises(ValueError, match="order route"):
        calendar.sessions(Venue.SMART, date(2024, 7, 1), date(2024, 7, 8))


def test_session_rejects_smart(calendar: VenueCalendar) -> None:
    with pytest.raises(ValueError, match="order route"):
        calendar.session(Venue.SMART, date(2024, 7, 8))


# ── Horizon enforcement ───────────────────────────────────────
#
# A date the adapter was not built to cover must raise rather than answer —
# see `_check_horizon`'s docstring on why "no sessions" is a claim the
# adapter is not entitled to make outside its range.


def test_sessions_rejects_a_date_past_the_horizon(narrow_calendar: VenueCalendar) -> None:
    with pytest.raises(ValueError, match="outside the calendar horizon"):
        narrow_calendar.sessions(Venue.NYSE, date(2024, 12, 1), date(2025, 1, 15))


def test_sessions_rejects_a_date_before_the_horizon(narrow_calendar: VenueCalendar) -> None:
    with pytest.raises(ValueError, match="outside the calendar horizon"):
        narrow_calendar.sessions(Venue.NYSE, date(2023, 12, 15), date(2024, 1, 15))


def test_session_rejects_a_date_past_the_horizon(narrow_calendar: VenueCalendar) -> None:
    with pytest.raises(ValueError, match="outside the calendar horizon"):
        narrow_calendar.session(Venue.NYSE, date(2025, 1, 2))


def test_session_rejects_a_date_before_the_horizon(narrow_calendar: VenueCalendar) -> None:
    with pytest.raises(ValueError, match="outside the calendar horizon"):
        narrow_calendar.session(Venue.NYSE, date(2023, 12, 29))


# ── Inverted ranges ───────────────────────────────────────────


def test_sessions_rejects_an_inverted_range(calendar: VenueCalendar) -> None:
    with pytest.raises(ValueError, match="range must move forwards"):
        calendar.sessions(Venue.NYSE, date(2024, 7, 8), date(2024, 7, 1))


def test_constructor_rejects_an_inverted_range() -> None:
    with pytest.raises(ValueError, match="calendar range must move forwards"):
        VenueCalendar(start=date(2024, 12, 31), end=date(2024, 1, 1))


def test_constructor_rejects_an_equal_start_and_end() -> None:
    """`end` must be strictly after `start` — a single-day horizon is refused."""
    with pytest.raises(ValueError, match="calendar range must move forwards"):
        VenueCalendar(start=date(2024, 1, 1), end=date(2024, 1, 1))


# ── CalendarPort conformance ──────────────────────────────────


def test_satisfies_calendar_port_structurally() -> None:
    """No inheritance, no registration — the shape alone is what's checked."""
    assert isinstance(VenueCalendar(), CalendarPort)


# ── Every mapped venue answers ────────────────────────────────


@pytest.mark.parametrize("venue", _ALL_LISTING_VENUES)
def test_every_listing_venue_answers_for_a_known_session(
    calendar: VenueCalendar, venue: Venue
) -> None:
    """A venue added to the `Venue` enum without a matching MIC must fail here
    rather than at some future crawler run."""
    session = calendar.session(venue, date(2024, 7, 8))
    assert isinstance(session, TradingSession)
