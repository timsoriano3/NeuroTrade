"""`CalendarPort` backed by the `exchange_calendars` package.

Holiday schedules are facts about the outside world, and maintaining them by
hand is a slow-motion mistake: roughly ten holidays a year per venue, plus half
days, plus one-off closures like a state funeral or Hurricane Sandy, plus the
dates those rules changed. `exchange_calendars` already encodes all of it and is
maintained against the venues' own published schedules.

**Why the bounds are constants rather than defaults.** Left alone, a calendar
covers twenty years back and one year forward *counted from today*, so the same
code answers differently tomorrow. `CalendarPort` requires determinism for a
given library version, so this adapter always builds with the explicit range in
`_HISTORY_START` / `_HISTORY_END`, and a date outside it raises rather than
quietly returning nothing. A crawler that read "no sessions" for a year it
cannot see would record the venue as never having traded.

**Version pinning matters here.** An upgrade that corrects a historical holiday
changes what "complete" means for data already on disk. The dependency carries a
lower bound, `uv.lock` fixes the exact version, and the tests assert specific
session bounds so a change in history fails the build instead of silently
reshaping the corpus.
"""

from __future__ import annotations

from datetime import date
from typing import Final

import exchange_calendars as xcals
import pandas as pd

from neurotrade.core.calendar import TradingSession
from neurotrade.core.types import Venue

__all__ = ["VenueCalendar"]

# ISO 10383 Market Identifier Codes, which is how `exchange_calendars` names its
# calendars. Every US venue here resolves to the same underlying schedule — they
# are aliases of XNYS — and that is correct rather than lazy: US equity venues
# keep identical holidays and identical regular hours. TSXV resolves to XTSE for
# the same reason. Mapping each venue explicitly instead of collapsing them by
# country keeps the intent readable and means a future divergence is a one-line
# change here.
_MIC_BY_VENUE: Final[dict[Venue, str]] = {
    Venue.NASDAQ: "XNAS",
    Venue.NYSE: "XNYS",
    Venue.AMEX: "XASE",
    Venue.ARCA: "ARCX",
    Venue.BATS: "BATS",
    Venue.TSX: "XTSE",
    Venue.TSXV: "XTSX",
}
# Venue.SMART is deliberately absent: it is IBKR's order router, not a listing
# venue, so asking when it opened is a bug in the caller.

_HISTORY_START: Final = date(2000, 1, 1)
"""Earliest date this adapter will answer for. Well before any corpus we intend
to build (§12.1 targets three to five years), and fixed so that two runs on
different days agree."""

_HISTORY_END: Final = date(2035, 12, 31)
"""Latest date this adapter will answer for. A fixed horizon rather than a
rolling one, for the same reason. Building the full span costs a fraction of a
second and happens once per venue per process."""


class VenueCalendar:
    """Which days a venue traded, and between which times.

    Satisfies `CalendarPort` structurally — there is no import from `core.ports`
    and nothing to subclass.

    Calendars are built lazily and kept, because building one costs far more
    than querying it and the crawler asks the same venue thousands of times.

    Example:
        >>> calendar = VenueCalendar()
        >>> calendar.sessions(Venue.NYSE, date(2024, 7, 3), date(2024, 7, 5))
        (datetime.date(2024, 7, 3), datetime.date(2024, 7, 5))
        >>> calendar.session(Venue.NYSE, date(2024, 7, 4)) is None  # Independence Day
        True
    """

    def __init__(self, *, start: date = _HISTORY_START, end: date = _HISTORY_END) -> None:
        """Build an adapter answering for a fixed date range.

        Args:
            start: Earliest date any query may mention, inclusive.
            end: Latest date any query may mention, inclusive.

        Raises:
            ValueError: If the range is empty or inverted.
        """
        if end <= start:
            raise ValueError(f"calendar range must move forwards: {start} -> {end}")
        self._start = start
        self._end = end
        self._calendars: dict[Venue, xcals.ExchangeCalendar] = {}
        self._early_closes: dict[Venue, frozenset[date]] = {}

    def sessions(self, venue: Venue, start: date, end: date) -> tuple[date, ...]:
        """Dates the venue traded within an inclusive range.

        Args:
            venue: Listing venue.
            start: First date to consider, inclusive.
            end: Last date to consider, inclusive.

        Returns:
            Session dates in ascending order, weekends and holidays absent.

        Raises:
            ValueError: If the venue has no calendar, the range is inverted, or
                either endpoint falls outside this adapter's horizon.

        Example:
            >>> VenueCalendar().sessions(Venue.TSX, date(2024, 6, 28), date(2024, 7, 2))
            (datetime.date(2024, 6, 28), datetime.date(2024, 7, 2))

            Canada Day closed the TSX on the 1st; the 29th and 30th were a
            weekend. The same range on a US venue keeps the 1st:

            >>> VenueCalendar().sessions(Venue.NYSE, date(2024, 6, 28), date(2024, 7, 2))
            (datetime.date(2024, 6, 28), datetime.date(2024, 7, 1), datetime.date(2024, 7, 2))
        """
        if end < start:
            raise ValueError(f"range must move forwards: {start} -> {end}")
        self._check_horizon(start)
        self._check_horizon(end)
        index = self._calendar(venue).sessions_in_range(start, end)
        return tuple(timestamp.date() for timestamp in index)

    def session(self, venue: Venue, session_date: date) -> TradingSession | None:
        """The bounds of one session, or `None` if the venue was closed.

        Args:
            venue: Listing venue.
            session_date: The date to look up.

        Returns:
            The session, or `None` if that date was not a trading day.

        Raises:
            ValueError: If the venue has no calendar, or the date falls outside
                this adapter's horizon. A holiday is not an error.

        Example:
            The US half day before Independence Day, 09:30 to 13:00 Eastern:

            >>> from neurotrade.core.events import BarInterval
            >>> half_day = VenueCalendar().session(Venue.NASDAQ, date(2024, 7, 3))
            >>> half_day.is_early_close, half_day.expected_bars(BarInterval.MIN_1)
            (True, 210)

            An ordinary session on the same venue runs the full 6.5 hours:

            >>> full = VenueCalendar().session(Venue.NASDAQ, date(2024, 7, 8))
            >>> full.is_early_close, full.expected_bars(BarInterval.MIN_1)
            (False, 390)
        """
        self._check_horizon(session_date)
        calendar = self._calendar(venue)
        if not calendar.is_session(session_date):
            return None
        return TradingSession(
            venue=venue,
            session_date=session_date,
            open_ns=_to_nanos(calendar.session_open(session_date)),
            close_ns=_to_nanos(calendar.session_close(session_date)),
            is_early_close=session_date in self._venue_early_closes(venue),
        )

    def _calendar(self, venue: Venue) -> xcals.ExchangeCalendar:
        """The underlying calendar for a venue, built once and kept."""
        cached = self._calendars.get(venue)
        if cached is not None:
            return cached
        mic = _MIC_BY_VENUE.get(venue)
        if mic is None:
            raise ValueError(
                f"{venue} has no exchange calendar — it is an order route, not a listing venue"
            )
        built = xcals.get_calendar(mic, start=self._start, end=self._end)
        self._calendars[venue] = built
        return built

    def _venue_early_closes(self, venue: Venue) -> frozenset[date]:
        """Half days for a venue, as plain dates.

        Held as a set because the alternative is a linear scan of every early
        close the venue has ever had, once per session the crawler looks at.
        """
        cached = self._early_closes.get(venue)
        if cached is not None:
            return cached
        closes = frozenset(timestamp.date() for timestamp in self._calendar(venue).early_closes)
        self._early_closes[venue] = closes
        return closes

    def _check_horizon(self, day: date) -> None:
        """Reject a date the underlying calendar was not built to cover.

        Raising beats returning nothing: outside the horizon the library has no
        opinion, and "no sessions" is a claim the adapter is not entitled to
        make.
        """
        if not self._start <= day <= self._end:
            raise ValueError(f"{day} is outside the calendar horizon {self._start}..{self._end}")


def _to_nanos(timestamp: pd.Timestamp) -> int:
    """UTC nanoseconds from a pandas `Timestamp`.

    Taken from the integer the `Timestamp` already holds rather than routed
    through `datetime`, which caps at microseconds. Session bounds land on whole
    minutes today, so nothing would be lost — but the exact path costs nothing
    and does not have to be revisited if that ever stops being true.
    """
    return int(timestamp.as_unit("ns").value)
