"""Tests for the corpus quality gate.

Driven entirely through fakes, since the gate is composition over four ports.
The cases that matter are the ones where a fault could pass as clean: a short
session looks complete to anything that only checks presence, and a survivorship
claim must not be stronger than the data supports.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date

import pytest

from neurotrade.core.calendar import TradingSession
from neurotrade.core.events import BarInterval
from neurotrade.core.quality import Coverage, Duplicate, Gap, SuspectSession
from neurotrade.core.types import Symbol, Venue
from neurotrade.core.universe import Universe
from neurotrade.ingest.quality import (
    CorpusReport,
    MissingSession,
    audit_corpus,
    summarise,
)

AAPL = Symbol("AAPL", Venue.NASDAQ)
MSFT = Symbol("MSFT", Venue.NASDAQ)
UNIVERSE = Universe([AAPL])

DAY_A = date(2026, 3, 12)
DAY_B = date(2026, 3, 13)
SPAN = (DAY_A, DAY_B)

# 2026-03-12 13:30 UTC = 09:30 ET open; close 20:00 UTC.
OPEN_NS = 1_773_322_200_000_000_000
CLOSE_NS = OPEN_NS + 390 * 60_000_000_000


class FakeCalendar:
    """A `CalendarPort` over a fixed pair of full sessions."""

    def __init__(self, days: Sequence[date] = (DAY_A, DAY_B)) -> None:
        self._days = tuple(days)

    def sessions(self, venue: Venue, start: date, end: date) -> tuple[date, ...]:
        return tuple(day for day in self._days if start <= day <= end)

    def session(self, venue: Venue, session_date: date) -> TradingSession | None:
        if session_date not in self._days:
            return None
        offset = (session_date - DAY_A).days * 86_400_000_000_000
        return TradingSession(
            venue=venue,
            session_date=session_date,
            open_ns=OPEN_NS + offset,
            close_ns=CLOSE_NS + offset,
            is_early_close=False,
        )


class FakeCatalog:
    """A `CatalogPort` over a bar-count table."""

    def __init__(self, counts: Mapping[Symbol, Mapping[date, int]]) -> None:
        self._counts = counts

    def bar_counts(self, symbol: Symbol, interval: BarInterval) -> Mapping[date, int]:
        return dict(self._counts.get(symbol, {}))


class FakeQuality:
    """A `CorpusQualityPort` answering from fixed lists."""

    def __init__(
        self,
        gaps: Sequence[Gap] = (),
        duplicates: Sequence[Duplicate] = (),
        suspect: Sequence[SuspectSession] = (),
    ) -> None:
        self._gaps = tuple(gaps)
        self._duplicates = tuple(duplicates)
        self._suspect = tuple(suspect)

    def coverage(
        self, symbol: Symbol | None = None, interval: BarInterval = BarInterval.MIN_1
    ) -> tuple[Coverage, ...]:
        return ()

    def gaps(
        self,
        symbol: Symbol,
        session_date: date,
        interval: BarInterval = BarInterval.MIN_1,
    ) -> tuple[Gap, ...]:
        return tuple(
            gap for gap in self._gaps if gap.symbol == symbol and gap.session_date == session_date
        )

    def duplicate_timestamps(
        self, interval: BarInterval = BarInterval.MIN_1
    ) -> tuple[Duplicate, ...]:
        return self._duplicates

    def suspect_sessions(
        self, interval: BarInterval = BarInterval.MIN_1
    ) -> tuple[SuspectSession, ...]:
        return self._suspect


def audit(
    counts: Mapping[Symbol, Mapping[date, int]],
    quality: FakeQuality | None = None,
    *,
    universe: Universe = UNIVERSE,
    survivors_only: bool = True,
) -> CorpusReport:
    """Run the gate with the fakes above."""
    return audit_corpus(
        universe,
        FakeCalendar(),
        FakeCatalog(counts),
        quality or FakeQuality(),
        start=SPAN[0],
        end=SPAN[1],
        survivors_only=survivors_only,
    )


# ── A full corpus ────────────────────────────────────────────────────────────


def test_a_complete_corpus_is_clean() -> None:
    report = audit({AAPL: {DAY_A: 390, DAY_B: 390}})
    assert report.clean
    assert report.describe().startswith("0 missing sessions")


def test_extended_hours_bars_do_not_count_as_a_fault() -> None:
    """`is_complete` is `>=` on purpose; more than 390 is legitimate."""
    assert audit({AAPL: {DAY_A: 900, DAY_B: 900}}).clean


# ── Missing and short ────────────────────────────────────────────────────────


def test_a_session_with_no_bars_is_missing() -> None:
    report = audit({AAPL: {DAY_A: 390}})
    assert report.missing_sessions == (MissingSession(AAPL, DAY_B),)
    assert not report.short_sessions


def test_a_session_with_too_few_bars_is_short_not_missing() -> None:
    """The dangerous one: it looks complete to anything checking presence."""
    report = audit({AAPL: {DAY_A: 390, DAY_B: 370}})
    assert not report.missing_sessions
    (short,) = report.short_sessions
    assert (short.session_date, short.held, short.expected, short.missing) == (DAY_B, 370, 390, 20)


def test_an_empty_corpus_reports_every_session_missing() -> None:
    report = audit({})
    assert len(report.missing_sessions) == 2
    assert not report.clean


# ── Gaps, duplicates, suspect ────────────────────────────────────────────────


def test_gaps_outside_the_range_are_dropped() -> None:
    inside = Gap(AAPL, DAY_A, BarInterval.MIN_1, OPEN_NS, OPEN_NS + 5 * 60_000_000_000)
    outside = Gap(AAPL, date(2020, 1, 2), BarInterval.MIN_1, 0, 60_000_000_000)
    report = audit({AAPL: {DAY_A: 390, DAY_B: 390}}, FakeQuality(gaps=[inside, outside]))
    assert report.gaps == (inside,)


def test_a_missing_session_is_not_also_scanned_for_gaps() -> None:
    """A hole needs a session to be inside; absence is already reported."""
    phantom = Gap(AAPL, DAY_B, BarInterval.MIN_1, OPEN_NS, OPEN_NS + 5 * 60_000_000_000)
    report = audit({AAPL: {DAY_A: 390}}, FakeQuality(gaps=[phantom]))
    assert report.gaps == ()
    assert report.missing_sessions == (MissingSession(AAPL, DAY_B),)


def test_a_duplicate_makes_the_report_dirty() -> None:
    dup = Duplicate(AAPL, DAY_A, BarInterval.MIN_1, OPEN_NS, 2)
    report = audit({AAPL: {DAY_A: 390, DAY_B: 390}}, FakeQuality(duplicates=[dup]))
    assert not report.clean
    assert report.duplicates[0].extra_rows == 1


def test_a_suspect_session_makes_the_report_dirty() -> None:
    flat = SuspectSession(AAPL, DAY_A, BarInterval.MIN_1, "zero volume", 390)
    report = audit({AAPL: {DAY_A: 390, DAY_B: 390}}, FakeQuality(suspect=[flat]))
    assert not report.clean
    assert "zero volume" in str(report.suspect[0])


# ── Survivorship ─────────────────────────────────────────────────────────────


def test_survivorship_is_not_measurable_from_a_survivor_only_source() -> None:
    """The audit must not imply a number it cannot justify."""
    report = audit({AAPL: {DAY_A: 390, DAY_B: 390}})
    assert report.survivorship is not None
    assert not report.survivorship.measurable
    assert "NOT measurable" in report.survivorship.describe()


def test_survivorship_becomes_measurable_when_the_source_lists_delistings() -> None:
    report = audit({AAPL: {DAY_A: 390, DAY_B: 390}}, survivors_only=False)
    assert report.survivorship is not None
    assert report.survivorship.measurable


def test_a_late_listing_is_recorded_as_the_visible_half() -> None:
    """History starting mid-range is the one thing a survivor-only feed shows."""
    report = audit({AAPL: {DAY_B: 390}})
    assert report.survivorship is not None
    assert report.survivorship.late_listings == ((AAPL, DAY_B),)


def test_survivorship_never_makes_the_gate_red() -> None:
    """A permanently red gate is an ignored one."""
    report = audit({AAPL: {DAY_A: 390, DAY_B: 390}})
    assert report.survivorship is not None
    assert not report.survivorship.measurable
    assert report.clean


def test_names_without_history_are_counted() -> None:
    report = audit({AAPL: {DAY_A: 390, DAY_B: 390}}, universe=Universe([AAPL, MSFT]))
    assert report.survivorship is not None
    assert (report.survivorship.candidates, report.survivorship.with_history) == (2, 1)


# ── Rejection and reporting ──────────────────────────────────────────────────


def test_an_inverted_range_is_refused() -> None:
    with pytest.raises(ValueError, match="is before start"):
        audit_corpus(
            UNIVERSE,
            FakeCalendar(),
            FakeCatalog({}),
            FakeQuality(),
            start=DAY_B,
            end=DAY_A,
        )


def test_summarise_truncates_and_says_how_many_it_withheld() -> None:
    """A crawl in progress finds thousands of missing sessions; duplicates
    must not be buried under them."""
    shown, withheld = summarise(list(range(100)), limit=3)
    assert (shown, withheld) == (["0", "1", "2"], 97)


def test_summarise_withholds_nothing_when_it_fits() -> None:
    assert summarise([1, 2], limit=10) == (["1", "2"], 0)


def test_daily_bars_skip_the_intra_session_gap_query() -> None:
    """Vacuous for a one-bar session, and one query per session is not free."""
    asked: list[date] = []

    class CountingQuality(FakeQuality):
        def gaps(
            self,
            symbol: Symbol,
            session_date: date,
            interval: BarInterval = BarInterval.MIN_1,
        ) -> tuple[Gap, ...]:
            asked.append(session_date)
            return ()

    audit_corpus(
        UNIVERSE,
        FakeCalendar(),
        FakeCatalog({AAPL: {DAY_A: 1, DAY_B: 1}}),
        CountingQuality(),
        start=SPAN[0],
        end=SPAN[1],
        interval=BarInterval.DAY_1,
    )
    assert asked == []
