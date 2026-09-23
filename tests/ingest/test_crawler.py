"""Tests for the backfill crawler's fetch loop (`crawl` and `order_windows`).

`asyncio_mode = "auto"` (pyproject) lets these be plain `async def` tests; no
`asyncio.run` or `@pytest.mark.asyncio` needed.

Heavier on rejection than on the happy path. `crawl` is the layer that talks
to a feed the system does not control, so what matters most is that it never
files a bar under the wrong instrument, never writes what fell outside the
session, and gives up on a symbol — but not the whole pass — the moment that
symbol looks broken. None of the fakes below import or subclass anything from
`core.ports`; conformance is structural, same as in `test_backfill.py`.

Many tests pass `window_days=1` on purpose: their subject is per-session
behaviour — the skip after a failure, the breaker, the limit — and the default
30-day window would fold MON, TUE and WED into a single request, testing the
grouping instead of the thing named. The windowing itself has its own section.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator, Sequence
from datetime import UTC, date, datetime

import pytest

from neurotrade.core.calendar import TradingSession
from neurotrade.core.clock import Nanos, to_nanos
from neurotrade.core.events import Bar, BarInterval
from neurotrade.core.types import Price, Quantity, Symbol, Venue
from neurotrade.core.universe import Universe
from neurotrade.ingest.backfill import BackfillCell, BackfillWindow
from neurotrade.ingest.crawler import (
    CellOutcome,
    CellStatus,
    CrawlReport,
    crawl,
    order_windows,
)

AAPL = Symbol("AAPL", Venue.NASDAQ)
MSFT = Symbol("MSFT", Venue.NASDAQ)
SYMA = Symbol("SYMA", Venue.NASDAQ)
SYMB = Symbol("SYMB", Venue.NASDAQ)
SYMC = Symbol("SYMC", Venue.NASDAQ)
SYMD = Symbol("SYMD", Venue.NASDAQ)

MON = date(2024, 3, 4)
TUE = date(2024, 3, 5)
WED = date(2024, 3, 6)

SOURCE = "ibkr"


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


def _bar(symbol: Symbol, ts_event: Nanos, *, interval: BarInterval = BarInterval.MIN_1) -> Bar:
    """A minimal, domain-valid 1-minute bar closing at `ts_event`."""
    return Bar(
        symbol=symbol,
        ts_event=ts_event,
        ts_init=ts_event,
        interval=interval,
        open=Price("100"),
        high=Price("101"),
        low=Price("99"),
        close=Price("100.5"),
        volume=Quantity(10),
    )


class FakeCalendar:
    """An in-memory `CalendarPort`, keyed by venue then date — no exchange_calendars."""

    def __init__(self, sessions: dict[Venue, dict[date, TradingSession]]) -> None:
        self._sessions = sessions

    def sessions(self, venue: Venue, start: date, end: date) -> tuple[date, ...]:
        by_date = self._sessions.get(venue, {})
        return tuple(sorted(d for d in by_date if start <= d <= end))

    def session(self, venue: Venue, session_date: date) -> TradingSession | None:
        return self._sessions.get(venue, {}).get(session_date)


class FakeCatalog:
    """An in-memory `CatalogPort`."""

    def __init__(self, counts: dict[Symbol, dict[date, int]] | None = None) -> None:
        self._counts = counts or {}

    def bar_counts(self, symbol: Symbol, interval: BarInterval) -> dict[date, int]:
        return self._counts.get(symbol, {})


class ScriptedFeed:
    """A `MarketDataPort` answering from a table, raising for scripted symbols.

    Records every call so tests can assert what was (and was not) fetched.
    """

    def __init__(
        self,
        answers: dict[Symbol, Sequence[Bar]] | None = None,
        *,
        raises: dict[Symbol, BaseException] | None = None,
    ) -> None:
        self._answers = answers or {}
        self._raises = raises or {}
        self.calls: list[tuple[Symbol, BarInterval, Nanos, Nanos]] = []

    async def fetch_bars(
        self, symbol: Symbol, interval: BarInterval, start: Nanos, end: Nanos
    ) -> Sequence[Bar]:
        self.calls.append((symbol, interval, start, end))
        if symbol in self._raises:
            raise self._raises[symbol]
        return self._answers.get(symbol, ())

    async def is_connected(self) -> bool:
        return True


class RecordingStore:
    """A `StoragePort` that records writes, or raises instead when told to."""

    def __init__(self, *, raises: Exception | None = None) -> None:
        self._raises = raises
        self.writes: list[tuple[tuple[Bar, ...], str, date]] = []

    def write_bars(self, bars: Sequence[Bar], *, source: str, session_date: date) -> None:
        if self._raises is not None:
            raise self._raises
        self.writes.append((tuple(bars), source, session_date))

    def read_bars(
        self, symbol: Symbol, interval: BarInterval, start: Nanos, end: Nanos
    ) -> Iterator[Bar]:
        return iter(())


# ── order_windows ──────────────────────────────────────────────


def _window(symbol: Symbol, *days_and_held: tuple[date, int]) -> BackfillWindow:
    """A window over `symbol`, one cell per (session date, bars held) pair."""
    cells = tuple(
        BackfillCell(
            symbol=symbol,
            session=_session(Venue.NASDAQ, day),
            interval=BarInterval.MIN_1,
            held_bars=held,
        )
        for day, held in days_and_held
    )
    return BackfillWindow(symbol=symbol, interval=BarInterval.MIN_1, cells=cells)


def test_windows_with_untouched_sessions_precede_windows_of_only_short_ones() -> None:
    """The sort is stable: within "has untouched" and within "all short" the
    original (recency-major) order from `plan_windows` must survive."""
    untouched_wed = _window(AAPL, (WED, 0))
    short_tue = _window(MSFT, (TUE, 1))
    untouched_mon = _window(MSFT, (MON, 0))
    short_wed = _window(AAPL, (WED, 2))

    ordered = order_windows([untouched_wed, short_tue, untouched_mon, short_wed])

    assert ordered == [untouched_wed, untouched_mon, short_tue, short_wed]


def test_one_untouched_session_is_enough_to_promote_a_mixed_window() -> None:
    """A window is one request. If any session in it has never been fetched,
    that request buys new ground and belongs with the untouched group."""
    mixed = _window(AAPL, (MON, 0), (TUE, 200))
    all_short = _window(MSFT, (MON, 1), (TUE, 200))

    assert order_windows([all_short, mixed]) == [mixed, all_short]


def test_order_windows_on_an_empty_iterable_is_empty() -> None:
    assert order_windows([]) == []


# ── crawl: fetch window ──────────────────────────────────────────


async def test_the_feed_is_asked_for_the_session_shifted_one_nanosecond() -> None:
    """The port's window is inclusive-start/exclusive-end on `ts_event`; a
    session's bars close in `(open, close]`. Shifting both bounds by one
    nanosecond is what turns that into exactly this session's bars."""
    session = _session(Venue.NASDAQ, MON)
    calendar = FakeCalendar({Venue.NASDAQ: {MON: session}})
    catalog = FakeCatalog()
    feed = ScriptedFeed()
    store = RecordingStore()

    await crawl(Universe([AAPL]), calendar, catalog, feed, store, start=MON, end=MON, source=SOURCE)

    assert feed.calls == [(AAPL, BarInterval.MIN_1, session.open_ns + 1, session.close_ns + 1)]


async def test_a_bar_closing_at_the_session_open_is_trimmed_not_written() -> None:
    """A bar stamped at the open closed *before* the bell rang and belongs to
    the previous session; a sloppy feed that ignores the shifted window must
    not get it into the corpus."""
    session = _session(Venue.NASDAQ, MON)
    calendar = FakeCalendar({Venue.NASDAQ: {MON: session}})
    catalog = FakeCatalog()
    at_open = _bar(AAPL, session.open_ns)
    just_after = _bar(AAPL, session.open_ns + BarInterval.MIN_1.nanos)
    feed = ScriptedFeed({AAPL: (at_open, just_after)})
    store = RecordingStore()

    report = await crawl(
        Universe([AAPL]), calendar, catalog, feed, store, start=MON, end=MON, source=SOURCE
    )

    assert report.outcomes[0].status is CellStatus.FILLED
    assert report.outcomes[0].written == 1
    ((written, _, _),) = store.writes
    assert written == (just_after,)


async def test_a_bar_closing_exactly_at_session_close_is_written() -> None:
    """Close-inclusive: the final bar of a session closes at the bell itself."""
    session = _session(Venue.NASDAQ, MON)
    calendar = FakeCalendar({Venue.NASDAQ: {MON: session}})
    catalog = FakeCatalog()
    at_close = _bar(AAPL, session.close_ns)
    feed = ScriptedFeed({AAPL: (at_close,)})
    store = RecordingStore()

    report = await crawl(
        Universe([AAPL]), calendar, catalog, feed, store, start=MON, end=MON, source=SOURCE
    )

    assert report.outcomes[0].status is CellStatus.FILLED
    ((written, _, _),) = store.writes
    assert written == (at_close,)


# ── crawl: trimming and emptiness ─────────────────────────────────


async def test_bars_outside_the_session_are_trimmed_not_written() -> None:
    session = _session(Venue.NASDAQ, MON)
    calendar = FakeCalendar({Venue.NASDAQ: {MON: session}})
    catalog = FakeCatalog()
    in_session = _bar(AAPL, session.open_ns + BarInterval.MIN_1.nanos)
    way_after = _bar(AAPL, session.close_ns + 10 * BarInterval.MIN_1.nanos)
    feed = ScriptedFeed({AAPL: (in_session, way_after)})
    store = RecordingStore()

    report = await crawl(
        Universe([AAPL]), calendar, catalog, feed, store, start=MON, end=MON, source=SOURCE
    )

    assert report.outcomes[0].written == 1
    ((written, _, _),) = store.writes
    assert written == (in_session,)


async def test_all_bars_outside_the_session_yields_empty_and_no_write() -> None:
    session = _session(Venue.NASDAQ, MON)
    calendar = FakeCalendar({Venue.NASDAQ: {MON: session}})
    catalog = FakeCatalog()
    way_after = _bar(AAPL, session.close_ns + 10 * BarInterval.MIN_1.nanos)
    feed = ScriptedFeed({AAPL: (way_after,)})
    store = RecordingStore()

    report = await crawl(
        Universe([AAPL]), calendar, catalog, feed, store, start=MON, end=MON, source=SOURCE
    )

    assert report.outcomes[0].status is CellStatus.EMPTY
    assert report.outcomes[0].written == 0
    assert store.writes == []


async def test_an_empty_feed_answer_yields_empty_and_no_write() -> None:
    session = _session(Venue.NASDAQ, MON)
    calendar = FakeCalendar({Venue.NASDAQ: {MON: session}})
    catalog = FakeCatalog()
    feed = ScriptedFeed({AAPL: ()})
    store = RecordingStore()

    report = await crawl(
        Universe([AAPL]), calendar, catalog, feed, store, start=MON, end=MON, source=SOURCE
    )

    assert report.outcomes[0].status is CellStatus.EMPTY
    assert store.writes == []


# ── crawl: filled outcomes ─────────────────────────────────────────


async def test_filled_records_source_session_date_and_bar_count() -> None:
    session = _session(Venue.NASDAQ, MON)
    calendar = FakeCalendar({Venue.NASDAQ: {MON: session}})
    catalog = FakeCatalog()
    bars = tuple(_bar(AAPL, session.open_ns + n * BarInterval.MIN_1.nanos) for n in (1, 2, 3))
    feed = ScriptedFeed({AAPL: bars})
    store = RecordingStore()

    report = await crawl(
        Universe([AAPL]), calendar, catalog, feed, store, start=MON, end=MON, source="polygon"
    )

    outcome = report.outcomes[0]
    assert outcome.status is CellStatus.FILLED
    assert outcome.written == 3
    assert outcome.session_date == MON
    ((written, source, session_date),) = store.writes
    assert written == bars
    assert source == "polygon"
    assert session_date == MON


# ── crawl: symbol/interval mismatch ────────────────────────────────


async def test_a_bar_for_another_symbol_raises_and_writes_nothing() -> None:
    session = _session(Venue.NASDAQ, MON)
    calendar = FakeCalendar({Venue.NASDAQ: {MON: session}})
    catalog = FakeCatalog()
    wrong_symbol = _bar(MSFT, session.open_ns + BarInterval.MIN_1.nanos)
    feed = ScriptedFeed({AAPL: (wrong_symbol,)})
    store = RecordingStore()

    with pytest.raises(ValueError, match=r"feed answered a request for"):
        await crawl(
            Universe([AAPL]), calendar, catalog, feed, store, start=MON, end=MON, source=SOURCE
        )

    assert store.writes == []


async def test_a_bar_for_another_interval_raises_and_writes_nothing() -> None:
    session = _session(Venue.NASDAQ, MON)
    calendar = FakeCalendar({Venue.NASDAQ: {MON: session}})
    catalog = FakeCatalog()
    wrong_interval = _bar(
        AAPL, session.open_ns + BarInterval.MIN_5.nanos, interval=BarInterval.MIN_5
    )
    feed = ScriptedFeed({AAPL: (wrong_interval,)})
    store = RecordingStore()

    with pytest.raises(ValueError, match=r"feed answered a request for"):
        await crawl(
            Universe([AAPL]), calendar, catalog, feed, store, start=MON, end=MON, source=SOURCE
        )

    assert store.writes == []


# ── crawl: failures, skips, other symbols ──────────────────────────


async def test_a_failed_symbol_is_skipped_for_the_rest_of_the_pass_others_still_fetched() -> None:
    """AAPL is missing Tuesday and Monday; MSFT is missing only Monday. AAPL's
    first request fails, so its Monday cell must be skipped without a fetch —
    but MSFT, an unrelated symbol, must still be tried."""
    session_tue = _session(Venue.NASDAQ, TUE)
    session_mon = _session(Venue.NASDAQ, MON)
    calendar = FakeCalendar({Venue.NASDAQ: {TUE: session_tue, MON: session_mon}})
    catalog = FakeCatalog({MSFT: {TUE: 390}})  # MSFT's Tuesday is already complete
    msft_bar = _bar(MSFT, session_mon.open_ns + BarInterval.MIN_1.nanos)
    feed = ScriptedFeed({MSFT: (msft_bar,)}, raises={AAPL: RuntimeError("boom")})
    store = RecordingStore()

    report = await crawl(
        Universe([AAPL, MSFT]),
        calendar,
        catalog,
        feed,
        store,
        start=MON,
        end=TUE,
        source=SOURCE,
        window_days=1,
    )

    assert [o.status for o in report.outcomes] == [
        CellStatus.FAILED,
        CellStatus.SKIPPED,
        CellStatus.FILLED,
    ]
    assert "RuntimeError" in (report.outcomes[0].error or "")
    assert [call[0] for call in feed.calls] == [AAPL, MSFT]  # AAPL@MON never fetched
    assert report.completed


async def test_skipped_cell_carries_no_error_and_writes_nothing() -> None:
    session_tue = _session(Venue.NASDAQ, TUE)
    session_mon = _session(Venue.NASDAQ, MON)
    calendar = FakeCalendar({Venue.NASDAQ: {TUE: session_tue, MON: session_mon}})
    catalog = FakeCatalog()
    feed = ScriptedFeed(raises={AAPL: RuntimeError("boom")})
    store = RecordingStore()

    report = await crawl(
        Universe([AAPL]),
        calendar,
        catalog,
        feed,
        store,
        start=MON,
        end=TUE,
        source=SOURCE,
        window_days=1,
    )

    skipped = report.outcomes[1]
    assert skipped.status is CellStatus.SKIPPED
    assert skipped.error is None
    assert skipped.written == 0
    assert store.writes == []


# ── crawl: consecutive-failure breaker ─────────────────────────────


async def test_the_breaker_stops_the_pass_after_n_distinct_symbols_fail_in_a_row() -> None:
    session = _session(Venue.NASDAQ, MON)
    calendar = FakeCalendar({Venue.NASDAQ: {MON: session}})
    catalog = FakeCatalog()
    feed = ScriptedFeed(raises={SYMA: RuntimeError("a"), SYMB: RuntimeError("b")})
    store = RecordingStore()

    report = await crawl(
        Universe([SYMA, SYMB, SYMC]),
        calendar,
        catalog,
        feed,
        store,
        start=MON,
        end=MON,
        source=SOURCE,
        max_consecutive_failures=2,
    )

    assert [o.status for o in report.outcomes] == [CellStatus.FAILED, CellStatus.FAILED]
    assert not report.completed
    assert report.stopped is not None
    assert "2 consecutive" in report.stopped
    assert [call[0] for call in feed.calls] == [SYMA, SYMB]  # SYMC never even tried


async def test_a_success_between_failures_resets_the_consecutive_count() -> None:
    """SYMA fails, SYMB succeeds, then SYMC and SYMD fail — only the second
    pair should count towards a limit of two, so the pass must reach the end
    rather than stop after SYMB's neighbour."""
    session = _session(Venue.NASDAQ, MON)
    calendar = FakeCalendar({Venue.NASDAQ: {MON: session}})
    catalog = FakeCatalog()
    symb_bar = _bar(SYMB, session.open_ns + BarInterval.MIN_1.nanos)
    feed = ScriptedFeed(
        {SYMB: (symb_bar,)},
        raises={SYMA: RuntimeError("a"), SYMC: RuntimeError("c"), SYMD: RuntimeError("d")},
    )
    store = RecordingStore()

    report = await crawl(
        Universe([SYMA, SYMB, SYMC, SYMD]),
        calendar,
        catalog,
        feed,
        store,
        start=MON,
        end=MON,
        source=SOURCE,
        max_consecutive_failures=2,
    )

    assert [o.status for o in report.outcomes] == [
        CellStatus.FAILED,
        CellStatus.FILLED,
        CellStatus.FAILED,
        CellStatus.FAILED,
    ]
    assert not report.completed  # SYMC, SYMD is the triggering pair


async def test_skips_neither_increment_nor_reset_the_consecutive_count() -> None:
    """SYMA fails at TUE, is skipped at MON (must not move the counter either
    way), then SYMB fails at MON. With a limit of two, the pair that actually
    trips the breaker is SYMA@TUE and SYMB@MON — so SYMC, offered afterwards,
    must never be fetched."""
    session_tue = _session(Venue.NASDAQ, TUE)
    session_mon = _session(Venue.NASDAQ, MON)
    calendar = FakeCalendar(
        {
            Venue.NASDAQ: {
                TUE: session_tue,
                MON: session_mon,
            }
        }
    )
    # Only SYMA has a Tuesday gap, so the plan is [SYMA@TUE, SYMA@MON, SYMB@MON, SYMC@MON].
    catalog = FakeCatalog({SYMB: {TUE: 390}, SYMC: {TUE: 390}})
    feed = ScriptedFeed(raises={SYMA: RuntimeError("a"), SYMB: RuntimeError("b")})
    store = RecordingStore()

    report = await crawl(
        Universe([SYMA, SYMB, SYMC]),
        calendar,
        catalog,
        feed,
        store,
        start=MON,
        end=TUE,
        source=SOURCE,
        window_days=1,
        max_consecutive_failures=2,
    )

    assert [o.status for o in report.outcomes] == [
        CellStatus.FAILED,
        CellStatus.SKIPPED,
        CellStatus.FAILED,
    ]
    assert not report.completed
    assert [call[0] for call in feed.calls] == [SYMA, SYMB]  # SYMC never fetched


# ── crawl: store and cancellation ──────────────────────────────────


async def test_a_store_error_propagates() -> None:
    """A write failure is a local fault — a full disk, a permission — every
    later cell would hit too, so it is not caught."""
    session = _session(Venue.NASDAQ, MON)
    calendar = FakeCalendar({Venue.NASDAQ: {MON: session}})
    catalog = FakeCatalog()
    bar = _bar(AAPL, session.open_ns + BarInterval.MIN_1.nanos)
    feed = ScriptedFeed({AAPL: (bar,)})
    store = RecordingStore(raises=OSError("disk full"))

    with pytest.raises(OSError, match=r"disk full"):
        await crawl(
            Universe([AAPL]), calendar, catalog, feed, store, start=MON, end=MON, source=SOURCE
        )


async def test_cancellation_propagates_rather_than_becoming_a_failed_outcome() -> None:
    """`CancelledError` is a `BaseException`, not caught by the `except
    Exception` around the feed call — an interrupted crawl must stop, not
    quietly record a failure and keep going."""
    session = _session(Venue.NASDAQ, MON)
    calendar = FakeCalendar({Venue.NASDAQ: {MON: session}})
    catalog = FakeCatalog()
    feed = ScriptedFeed(raises={AAPL: asyncio.CancelledError()})
    store = RecordingStore()

    with pytest.raises(asyncio.CancelledError):
        await crawl(
            Universe([AAPL]), calendar, catalog, feed, store, start=MON, end=MON, source=SOURCE
        )


# ── crawl: ordering through the whole pass ──────────────────────────


async def test_untouched_cells_are_fetched_before_short_ones_across_the_pass() -> None:
    """SYMA's Monday session is untouched; SYMA's Wednesday session merely has
    one bar missing. Both are offered before any date-major concern, so the
    untouched Monday cell must be fetched first."""
    session_mon = _session(Venue.NASDAQ, MON)
    session_wed = _session(Venue.NASDAQ, WED)
    calendar = FakeCalendar({Venue.NASDAQ: {MON: session_mon, WED: session_wed}})
    catalog = FakeCatalog({SYMA: {WED: 389}})  # Wednesday: one bar short, so "short" not untouched
    feed = ScriptedFeed()
    store = RecordingStore()

    await crawl(
        Universe([SYMA]),
        calendar,
        catalog,
        feed,
        store,
        start=MON,
        end=WED,
        source=SOURCE,
        window_days=1,
    )

    assert [call[3] for call in feed.calls] == [session_mon.close_ns + 1, session_wed.close_ns + 1]


# ── crawl: limit and validation ──────────────────────────────────────


async def test_limit_truncates_requests_offered_but_planned_reports_the_full_count() -> None:
    calendar = FakeCalendar(
        {Venue.NASDAQ: {day: _session(Venue.NASDAQ, day) for day in (MON, TUE, WED)}}
    )
    catalog = FakeCatalog()
    feed = ScriptedFeed()
    store = RecordingStore()

    report = await crawl(
        Universe([AAPL]),
        calendar,
        catalog,
        feed,
        store,
        start=MON,
        end=WED,
        source=SOURCE,
        window_days=1,
        limit=1,
    )

    assert report.planned == 3
    assert report.planned_windows == 3
    assert len(report.outcomes) == 1
    assert len(feed.calls) == 1


async def test_limit_counts_requests_not_sessions() -> None:
    """One window holding three sessions is one request, so a limit of one
    lets all three through. Limiting sessions instead would make `--limit`
    mean something different depending on how the dates happened to group."""
    calendar = FakeCalendar(
        {Venue.NASDAQ: {day: _session(Venue.NASDAQ, day) for day in (MON, TUE, WED)}}
    )
    catalog = FakeCatalog()
    feed = ScriptedFeed()
    store = RecordingStore()

    report = await crawl(
        Universe([AAPL]),
        calendar,
        catalog,
        feed,
        store,
        start=MON,
        end=WED,
        source=SOURCE,
        limit=1,
    )

    assert report.planned == 3
    assert report.planned_windows == 1
    assert len(report.outcomes) == 3
    assert len(feed.calls) == 1


@pytest.mark.parametrize("limit", [0, -1])
async def test_a_non_positive_limit_raises_value_error(limit: int) -> None:
    calendar = FakeCalendar({})
    catalog = FakeCatalog()
    feed = ScriptedFeed()
    store = RecordingStore()

    with pytest.raises(ValueError, match=r"limit must be at least 1"):
        await crawl(
            Universe([AAPL]),
            calendar,
            catalog,
            feed,
            store,
            start=MON,
            end=MON,
            source=SOURCE,
            limit=limit,
        )


@pytest.mark.parametrize("max_consecutive_failures", [0, -1])
async def test_a_non_positive_max_consecutive_failures_raises_value_error(
    max_consecutive_failures: int,
) -> None:
    calendar = FakeCalendar({})
    catalog = FakeCatalog()
    feed = ScriptedFeed()
    store = RecordingStore()

    with pytest.raises(ValueError, match=r"max_consecutive_failures must be at least 1"):
        await crawl(
            Universe([AAPL]),
            calendar,
            catalog,
            feed,
            store,
            start=MON,
            end=MON,
            source=SOURCE,
            max_consecutive_failures=max_consecutive_failures,
        )


# ── crawl: a complete session is never fetched ────────────────────────


async def test_a_complete_session_in_the_catalog_is_never_fetched() -> None:
    session = _session(Venue.NASDAQ, MON)
    calendar = FakeCalendar({Venue.NASDAQ: {MON: session}})
    catalog = FakeCatalog({AAPL: {MON: 390}})
    feed = ScriptedFeed()
    store = RecordingStore()

    report = await crawl(
        Universe([AAPL]), calendar, catalog, feed, store, start=MON, end=MON, source=SOURCE
    )

    assert report.planned == 0
    assert report.outcomes == ()
    assert feed.calls == []


# ── crawl: on_outcome callback ─────────────────────────────────────────


async def test_on_outcome_is_called_once_per_cell_in_order() -> None:
    calendar = FakeCalendar(
        {Venue.NASDAQ: {day: _session(Venue.NASDAQ, day) for day in (MON, TUE)}}
    )
    catalog = FakeCatalog()
    feed = ScriptedFeed()
    store = RecordingStore()
    seen: list[CellOutcome] = []

    report = await crawl(
        Universe([AAPL]),
        calendar,
        catalog,
        feed,
        store,
        start=MON,
        end=TUE,
        source=SOURCE,
        on_outcome=seen.append,
    )

    assert seen == list(report.outcomes)
    assert len(seen) == 2


# ── crawl: windowing ────────────────────────────────────────────────


async def test_consecutive_sessions_are_fetched_as_one_request_spanning_them() -> None:
    """The whole point of the change: three missing sessions cost one paced
    request, bounded by the oldest open and the newest close."""
    session_mon = _session(Venue.NASDAQ, MON)
    session_wed = _session(Venue.NASDAQ, WED)
    calendar = FakeCalendar(
        {Venue.NASDAQ: {day: _session(Venue.NASDAQ, day) for day in (MON, TUE, WED)}}
    )
    catalog = FakeCatalog()
    feed = ScriptedFeed()
    store = RecordingStore()

    report = await crawl(
        Universe([AAPL]), calendar, catalog, feed, store, start=MON, end=WED, source=SOURCE
    )

    assert feed.calls == [
        (AAPL, BarInterval.MIN_1, session_mon.open_ns + 1, session_wed.close_ns + 1)
    ]
    assert report.requests == 1
    assert report.planned == 3  # still measured in instrument-sessions


async def test_a_windows_bars_are_split_and_written_under_each_session_date() -> None:
    """One request, three writes. A bar filed under the wrong session date is
    invisible to the plan, which counts by date, so the split is the part that
    has to be right."""
    sessions = {day: _session(Venue.NASDAQ, day) for day in (MON, TUE, WED)}
    calendar = FakeCalendar({Venue.NASDAQ: sessions})
    catalog = FakeCatalog()
    bars = tuple(
        _bar(AAPL, sessions[day].open_ns + n * BarInterval.MIN_1.nanos)
        for day in (MON, TUE, WED)
        for n in (1, 2)
    )
    feed = ScriptedFeed({AAPL: bars})
    store = RecordingStore()

    report = await crawl(
        Universe([AAPL]), calendar, catalog, feed, store, start=MON, end=WED, source=SOURCE
    )

    assert [(o.session_date, o.status, o.written) for o in report.outcomes] == [
        (MON, CellStatus.FILLED, 2),
        (TUE, CellStatus.FILLED, 2),
        (WED, CellStatus.FILLED, 2),
    ]
    assert store.writes == [
        (bars[0:2], SOURCE, MON),
        (bars[2:4], SOURCE, TUE),
        (bars[4:6], SOURCE, WED),
    ]


async def test_bars_arriving_out_of_order_still_land_in_the_right_sessions() -> None:
    """The split walks bars and sessions in step, so an unsorted answer would
    be walked past and silently dropped. It is sorted first for that reason."""
    sessions = {day: _session(Venue.NASDAQ, day) for day in (MON, TUE)}
    calendar = FakeCalendar({Venue.NASDAQ: sessions})
    catalog = FakeCatalog()
    monday = _bar(AAPL, sessions[MON].open_ns + BarInterval.MIN_1.nanos)
    tuesday = _bar(AAPL, sessions[TUE].open_ns + BarInterval.MIN_1.nanos)
    feed = ScriptedFeed({AAPL: (tuesday, monday)})  # newest first
    store = RecordingStore()

    report = await crawl(
        Universe([AAPL]), calendar, catalog, feed, store, start=MON, end=TUE, source=SOURCE
    )

    assert [(o.session_date, o.written) for o in report.outcomes] == [(MON, 1), (TUE, 1)]
    assert store.writes == [((monday,), SOURCE, MON), ((tuesday,), SOURCE, TUE)]


async def test_a_session_the_feed_had_nothing_for_is_empty_while_its_neighbours_fill() -> None:
    """A halt in the middle of a window must not cost the sessions either side
    of it — and must still read as empty rather than as a short fill."""
    sessions = {day: _session(Venue.NASDAQ, day) for day in (MON, TUE, WED)}
    calendar = FakeCalendar({Venue.NASDAQ: sessions})
    catalog = FakeCatalog()
    monday = _bar(AAPL, sessions[MON].open_ns + BarInterval.MIN_1.nanos)
    wednesday = _bar(AAPL, sessions[WED].open_ns + BarInterval.MIN_1.nanos)
    feed = ScriptedFeed({AAPL: (monday, wednesday)})
    store = RecordingStore()

    report = await crawl(
        Universe([AAPL]), calendar, catalog, feed, store, start=MON, end=WED, source=SOURCE
    )

    assert [(o.session_date, o.status) for o in report.outcomes] == [
        (MON, CellStatus.FILLED),
        (TUE, CellStatus.EMPTY),
        (WED, CellStatus.FILLED),
    ]
    assert [session_date for _, _, session_date in store.writes] == [MON, WED]


async def test_a_failed_request_fails_every_session_in_its_window_but_counts_once() -> None:
    """A window is one request, so it is one failure. Counting its sessions
    would trip a breaker of five on the first unresolvable listing, since a
    month's window holds about twenty-two of them."""
    sessions = {day: _session(Venue.NASDAQ, day) for day in (MON, TUE, WED)}
    calendar = FakeCalendar({Venue.NASDAQ: sessions})
    catalog = FakeCatalog()
    feed = ScriptedFeed(raises={AAPL: RuntimeError("no such contract")})
    store = RecordingStore()

    report = await crawl(
        Universe([AAPL]),
        calendar,
        catalog,
        feed,
        store,
        start=MON,
        end=WED,
        source=SOURCE,
        max_consecutive_failures=2,
    )

    assert [o.status for o in report.outcomes] == [CellStatus.FAILED] * 3
    assert all("no such contract" in (o.error or "") for o in report.outcomes)
    assert report.completed  # one failure, not three


async def test_a_gap_wider_than_the_window_is_two_requests() -> None:
    """Windows are capped in calendar days, not in sessions."""
    far = date(2024, 5, 6)
    sessions = {day: _session(Venue.NASDAQ, day) for day in (MON, far)}
    calendar = FakeCalendar({Venue.NASDAQ: sessions})
    catalog = FakeCatalog()
    feed = ScriptedFeed()
    store = RecordingStore()

    report = await crawl(
        Universe([AAPL]), calendar, catalog, feed, store, start=MON, end=far, source=SOURCE
    )

    assert report.requests == 2
    assert [call[0] for call in feed.calls] == [AAPL, AAPL]
    # Newest span first, as with cells.
    assert [o.session_date for o in report.outcomes] == [far, MON]


@pytest.mark.parametrize("window_days", [0, -1])
async def test_a_non_positive_window_days_raises_value_error(window_days: int) -> None:
    calendar = FakeCalendar({Venue.NASDAQ: {MON: _session(Venue.NASDAQ, MON)}})
    catalog = FakeCatalog()
    feed = ScriptedFeed()
    store = RecordingStore()

    with pytest.raises(ValueError, match=r"window_days must be at least 1"):
        await crawl(
            Universe([AAPL]),
            calendar,
            catalog,
            feed,
            store,
            start=MON,
            end=MON,
            source=SOURCE,
            window_days=window_days,
        )


# ── CrawlReport: derived counts ─────────────────────────────────────────


def _outcome(status: CellStatus, *, written: int = 0) -> CellOutcome:
    return CellOutcome(symbol=AAPL, session_date=MON, status=status, written=written)


def test_requests_and_planned_windows_are_recorded_not_derived() -> None:
    """Since windowing, requests cannot be counted from the outcomes: four
    sessions may be one request. The pass records what it actually sent."""
    report = CrawlReport(
        outcomes=(
            _outcome(CellStatus.FILLED, written=5),
            _outcome(CellStatus.FILLED, written=2),
            _outcome(CellStatus.EMPTY),
            _outcome(CellStatus.EMPTY),
        ),
        planned=4,
        planned_windows=1,
        requests=1,
    )
    assert (report.requests, report.planned_windows) == (1, 1)


def test_bars_written_sums_across_outcomes() -> None:
    report = CrawlReport(
        outcomes=(
            _outcome(CellStatus.FILLED, written=5),
            _outcome(CellStatus.FILLED, written=2),
            _outcome(CellStatus.EMPTY),
        ),
        planned=3,
    )
    assert report.bars_written == 7


def test_count_returns_how_many_outcomes_match_a_status() -> None:
    report = CrawlReport(
        outcomes=(
            _outcome(CellStatus.FAILED),
            _outcome(CellStatus.FAILED),
            _outcome(CellStatus.FILLED, written=1),
        ),
        planned=3,
    )
    assert report.count(CellStatus.FAILED) == 2
    assert report.count(CellStatus.SKIPPED) == 0


def test_completed_is_true_only_when_stopped_is_none() -> None:
    assert CrawlReport(outcomes=(), planned=0).completed
    assert not CrawlReport(outcomes=(), planned=0, stopped="gave up").completed
