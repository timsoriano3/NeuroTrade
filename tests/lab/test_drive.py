"""The shared drive loop, and the property that justifies sharing it.

`ReplayEngine` and `BacktestEngine` exist to be the same machine fed from two
places. The test that earns the refactor is the last one here: the same bars,
driven once from an event log and once from the corpus, must produce the same
digest. If they ever diverge, a backtest result stops saying anything about what
a replay of the live session would do — which is §3.6's failure mode exactly.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import date

import pytest

from neurotrade.bus import EventBus
from neurotrade.core.clock import SimClock
from neurotrade.core.events import Bar, BarInterval, Event
from neurotrade.core.types import Price, Quantity, Symbol, Venue
from neurotrade.lab.drive import RunDigest, RunResult, drive
from neurotrade.lab.engine import BacktestEngine
from neurotrade.lab.feed import CorpusFeed
from neurotrade.lab.replay import ReplayEngine

AAPL = Symbol("AAPL", Venue.NASDAQ)
MSFT = Symbol("MSFT", Venue.NASDAQ)
FOREVER = 1_000_000


def bar(symbol: Symbol, ts: int) -> Bar:
    return Bar(
        symbol=symbol,
        ts_event=ts,
        ts_init=ts,
        interval=BarInterval.MIN_1,
        open=Price("100"),
        high=Price("101"),
        low=Price("99"),
        close=Price("100.5"),
        volume=Quantity(1_000),
    )


BARS = [bar(AAPL, 10), bar(MSFT, 10), bar(AAPL, 20), bar(MSFT, 20)]


class ListLog:
    """An `EventStorePort` over a fixed list."""

    def __init__(self, events: Sequence[Event]) -> None:
        self._events = events

    def append(self, event: Event) -> None:
        raise AssertionError("the drive loop must never write")

    def stream(self, start: int, end: int) -> Iterator[Event]:
        return iter([e for e in self._events if start <= e.ts_event < end])


class ListStore:
    """A `StoragePort` over the same list, filtered per symbol."""

    def __init__(self, bars: Sequence[Bar]) -> None:
        self._bars = bars

    def write_bars(self, bars: Sequence[Bar], *, source: str, session_date: date) -> None:
        raise AssertionError("the drive loop must never write")

    def read_bars(
        self, symbol: Symbol, interval: BarInterval, start: int, end: int
    ) -> Iterator[Bar]:
        return iter([b for b in self._bars if b.symbol == symbol and start <= b.ts_event < end])


def run_drive(events: Sequence[Event], clock: SimClock | None = None) -> RunResult:
    bus, digest = EventBus(), RunDigest()
    bus.subscribe(Event, digest)
    return drive(events, clock=clock or SimClock(0), bus=bus, digest=digest)


# ── Counting and span ────────────────────────────────────────


def test_counts_what_it_read() -> None:
    assert run_drive(BARS).events_read == 4


def test_records_first_and_last_timestamps() -> None:
    result = run_drive(BARS)
    assert (result.first_ts, result.last_ts, result.span_ns) == (10, 20, 10)


def test_an_empty_source_is_empty_not_an_error() -> None:
    result = run_drive([])
    assert result.is_empty
    assert (result.first_ts, result.last_ts, result.span_ns) == (None, None, 0)


def test_a_single_event_spans_nothing() -> None:
    assert run_drive([bar(AAPL, 10)]).span_ns == 0


# ── The clock ────────────────────────────────────────────────


def test_the_clock_is_moved_before_each_publish() -> None:
    clock = SimClock(0)
    seen: list[int] = []
    bus, digest = EventBus(), RunDigest()
    bus.subscribe(Event, digest)
    bus.subscribe(Event, lambda _: seen.append(clock.now_ns()))
    drive(BARS, clock=clock, bus=bus, digest=digest)
    assert seen == [10, 10, 20, 20]


def test_the_clock_ends_on_the_last_event() -> None:
    clock = SimClock(0)
    run_drive(BARS, clock)
    assert clock.now_ns() == 20


@pytest.mark.parametrize(
    "out_of_order",
    [[bar(AAPL, 20), bar(AAPL, 10)], [bar(AAPL, 10), bar(AAPL, 30), bar(AAPL, 20)]],
    ids=["reversed", "late-regression"],
)
def test_an_out_of_order_source_fails(out_of_order: list[Bar]) -> None:
    """The mis-ordering has to surface here, not as a quietly wrong session."""
    with pytest.raises(ValueError, match="backwards"):
        run_drive(out_of_order)


# ── The digest ───────────────────────────────────────────────


def test_the_digest_is_seeded_so_an_empty_run_still_hashes() -> None:
    """Codec-version seeded: two empty runs agree, and are not the zero hash."""
    assert run_drive([]).digest == run_drive([]).digest
    assert len(run_drive([]).digest) == 32


def test_order_changes_the_digest() -> None:
    """A rolling hash, not a set — sequence is part of what is proven."""
    forwards = run_drive([bar(AAPL, 10), bar(MSFT, 20)]).digest
    swapped = run_drive([bar(MSFT, 10), bar(AAPL, 20)]).digest
    assert forwards != swapped


def test_dispatched_can_exceed_read_when_handlers_publish() -> None:
    bus, digest = EventBus(), RunDigest()
    bus.subscribe(Event, digest)

    def echo(event: Event) -> None:
        assert isinstance(event, Bar)
        if event.symbol == AAPL:
            bus.publish(bar(MSFT, event.ts_event))

    bus.subscribe(Bar, echo)
    result = drive([bar(AAPL, 10)], clock=SimClock(0), bus=bus, digest=digest)
    assert (result.events_read, result.events_dispatched) == (1, 2)


# ── The property the refactor exists for ─────────────────────


def test_replay_and_backtest_agree_on_the_same_bars() -> None:
    """One implementation, two sources (§3.6). Divergence here is the failure."""
    replayed = ReplayEngine(ListLog(BARS), SimClock(0)).run(0, FOREVER)
    backtested = BacktestEngine(CorpusFeed(ListStore(BARS), (AAPL, MSFT), BarInterval.MIN_1)).run(
        0, FOREVER
    )

    assert replayed.digest == backtested.digest
    assert replayed.events_read == backtested.run.events_read


def test_the_two_engines_report_the_same_record_type() -> None:
    """`ReplayResult` is `RunResult`; a backtest's `run` is comparable to it."""
    replayed = ReplayEngine(ListLog(BARS), SimClock(0)).run(0, FOREVER)
    backtested = BacktestEngine(CorpusFeed(ListStore(BARS), (AAPL, MSFT), BarInterval.MIN_1)).run(
        0, FOREVER
    )
    assert replayed == backtested.run
