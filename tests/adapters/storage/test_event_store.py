"""Tests for the append-only event log.

The properties under test are the ones a replay depends on: everything written
comes back, in order, unchanged, and a damaged log is refused rather than
quietly truncated.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from neurotrade.adapters.storage.event_store import CorruptEventLog, EventStore
from neurotrade.core.events import (
    Bar,
    BarInterval,
    Event,
    MarketSession,
    SessionBoundary,
    TradingHalt,
)
from neurotrade.core.intent import EntryTrigger, Intent
from neurotrade.core.ports import EventStorePort
from neurotrade.core.types import Price, Quantity, Side, Symbol, Venue

AAPL = Symbol("AAPL", Venue.NASDAQ)
MINUTE = 60_000_000_000
OPEN_NS = 1_773_495_000_000_000_000


def bar(index: int = 0, *, seq: int = 0) -> Bar:
    ts = OPEN_NS + index * MINUTE
    return Bar(
        symbol=AAPL,
        ts_event=ts,
        ts_init=ts,
        seq=seq,
        interval=BarInterval.MIN_1,
        open=Price("100.12345678"),
        high=Price("101"),
        low=Price("99"),
        close=Price("100.5"),
        volume=Quantity(1_000),
    )


@pytest.fixture
def log(tmp_path: Path) -> Path:
    return tmp_path / "session.jsonl"


def read_all(path: Path) -> list[Event]:
    return list(EventStore(path).stream(0, OPEN_NS + 10_000 * MINUTE))


# ── Conformance ──────────────────────────────────────────────


def test_satisfies_the_event_store_port(log: Path) -> None:
    assert isinstance(EventStore(log), EventStorePort)


# ── Round trip ───────────────────────────────────────────────


def test_everything_written_comes_back(log: Path) -> None:
    written = [bar(i) for i in range(20)]
    with EventStore(log) as store:
        for event in written:
            store.append(event)
    assert read_all(log) == written


def test_events_come_back_unchanged(log: Path) -> None:
    """Equal, not similar — a price differing in the eighth decimal changes
    what a strategy decides."""
    with EventStore(log) as store:
        store.append(bar(0))
    restored = read_all(log)[0]
    assert isinstance(restored, Bar)
    assert restored.open.value == Decimal("100.12345678")


def test_heterogeneous_events_interleave(log: Path) -> None:
    """A real session mixes market data, venue events and system output."""
    events: list[Event] = [
        SessionBoundary(
            venue=Venue.NASDAQ, ts_event=OPEN_NS, ts_init=OPEN_NS, session=MarketSession.REGULAR
        ),
        bar(1),
        Intent(
            symbol=AAPL,
            ts_event=OPEN_NS + 2 * MINUTE,
            ts_init=OPEN_NS + 2 * MINUTE,
            side=Side.BUY,
            entry=EntryTrigger.MARKET,
            entry_price=None,
            invalidation=Price("99"),
            target_r=Decimal(2),
            horizon_ns=MINUTE,
            strategy="orb",
            strategy_version="1.0.0",
            rationale="range break",
        ),
        TradingHalt(symbol=AAPL, ts_event=OPEN_NS + 3 * MINUTE, ts_init=OPEN_NS + 3 * MINUTE),
    ]
    with EventStore(log) as store:
        for event in events:
            store.append(event)
    assert read_all(log) == events


# ── Ordering ─────────────────────────────────────────────────


def test_events_come_back_in_sort_key_order(log: Path) -> None:
    """The replay contract: (ts_event, seq), whatever order they arrived in."""
    with EventStore(log) as store:
        for event in [bar(2), bar(0), bar(1)]:
            store.append(event)
    assert [event.ts_event for event in read_all(log)] == [OPEN_NS + i * MINUTE for i in (0, 1, 2)]


def test_seq_breaks_ties_on_equal_timestamps(log: Path) -> None:
    with EventStore(log) as store:
        for event in [bar(0, seq=2), bar(0, seq=0), bar(0, seq=1)]:
            store.append(event)
    assert [event.seq for event in read_all(log)] == [0, 1, 2]


# ── Append-only ──────────────────────────────────────────────


def test_reopening_continues_rather_than_truncating(log: Path) -> None:
    """Losing yesterday's log by opening today's would destroy the audit trail."""
    with EventStore(log) as store:
        store.append(bar(0))
    with EventStore(log) as store:
        store.append(bar(1))
    assert len(read_all(log)) == 2


def test_the_file_is_one_line_per_event(log: Path) -> None:
    with EventStore(log) as store:
        for i in range(5):
            store.append(bar(i))
    assert len(log.read_text().strip().splitlines()) == 5


def test_parent_directories_are_created(tmp_path: Path) -> None:
    nested = tmp_path / "runs" / "2026" / "session.jsonl"
    with EventStore(nested) as store:
        store.append(bar(0))
    assert nested.exists()


# ── Durability ───────────────────────────────────────────────


def test_records_reach_disk_without_closing(log: Path) -> None:
    """§6.3 needs a trade reconstructable later, which is not true of records
    sitting in a process buffer when it dies."""
    store = EventStore(log)
    try:
        store.append(bar(0))
        assert log.read_text().count("\n") == 1  # readable by another process already
    finally:
        store.close()


def test_buffered_mode_still_persists_on_close(log: Path) -> None:
    store = EventStore(log, buffered=True)
    store.append(bar(0))
    store.close()
    assert len(read_all(log)) == 1


def test_streaming_flushes_pending_writes(log: Path) -> None:
    """Reading your own writes must work without closing first."""
    store = EventStore(log, buffered=True)
    try:
        store.append(bar(0))
        assert len(list(store.stream(0, OPEN_NS + MINUTE))) == 1
    finally:
        store.close()


# ── Range filtering ──────────────────────────────────────────


def test_range_is_half_open(log: Path) -> None:
    with EventStore(log) as store:
        for i in range(10):
            store.append(bar(i))
    got = list(EventStore(log).stream(OPEN_NS + 2 * MINUTE, OPEN_NS + 5 * MINUTE))
    assert [event.ts_event for event in got] == [OPEN_NS + i * MINUTE for i in (2, 3, 4)]


def test_an_empty_range_returns_nothing(log: Path) -> None:
    with EventStore(log) as store:
        store.append(bar(5))
    assert list(EventStore(log).stream(0, OPEN_NS)) == []


# ── Absence and damage ───────────────────────────────────────


def test_a_log_that_does_not_exist_streams_empty(tmp_path: Path) -> None:
    """A session that has not run yet is ordinary, not an error."""
    assert list(EventStore(tmp_path / "never-written.jsonl").stream(0, 10**19)) == []


def test_blank_lines_are_ignored(log: Path) -> None:
    with EventStore(log) as store:
        store.append(bar(0))
    log.write_text(log.read_text() + "\n\n")
    assert len(read_all(log)) == 1


def test_a_corrupt_line_is_refused_and_located(log: Path) -> None:
    """A replay that skipped it would report success having simulated a
    different session."""
    with EventStore(log) as store:
        store.append(bar(0))
        store.append(bar(1))
    log.write_text(log.read_text() + "{not valid json\n")

    with pytest.raises(CorruptEventLog, match=r":3 could not be decoded"):
        read_all(log)


def test_an_unknown_event_tag_is_refused(log: Path) -> None:
    """A log from a newer build must not be partly replayed."""
    log.write_text('{"_t":"from_the_future","v":"1","ts_event":1,"ts_init":1,"seq":0}\n')
    with pytest.raises(CorruptEventLog, match="could not be decoded"):
        read_all(log)


# ── Housekeeping ─────────────────────────────────────────────


def test_close_is_idempotent(log: Path) -> None:
    store = EventStore(log)
    store.append(bar(0))
    store.close()
    store.close()
    assert len(read_all(log)) == 1


def test_path_is_exposed(log: Path) -> None:
    assert EventStore(log).path == log
