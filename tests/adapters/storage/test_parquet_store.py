"""Tests for the Parquet-backed corpus.

Idempotency is the property under test. A resumable crawler re-fetches ranges
after every interruption, and a duplicated bar does not look like corruption —
it looks like double the volume, which relative-volume ranking then puts at the
top of the watchlist.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from neurotrade.adapters.storage.parquet_store import ParquetStore
from neurotrade.core.clock import SimClock
from neurotrade.core.events import Bar, BarInterval
from neurotrade.core.ports import StoragePort
from neurotrade.core.types import Price, Quantity, Symbol, Venue

AAPL = Symbol("AAPL", Venue.NASDAQ)
SHOP = Symbol("SHOP", Venue.TSX)
TD_TSX = Symbol("TD", Venue.TSX)
TD_US = Symbol("TD", Venue.NASDAQ)

MINUTE = 60_000_000_000
OPEN_NS = 1_773_495_000_000_000_000
SESSION = date(2026, 3, 14)
NEXT_SESSION = date(2026, 3, 16)


@pytest.fixture
def store(tmp_path: Path) -> ParquetStore:
    return ParquetStore(tmp_path / "bars", SimClock(OPEN_NS))


def bar(
    index: int = 0,
    *,
    symbol: Symbol = AAPL,
    interval: BarInterval = BarInterval.MIN_1,
    close: str = "100",
) -> Bar:
    ts = OPEN_NS + index * MINUTE
    return Bar(
        symbol=symbol,
        ts_event=ts,
        ts_init=ts,
        seq=0,
        interval=interval,
        open=Price(close),
        high=Price(close),
        low=Price(close),
        close=Price(close),
        volume=Quantity(1_000),
    )


def session(count: int, **kwargs: object) -> list[Bar]:
    return [bar(i, **kwargs) for i in range(count)]  # type: ignore[arg-type]


def read_all(store: ParquetStore, symbol: Symbol = AAPL) -> list[Bar]:
    return list(store.read_bars(symbol, BarInterval.MIN_1, 0, OPEN_NS + 10_000 * MINUTE))


# ── Conformance ──────────────────────────────────────────────


def test_satisfies_the_storage_port(store: ParquetStore) -> None:
    """Structural conformance — no import from core.ports in the adapter."""
    assert isinstance(store, StoragePort)


# ── Idempotency: the requirement ─────────────────────────────


def test_writing_the_same_session_twice_does_not_duplicate(store: ParquetStore) -> None:
    """The crawler re-fetches after an interruption. This must be a no-op."""
    bars = session(10)
    store.write_bars(bars, source="ibkr", session_date=SESSION)
    store.write_bars(bars, source="ibkr", session_date=SESSION)
    assert read_all(store) == bars


def test_re_fetching_an_overlapping_range_merges(store: ParquetStore) -> None:
    """An interrupted crawl resumes from a boundary it already partly covered."""
    store.write_bars(session(6), source="ibkr", session_date=SESSION)
    store.write_bars(session(10)[4:], source="ibkr", session_date=SESSION)
    assert read_all(store) == session(10)


def test_a_later_write_extends_rather_than_replaces(store: ParquetStore) -> None:
    store.write_bars(session(3), source="ibkr", session_date=SESSION)
    store.write_bars([bar(9)], source="ibkr", session_date=SESSION)
    assert [b.ts_event for b in read_all(store)] == [OPEN_NS + i * MINUTE for i in (0, 1, 2, 9)]


def test_the_first_sighting_of_a_bar_is_kept(tmp_path: Path) -> None:
    """Re-fetching an identical bar must not restamp when we learned it."""
    clock = SimClock(OPEN_NS)
    store = ParquetStore(tmp_path / "bars", clock)
    store.write_bars([bar(0)], source="ibkr", session_date=SESSION)
    clock.advance_ns(3_600_000_000_000)
    store.write_bars([bar(0)], source="ibkr", session_date=SESSION)

    import pyarrow.parquet as pq

    from neurotrade.adapters.storage.schemas import partition_path

    rows = pq.read_table(partition_path(tmp_path / "bars", AAPL, SESSION) / "bars.parquet")
    assert rows.num_rows == 1
    assert rows.to_pylist()[0]["ingested_at"] == OPEN_NS


def test_bars_of_different_intervals_coexist(store: ParquetStore) -> None:
    """A 1-minute and a 5-minute bar can share a close time and are distinct."""
    store.write_bars(
        [bar(0, interval=BarInterval.MIN_1), bar(0, interval=BarInterval.MIN_5)],
        source="ibkr",
        session_date=SESSION,
    )
    assert len(read_all(store)) == 1  # only the 1-minute one
    five = list(store.read_bars(AAPL, BarInterval.MIN_5, 0, OPEN_NS + MINUTE))
    assert len(five) == 1


# ── Ordering ─────────────────────────────────────────────────


def test_bars_come_back_in_ascending_order(store: ParquetStore) -> None:
    store.write_bars(list(reversed(session(20))), source="ibkr", session_date=SESSION)
    assert read_all(store) == session(20)


def test_write_order_does_not_affect_stored_bytes(tmp_path: Path) -> None:
    """Two crawls fetching the same session in different orders agree on disk.

    Without this a re-crawl produces a different file for identical data, and
    any checksum over the corpus becomes meaningless.
    """
    from neurotrade.adapters.storage.schemas import partition_path

    forwards = ParquetStore(tmp_path / "a", SimClock(OPEN_NS))
    backwards = ParquetStore(tmp_path / "b", SimClock(OPEN_NS))
    forwards.write_bars(session(20), source="ibkr", session_date=SESSION)
    backwards.write_bars(list(reversed(session(20))), source="ibkr", session_date=SESSION)

    left = (partition_path(tmp_path / "a", AAPL, SESSION) / "bars.parquet").read_bytes()
    right = (partition_path(tmp_path / "b", AAPL, SESSION) / "bars.parquet").read_bytes()
    assert left == right


# ── Range reads ──────────────────────────────────────────────


def test_range_is_half_open(store: ParquetStore) -> None:
    """Start inclusive, end exclusive — so consecutive ranges tile without overlap."""
    store.write_bars(session(10), source="ibkr", session_date=SESSION)
    got = list(store.read_bars(AAPL, BarInterval.MIN_1, OPEN_NS + 2 * MINUTE, OPEN_NS + 5 * MINUTE))
    assert [b.ts_event for b in got] == [OPEN_NS + i * MINUTE for i in (2, 3, 4)]


def test_reads_span_sessions(store: ParquetStore) -> None:
    """A multi-day lookback crosses partitions transparently."""
    store.write_bars(session(3), source="ibkr", session_date=SESSION)
    store.write_bars(
        [bar(i, close="200") for i in range(1_000, 1_003)],
        source="ibkr",
        session_date=NEXT_SESSION,
    )
    assert len(read_all(store)) == 6


def test_reading_an_empty_range_returns_nothing(store: ParquetStore) -> None:
    store.write_bars(session(3), source="ibkr", session_date=SESSION)
    assert list(store.read_bars(AAPL, BarInterval.MIN_1, 0, OPEN_NS)) == []


def test_reading_an_unknown_symbol_returns_nothing(store: ParquetStore) -> None:
    """A missing instrument is ordinary during a backfill, not an error."""
    store.write_bars(session(3), source="ibkr", session_date=SESSION)
    assert read_all(store, SHOP) == []


def test_reading_before_anything_is_written(store: ParquetStore) -> None:
    assert read_all(store) == []


# ── Instrument isolation ─────────────────────────────────────


def test_same_ticker_on_two_venues_stays_separate(store: ParquetStore) -> None:
    """TD is Toronto-Dominion on TSX and Tandem Diabetes on NASDAQ."""
    store.write_bars(
        [bar(0, symbol=TD_TSX, close="80"), bar(0, symbol=TD_US, close="30")],
        source="ibkr",
        session_date=SESSION,
    )
    assert [b.close for b in read_all(store, TD_TSX)] == [Price("80")]
    assert [b.close for b in read_all(store, TD_US)] == [Price("30")]


def test_a_write_may_span_instruments(store: ParquetStore) -> None:
    store.write_bars(
        [bar(0, symbol=AAPL), bar(0, symbol=SHOP)], source="ibkr", session_date=SESSION
    )
    assert len(read_all(store, AAPL)) == 1
    assert len(read_all(store, SHOP)) == 1


# ── Provenance and validation ────────────────────────────────


def test_an_unknown_source_is_rejected(store: ParquetStore) -> None:
    """A typo would scatter an unqueryable provenance value through the corpus."""
    with pytest.raises(ValueError, match="typo"):
        store.write_bars(session(1), source="typo", session_date=SESSION)


def test_writing_nothing_is_a_no_op(store: ParquetStore) -> None:
    """A holiday or a halted symbol legitimately yields no bars."""
    store.write_bars([], source="ibkr", session_date=SESSION)
    assert read_all(store) == []


def test_ingested_at_comes_from_the_clock(tmp_path: Path) -> None:
    """Not the OS clock — so a replayed write produces identical bytes."""
    import pyarrow.parquet as pq

    from neurotrade.adapters.storage.schemas import partition_path

    ParquetStore(tmp_path / "bars", SimClock(42)).write_bars(
        [bar(0)], source="ibkr", session_date=SESSION
    )
    rows = pq.read_table(partition_path(tmp_path / "bars", AAPL, SESSION) / "bars.parquet")
    assert rows.to_pylist()[0]["ingested_at"] == 42


# ── Session listing ──────────────────────────────────────────


def test_sessions_lists_what_is_held(store: ParquetStore) -> None:
    """The crawler resumes from this, so it reads directory names only."""
    store.write_bars(session(1), source="ibkr", session_date=SESSION)
    store.write_bars(session(1), source="ibkr", session_date=NEXT_SESSION)
    assert store.sessions(AAPL) == (SESSION, NEXT_SESSION)


def test_sessions_is_empty_for_an_unknown_symbol(store: ParquetStore) -> None:
    assert store.sessions(SHOP) == ()


def test_sessions_is_per_instrument(store: ParquetStore) -> None:
    store.write_bars(session(1), source="ibkr", session_date=SESSION)
    store.write_bars([bar(0, symbol=SHOP)], source="ibkr", session_date=NEXT_SESSION)
    assert store.sessions(AAPL) == (SESSION,)
    assert store.sessions(SHOP) == (NEXT_SESSION,)
