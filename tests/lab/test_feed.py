"""The corpus feed: merging a universe's bars into one ordered stream.

The property under test is *total* ordering. Bars tying on `ts_event` are the
normal case at one-minute resolution — every instrument closes its bar on the
same tick — so a merge that only sorts by time would be non-deterministic in
exactly the situation that arises every minute of every session.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import date

import pytest

from neurotrade.core.events import Bar, BarInterval
from neurotrade.core.types import Price, Quantity, Symbol, Venue
from neurotrade.lab.feed import CorpusFeed, merge_bars

AAPL = Symbol("AAPL", Venue.NASDAQ)
MSFT = Symbol("MSFT", Venue.NASDAQ)
SPY = Symbol("SPY", Venue.ARCA)


def bar(symbol: Symbol, ts: int, *, seq: int = 0) -> Bar:
    """A minimal one-minute bar. Prices are irrelevant to ordering."""
    return Bar(
        symbol=symbol,
        ts_event=ts,
        ts_init=ts,
        seq=seq,
        interval=BarInterval.MIN_1,
        open=Price("100"),
        high=Price("101"),
        low=Price("99"),
        close=Price("100.5"),
        volume=Quantity(1_000),
    )


class DictStore:
    """A `StoragePort` over a dict of per-symbol bars."""

    def __init__(self, bars: dict[Symbol, Sequence[Bar]]) -> None:
        self._bars = bars
        self.reads: list[Symbol] = []

    def write_bars(self, bars: Sequence[Bar], *, source: str, session_date: date) -> None:
        raise AssertionError("the feed must never write")

    def read_bars(
        self, symbol: Symbol, interval: BarInterval, start: int, end: int
    ) -> Iterator[Bar]:
        self.reads.append(symbol)
        return iter([b for b in self._bars.get(symbol, ()) if start <= b.ts_event < end])


# ── Ordering ─────────────────────────────────────────────────


def test_merges_two_symbols_in_time_order() -> None:
    store = DictStore({AAPL: [bar(AAPL, 10), bar(AAPL, 30)], MSFT: [bar(MSFT, 20), bar(MSFT, 40)]})
    merged = merge_bars(store, [AAPL, MSFT], BarInterval.MIN_1, 0, 100)
    assert [(b.symbol.ticker, b.ts_event) for b in merged] == [
        ("AAPL", 10),
        ("MSFT", 20),
        ("AAPL", 30),
        ("MSFT", 40),
    ]


def test_ties_on_timestamp_break_by_symbol() -> None:
    """The every-minute case: all instruments close their bar on the same tick."""
    store = DictStore({s: [bar(s, 10)] for s in (AAPL, MSFT, SPY)})
    merged = merge_bars(store, [SPY, MSFT, AAPL], BarInterval.MIN_1, 0, 100)
    assert [b.symbol.ticker for b in merged] == ["AAPL", "MSFT", "SPY"]


def test_ties_break_by_seq_before_symbol() -> None:
    """`seq` outranks the ticker: it is the event's own declared tiebreaker."""
    store = DictStore({MSFT: [bar(MSFT, 10, seq=0)], AAPL: [bar(AAPL, 10, seq=1)]})
    merged = merge_bars(store, [AAPL, MSFT], BarInterval.MIN_1, 0, 100)
    assert [b.symbol.ticker for b in merged] == ["MSFT", "AAPL"]


@pytest.mark.parametrize(
    "order",
    [(AAPL, MSFT, SPY), (SPY, AAPL, MSFT), (MSFT, SPY, AAPL)],
    ids=["sorted", "reversed-ish", "shuffled"],
)
def test_caller_symbol_order_cannot_change_the_result(
    order: tuple[Symbol, ...],
) -> None:
    """Determinism: a set or dict-keys argument must not move the output."""
    store = DictStore({s: [bar(s, 10), bar(s, 20)] for s in (AAPL, MSFT, SPY)})
    merged = merge_bars(store, order, BarInterval.MIN_1, 0, 100)
    assert [(b.ts_event, b.symbol.ticker) for b in merged] == [
        (10, "AAPL"),
        (10, "MSFT"),
        (10, "SPY"),
        (20, "AAPL"),
        (20, "MSFT"),
        (20, "SPY"),
    ]


def test_duplicate_symbols_are_read_once() -> None:
    store = DictStore({AAPL: [bar(AAPL, 10)]})
    assert len(list(merge_bars(store, [AAPL, AAPL], BarInterval.MIN_1, 0, 100))) == 1
    assert store.reads == [AAPL]


# ── Ranges and emptiness ─────────────────────────────────────


def test_range_is_half_open() -> None:
    store = DictStore({AAPL: [bar(AAPL, 10), bar(AAPL, 20), bar(AAPL, 30)]})
    merged = merge_bars(store, [AAPL], BarInterval.MIN_1, 10, 30)
    assert [b.ts_event for b in merged] == [10, 20]


def test_empty_universe_yields_nothing() -> None:
    assert list(merge_bars(DictStore({}), [], BarInterval.MIN_1, 0, 100)) == []


def test_symbol_absent_from_corpus_is_not_an_error() -> None:
    """A name with no bars for the range is an ordinary outcome, not a fault."""
    store = DictStore({AAPL: [bar(AAPL, 10)]})
    merged = merge_bars(store, [AAPL, MSFT], BarInterval.MIN_1, 0, 100)
    assert [b.symbol.ticker for b in merged] == ["AAPL"]


# ── Laziness ─────────────────────────────────────────────────


def test_bars_are_not_materialised_up_front() -> None:
    """A five-year run must hold one bar per symbol, not five years of them."""
    pulled: list[int] = []

    class CountingStore(DictStore):
        def read_bars(
            self, symbol: Symbol, interval: BarInterval, start: int, end: int
        ) -> Iterator[Bar]:
            for b in super().read_bars(symbol, interval, start, end):
                pulled.append(b.ts_event)
                yield b

    store = CountingStore({AAPL: [bar(AAPL, t) for t in range(0, 100, 10)]})
    merged = merge_bars(store, [AAPL], BarInterval.MIN_1, 0, 100)
    next(iter(merged))
    assert len(pulled) == 1


# ── CorpusFeed ───────────────────────────────────────────────


def test_feed_binds_store_symbols_and_interval() -> None:
    store = DictStore({AAPL: [bar(AAPL, 10)], MSFT: [bar(MSFT, 20)]})
    feed = CorpusFeed(store, (AAPL, MSFT), BarInterval.MIN_1)
    assert [b.symbol.ticker for b in feed.events(0, 100)] == ["AAPL", "MSFT"]


def test_feed_defaults_to_one_minute_bars() -> None:
    """§12.1's corpus standard; everything else aggregates from it."""
    assert CorpusFeed(DictStore({}), ()).interval is BarInterval.MIN_1


def test_feed_events_is_repeatable() -> None:
    """Each call re-reads, so a CV fold can run the same range twice."""
    store = DictStore({AAPL: [bar(AAPL, 10), bar(AAPL, 20)]})
    feed = CorpusFeed(store, (AAPL,), BarInterval.MIN_1)
    assert [b.ts_event for b in feed.events(0, 100)] == [b.ts_event for b in feed.events(0, 100)]
