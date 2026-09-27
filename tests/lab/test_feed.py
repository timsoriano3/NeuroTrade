"""The corpus feed: merging a universe's bars into one ordered stream.

The property under test is *total* ordering. Bars tying on `ts_event` are the
normal case at one-minute resolution — every instrument closes its bar on the
same tick — so a merge that only sorts by time would be non-deterministic in
exactly the situation that arises every minute of every session.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import date
from decimal import Decimal

import pytest

from neurotrade.core.actions import AdjustmentSeries, CorporateAction
from neurotrade.core.events import Bar, BarInterval
from neurotrade.core.ports import StoragePort
from neurotrade.core.types import Price, Quantity, Symbol, Venue
from neurotrade.lab.feed import AdjustingStore, CorpusFeed, merge_bars

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


# ── Split adjustment on read ─────────────────────────────────

NVDA = Symbol("NVDA", Venue.NASDAQ)

_SESSION_NS = {
    date(2024, 6, 7): 1_717_790_400_000_000_000,  # the Friday before the 10:1
    date(2024, 6, 10): 1_718_049_600_000_000_000,  # the split's effective session
}


def priced(symbol: Symbol, ts: int, close: str, *, volume: str = "1000") -> Bar:
    """A bar whose prices matter, unlike `bar` above."""
    return Bar(
        symbol=symbol,
        ts_event=ts,
        ts_init=ts,
        interval=BarInterval.MIN_1,
        open=Price(close),
        high=Price(close),
        low=Price(close),
        close=Price(close),
        volume=Quantity(volume),
    )


def nvda_split() -> CorporateAction:
    """NVDA's real 10:1, effective 2024-06-10 — one of four in the crawled range."""
    return CorporateAction(NVDA, date(2024, 6, 10), split_ratio=Decimal(10))


def nvda_store(series: AdjustmentSeries) -> AdjustingStore:
    """A two-session NVDA corpus straddling the split, read through `series`."""
    inner = DictStore(
        {
            NVDA: [
                priced(NVDA, _SESSION_NS[date(2024, 6, 7)], "1200"),
                priced(NVDA, _SESSION_NS[date(2024, 6, 10)], "120"),
            ]
        }
    )
    return AdjustingStore(inner, {NVDA: series}, as_of=date(2024, 6, 30))


def test_a_ten_for_one_split_is_rescaled_into_one_basis() -> None:
    """The defect this class exists for: 1200 -> 120 is a 90% fall, not a gap."""
    store = nvda_store(AdjustmentSeries(NVDA, [nvda_split()]))
    closes = [b.close.value for b in store.read_bars(NVDA, BarInterval.MIN_1, 0, 2**63 - 1)]
    assert closes[0] == Decimal("120.00000000")  # pre-split, divided by ten
    assert closes[1] == Decimal("120")  # already in basis, untouched
    # The overnight move a strategy reads is now ~0%, not -90%.
    assert abs(closes[1] / closes[0] - 1) < Decimal("0.01")


def test_without_the_actions_the_same_corpus_shows_a_ninety_percent_gap() -> None:
    """Names the bug rather than only the fix: an empty series adjusts nothing.

    This is what every measurement on `--source ibkr` did before the wrapper,
    and what it still does for any split the actions corpus does not carry.
    """
    store = nvda_store(AdjustmentSeries(NVDA, ()))
    closes = [b.close.value for b in store.read_bars(NVDA, BarInterval.MIN_1, 0, 2**63 - 1)]
    assert closes[1] / closes[0] - 1 == Decimal("-0.9")


def test_volume_follows_the_split_the_other_way() -> None:
    """A 10:1 split multiplies the share count as it divides the price."""
    store = nvda_store(AdjustmentSeries(NVDA, [nvda_split()]))
    volumes = [b.volume.value for b in store.read_bars(NVDA, BarInterval.MIN_1, 0, 2**63 - 1)]
    assert volumes[0] == Decimal("10000.00000000")
    assert volumes[1] == Decimal("1000")


def test_a_symbol_with_no_actions_is_read_through_untouched() -> None:
    """Identity, not equality: the guarantee that lets an old digest reproduce.

    Every existing Phase 2 measurement ran on a window with no splits in it, so
    wrapping the corpus must be indistinguishable from not wrapping it there —
    otherwise a digest changing would be ambiguous between this change and a
    real one.
    """
    original = [bar(AAPL, 10), bar(AAPL, 20)]
    store = AdjustingStore(
        DictStore({AAPL: original}), {AAPL: AdjustmentSeries(AAPL, ())}, as_of=date(2024, 6, 30)
    )
    read = list(store.read_bars(AAPL, BarInterval.MIN_1, 0, 100))
    assert [b is o for b, o in zip(read, original, strict=True)] == [True, True]


def test_a_symbol_absent_from_the_mapping_is_also_read_through() -> None:
    original = [bar(AAPL, 10)]
    store = AdjustingStore(DictStore({AAPL: original}), {}, as_of=date(2024, 6, 30))
    assert next(iter(store.read_bars(AAPL, BarInterval.MIN_1, 0, 100))) is original[0]


def test_it_satisfies_the_storage_port() -> None:
    """Structural, so `measure_strategy` and the gate cannot tell it apart."""
    store = AdjustingStore(DictStore({}), {}, as_of=date(2024, 6, 30))
    assert isinstance(store, StoragePort)


def test_writing_adjusted_bars_back_is_refused() -> None:
    """`raw/` is immutable, and these prices carry a basis nothing records."""
    store = AdjustingStore(DictStore({}), {}, as_of=date(2024, 6, 30))
    with pytest.raises(NotImplementedError, match="read-side view"):
        store.write_bars([bar(AAPL, 10)], source="ibkr", session_date=date(2024, 6, 10))


def test_reading_stays_lazy() -> None:
    """A universe-wide read must not materialise a symbol's whole range."""
    store = AdjustingStore(
        DictStore({NVDA: [priced(NVDA, _SESSION_NS[date(2024, 6, 7)], "1200")]}),
        {NVDA: AdjustmentSeries(NVDA, [nvda_split()])},
        as_of=date(2024, 6, 30),
    )
    stream = store.read_bars(NVDA, BarInterval.MIN_1, 0, 2**63 - 1)
    assert isinstance(stream, Iterator)


# ── What the run reports ─────────────────────────────────────


def test_splits_in_force_excludes_one_before_the_window() -> None:
    """A split older than the first session has a factor of 1 on every bar in it."""
    store = AdjustingStore(
        DictStore({}), {NVDA: AdjustmentSeries(NVDA, [nvda_split()])}, as_of=date(2024, 6, 30)
    )
    assert store.splits_in_force(since=date(2024, 1, 1)) == ((NVDA, nvda_split()),)
    assert store.splits_in_force(since=date(2024, 6, 10)) == ()


def test_splits_in_force_excludes_a_dividend() -> None:
    """`total_return` is false, so a dividend is not applied and is not claimed."""
    payment = CorporateAction(NVDA, date(2024, 6, 12), dividend=Decimal("0.01"))
    store = AdjustingStore(
        DictStore({}), {NVDA: AdjustmentSeries(NVDA, [payment])}, as_of=date(2024, 6, 30)
    )
    assert store.splits_in_force(since=date(2024, 1, 1)) == ()


def test_splits_in_force_orders_by_date_then_symbol() -> None:
    wmt = Symbol("WMT", Venue.NYSE)
    store = AdjustingStore(
        DictStore({}),
        {
            NVDA: AdjustmentSeries(NVDA, [nvda_split()]),
            wmt: AdjustmentSeries(wmt, [CorporateAction(wmt, date(2024, 2, 26), Decimal(3))]),
        },
        as_of=date(2024, 6, 30),
    )
    assert [symbol.ticker for symbol, _ in store.splits_in_force(since=date(2024, 1, 1))] == [
        "WMT",
        "NVDA",
    ]
