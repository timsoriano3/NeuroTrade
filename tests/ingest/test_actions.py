"""Tests for the action fetch and the gap audit.

Both work through ports, so both are driven here by fakes. The cases that
matter are the ones where an error could be mistaken for a clean result: a feed
that refuses a symbol must not leave that symbol looking like one that never
split, and a corpus with a missing action must not scan clean.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator, Sequence
from datetime import date
from decimal import Decimal

import pytest

from neurotrade.core.actions import CorporateAction
from neurotrade.core.clock import Nanos
from neurotrade.core.events import Bar, BarInterval
from neurotrade.core.types import Price, Quantity, Symbol, Venue
from neurotrade.core.universe import Universe
from neurotrade.ingest.actions import fetch_actions, scan_gaps

AAPL = Symbol("AAPL", Venue.NASDAQ)
MSFT = Symbol("MSFT", Venue.NASDAQ)
UNIVERSE = Universe([AAPL, MSFT])

DAY = 86_400_000_000_000
BASE_NS = 1_685_577_600_000_000_000  # 2023-06-01 00:00:00 UTC
SPAN = (date(2023, 6, 1), date(2023, 6, 30))


class FakeFeed:
    """A `CorporateActionsPort` that answers from a table, or refuses."""

    def __init__(
        self,
        actions: dict[Symbol, Sequence[CorporateAction]],
        refuse: set[Symbol] | None = None,
    ) -> None:
        self._actions = actions
        self._refuse = refuse or set()

    async def fetch_actions(
        self, symbol: Symbol, start: date, end: date
    ) -> Sequence[CorporateAction]:
        if symbol in self._refuse:
            raise RuntimeError("yahoo said no")
        return self._actions.get(symbol, ())


class FakeStore:
    """A `StoragePort` read side over an in-memory bar table."""

    def __init__(self, bars: dict[Symbol, Sequence[Bar]]) -> None:
        self._bars = bars

    def write_bars(self, bars: Sequence[Bar], *, source: str, session_date: date) -> None:
        raise AssertionError("the audit must never write")

    def read_bars(
        self, symbol: Symbol, interval: BarInterval, start: Nanos, end: Nanos
    ) -> Iterator[Bar]:
        return iter(bar for bar in self._bars.get(symbol, ()) if start <= bar.ts_event < end)


def bar(ts: int, price: str, symbol: Symbol = AAPL) -> Bar:
    value = Price(price)
    return Bar(
        symbol=symbol,
        ts_event=ts,
        ts_init=ts,
        interval=BarInterval.DAY_1,
        open=value,
        high=value,
        low=value,
        close=value,
        volume=Quantity("1000"),
    )


def split(day: date, ratio: str, symbol: Symbol = AAPL) -> CorporateAction:
    return CorporateAction(symbol, day, split_ratio=Decimal(ratio))


# ── fetch_actions ────────────────────────────────────────────────────────────


def test_every_symbol_is_fetched_including_empty_ones() -> None:
    """An empty result is a recorded fact, not an omission."""
    collected, report = asyncio.run(
        fetch_actions(
            UNIVERSE,
            FakeFeed({AAPL: (split(date(2023, 6, 2), "4"),)}),
            start=SPAN[0],
            end=SPAN[1],
        )
    )
    assert set(collected) == {AAPL, MSFT}
    assert collected[MSFT] == ()
    assert (report.symbols, report.splits, report.dividends) == (2, 1, 0)


def test_a_refused_symbol_is_a_failure_not_an_empty_result() -> None:
    """Storing it empty would assert the name never split."""
    collected, report = asyncio.run(
        fetch_actions(UNIVERSE, FakeFeed({}, refuse={MSFT}), start=SPAN[0], end=SPAN[1])
    )
    assert MSFT not in collected
    assert [symbol for symbol, _ in report.failures] == [MSFT]
    assert report.symbols == 1


def test_dividends_and_splits_are_counted_separately() -> None:
    both = CorporateAction(AAPL, date(2023, 6, 2), Decimal(4), Decimal("0.22"))
    _, report = asyncio.run(
        fetch_actions(UNIVERSE, FakeFeed({AAPL: (both,)}), start=SPAN[0], end=SPAN[1])
    )
    assert (report.splits, report.dividends) == (1, 1)


def test_an_inverted_range_is_refused() -> None:
    with pytest.raises(ValueError, match="is before start"):
        asyncio.run(
            fetch_actions(UNIVERSE, FakeFeed({}), start=date(2023, 6, 30), end=date(2023, 6, 1))
        )


def test_progress_is_reported_per_symbol() -> None:
    seen: list[tuple[Symbol, int]] = []
    asyncio.run(
        fetch_actions(
            UNIVERSE,
            FakeFeed({AAPL: (split(date(2023, 6, 2), "4"),)}),
            start=SPAN[0],
            end=SPAN[1],
            on_symbol=lambda symbol, count: seen.append((symbol, count)),
        )
    )
    assert seen == [(AAPL, 1), (MSFT, 0)]


# ── scan_gaps ────────────────────────────────────────────────────────────────


def test_a_recorded_split_explains_its_own_move() -> None:
    store = FakeStore({AAPL: [bar(BASE_NS, "400"), bar(BASE_NS + DAY, "100")]})
    report = scan_gaps(
        UNIVERSE,
        store,
        {AAPL: (split(date(2023, 6, 2), "4"),)},
        start=SPAN[0],
        end=SPAN[1],
    )
    assert report.clean
    assert (report.symbols, report.bars) == (1, 2)


def test_a_missing_split_is_caught() -> None:
    """The failure this whole commit exists to make visible."""
    store = FakeStore({AAPL: [bar(BASE_NS, "400"), bar(BASE_NS + DAY, "100")]})
    report = scan_gaps(UNIVERSE, store, {}, start=SPAN[0], end=SPAN[1])
    assert not report.clean
    assert report.gaps[0].session_date == date(2023, 6, 2)


def test_symbols_with_no_bars_are_not_counted_as_scanned() -> None:
    """A name absent from the corpus was not audited, and must not look like it was."""
    store = FakeStore({AAPL: [bar(BASE_NS, "100")]})
    report = scan_gaps(UNIVERSE, store, {}, start=SPAN[0], end=SPAN[1])
    assert report.symbols == 1


def test_a_tighter_threshold_catches_a_smaller_move() -> None:
    store = FakeStore({AAPL: [bar(BASE_NS, "100"), bar(BASE_NS + DAY, "80")]})
    assert scan_gaps(UNIVERSE, store, {}, start=SPAN[0], end=SPAN[1]).clean
    tighter = scan_gaps(UNIVERSE, store, {}, start=SPAN[0], end=SPAN[1], threshold=Decimal("0.1"))
    assert not tighter.clean


def test_each_symbol_is_adjusted_with_its_own_actions() -> None:
    """One company's split must not explain another's gap."""
    store = FakeStore(
        {
            AAPL: [bar(BASE_NS, "400"), bar(BASE_NS + DAY, "100")],
            MSFT: [bar(BASE_NS, "400", MSFT), bar(BASE_NS + DAY, "100", MSFT)],
        }
    )
    report = scan_gaps(
        UNIVERSE,
        store,
        {AAPL: (split(date(2023, 6, 2), "4"),)},
        start=SPAN[0],
        end=SPAN[1],
    )
    assert [gap.symbol for gap in report.gaps] == [MSFT]


def test_an_inverted_scan_range_is_refused() -> None:
    with pytest.raises(ValueError, match="is before start"):
        scan_gaps(UNIVERSE, FakeStore({}), {}, start=date(2023, 6, 30), end=date(2023, 6, 1))


def test_the_audit_never_writes() -> None:
    """`FakeStore.write_bars` asserts; reaching it fails the test."""
    store = FakeStore({AAPL: [bar(BASE_NS, "100")]})
    scan_gaps(UNIVERSE, store, {}, start=SPAN[0], end=SPAN[1])


def test_an_already_adjusted_source_is_not_adjusted_twice() -> None:
    """Yahoo's OHLC is split-adjusted even with `auto_adjust=False`.

    The bars below are already on one basis — no gap across the split date.
    Applying the split factor again would divide the earlier bar by four and
    manufacture the very gap the audit exists to rule out.
    """
    store = FakeStore({AAPL: [bar(BASE_NS, "100"), bar(BASE_NS + DAY, "101")]})
    report = scan_gaps(
        UNIVERSE,
        store,
        {AAPL: (split(date(2023, 6, 2), "4"),)},
        start=SPAN[0],
        end=SPAN[1],
        already_split_adjusted=True,
    )
    assert report.clean


def test_an_already_adjusted_source_still_catches_a_real_gap() -> None:
    """Skipping the factor must not skip the scan."""
    store = FakeStore({AAPL: [bar(BASE_NS, "100"), bar(BASE_NS + DAY, "25")]})
    report = scan_gaps(UNIVERSE, store, {}, start=SPAN[0], end=SPAN[1], already_split_adjusted=True)
    assert not report.clean
