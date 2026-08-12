"""Tests for the port protocols.

Ports have no behaviour, so what is worth testing is the architectural property
they exist for: that an adapter can satisfy one without importing anything from
core, and that core does not depend on any adapter.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import get_type_hints

import pytest

from neurotrade.core.clock import Nanos
from neurotrade.core.events import Bar, BarInterval, Event
from neurotrade.core.ids import OrderId
from neurotrade.core.orders import Order
from neurotrade.core.ports import (
    BrokerPort,
    EventStorePort,
    MarketDataPort,
    StoragePort,
)
from neurotrade.core.types import Price, Quantity, Symbol, Venue

AAPL = Symbol("AAPL", Venue.NASDAQ)
PORTS = [MarketDataPort, BrokerPort, StoragePort, EventStorePort]


# ── Structural conformance ───────────────────────────────────
#
# None of these fakes imports or subclasses anything from core.ports. That is
# the whole point: the dependency arrow runs from adapters to core, never back.


class FakeFeed:
    async def fetch_bars(
        self, symbol: Symbol, interval: BarInterval, start: Nanos, end: Nanos
    ) -> Sequence[Bar]:
        return []

    async def is_connected(self) -> bool:
        return True


class FakeBroker:
    def __init__(self) -> None:
        self.submitted: list[Order] = []
        self.cancelled: list[OrderId] = []

    async def submit(self, order: Order) -> None:
        self.submitted.append(order)

    async def cancel(self, order_id: OrderId) -> None:
        self.cancelled.append(order_id)

    async def is_connected(self) -> bool:
        return True


class FakeStore:
    def __init__(self) -> None:
        self.bars: list[Bar] = []

    def write_bars(self, bars: Sequence[Bar]) -> None:
        self.bars.extend(bars)

    def read_bars(
        self, symbol: Symbol, interval: BarInterval, start: Nanos, end: Nanos
    ) -> Iterator[Bar]:
        return iter(
            sorted(
                (
                    b
                    for b in self.bars
                    if b.symbol == symbol and b.interval is interval and start <= b.ts_event < end
                ),
                key=lambda b: b.sort_key,
            )
        )


class FakeEventStore:
    def __init__(self) -> None:
        self.events: list[Event] = []

    def append(self, event: Event) -> None:
        self.events.append(event)

    def stream(self, start: Nanos, end: Nanos) -> Iterator[Event]:
        return iter(
            sorted(
                (e for e in self.events if start <= e.ts_event < end),
                key=lambda e: e.sort_key,
            )
        )


@pytest.mark.parametrize(
    ("fake", "port"),
    [
        (FakeFeed(), MarketDataPort),
        (FakeBroker(), BrokerPort),
        (FakeStore(), StoragePort),
        (FakeEventStore(), EventStorePort),
    ],
)
def test_a_plain_class_satisfies_its_port(fake: object, port: type) -> None:
    """Conformance is structural — no inheritance, no registration."""
    assert isinstance(fake, port)


def test_a_class_missing_a_method_does_not_satisfy_the_port() -> None:
    class Incomplete:
        async def submit(self, order: Order) -> None: ...

    assert not isinstance(Incomplete(), BrokerPort)


# ── The architectural property ───────────────────────────────


def test_core_imports_no_adapter() -> None:
    """`core/` sits at the bottom of the dependency graph (§18).

    import-linter enforces this across the whole repo in CI; this catches the
    specific case that would make the hexagon meaningless — core reaching into
    an adapter — at test time.
    """
    core = Path(__file__).resolve().parents[2] / "src" / "neurotrade" / "core"
    offenders: dict[str, list[str]] = {}

    for module in core.rglob("*.py"):
        bad = []
        for node in ast.walk(ast.parse(module.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module:
                target = node.module
            elif isinstance(node, ast.Import):
                target = node.names[0].name
            else:
                continue
            if target.startswith("neurotrade.") and not target.startswith("neurotrade.core"):
                bad.append(target)
        if bad:
            offenders[module.name] = bad

    assert not offenders, f"core/ imported from outside core/: {offenders}"


def test_ports_are_protocols_not_base_classes() -> None:
    """Adapters must not need to inherit from us to be usable."""
    for port in PORTS:
        assert getattr(port, "_is_protocol", False), f"{port.__name__} is not a Protocol"


# ── Signatures ───────────────────────────────────────────────


def test_network_ports_are_async_and_local_ports_are_not() -> None:
    """Async is for waiting on sockets; local NVMe reads have nothing to wait on."""
    import inspect

    assert inspect.iscoroutinefunction(MarketDataPort.fetch_bars)
    assert inspect.iscoroutinefunction(BrokerPort.submit)
    assert not inspect.iscoroutinefunction(StoragePort.read_bars)
    assert not inspect.iscoroutinefunction(EventStorePort.append)


def test_submit_does_not_return_a_fill() -> None:
    """One order can produce several fills, minutes apart, or none.

    A `submit() -> Fill` signature would only be honest for immediately-filled
    market orders, so fills arrive as events instead.
    """
    assert get_type_hints(BrokerPort.submit)["return"] is type(None)


def test_event_store_accepts_any_event_type() -> None:
    """New event types must not require changes to the log."""
    assert get_type_hints(EventStorePort.append)["event"] is Event


# ── Fakes behave, so the tests above mean something ──────────


def test_fake_store_round_trips_a_bar() -> None:
    store = FakeStore()
    bar = Bar(
        symbol=AAPL,
        ts_event=1_000,
        ts_init=1_000,
        interval=BarInterval.MIN_1,
        open=Price("100"),
        high=Price("101"),
        low=Price("99"),
        close=Price("100.5"),
        volume=Quantity(10),
    )
    store.write_bars([bar])
    assert list(store.read_bars(AAPL, BarInterval.MIN_1, 0, 2_000)) == [bar]
    assert list(store.read_bars(AAPL, BarInterval.MIN_1, 2_000, 3_000)) == []


def test_fake_event_store_streams_in_sort_key_order() -> None:
    store = FakeEventStore()
    later = Event(ts_event=100, ts_init=100, seq=1)
    earlier = Event(ts_event=100, ts_init=100, seq=0)
    store.append(later)
    store.append(earlier)
    assert [e.seq for e in store.stream(0, 200)] == [0, 1]
