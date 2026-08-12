"""Ports — the interfaces the trading core talks to the outside world through.

This module is what makes the architecture hexagonal (§4.1). The core defines
what it *needs*; adapters in ``adapters/`` supply implementations. Nothing here
imports IBKR, Parquet, DuckDB or Postgres, and nothing in ``core/`` ever will.
Swapping broker or storage becomes a new adapter rather than a rewrite, and —
more usefully day to day — the entire trading loop can be tested with in-memory
fakes, no network and no database.

These are `Protocol` classes, so conformance is **structural**: an adapter
satisfies a port by having the right methods, without importing or subclassing
anything from here. That keeps the dependency arrow pointing one way, from
adapters to core, which is the property `import-linter` will enforce in CI.

**Why some ports are async and others are not.** Async exists to stop a program
blocking while it waits on a network round trip. `BrokerPort` and
`MarketDataPort` cross a socket to IBKR, so they are async. `StoragePort` and
`EventStorePort` read and write local NVMe through DuckDB and Parquet, which is
CPU-bound work with no waiting to overlap; making them async would add overhead
and force every research script and notebook into an event loop for nothing.

**Fills are not return values.** `BrokerPort.submit` returns when the order has
been *accepted*, not when it has executed. Executions arrive later as `Fill`
events on the bus, because a single order can produce several fills, minutes
apart, or none at all. Any port shaped as `submit() -> Fill` would be a lie that
only holds for immediately-filled market orders.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Protocol, runtime_checkable

from neurotrade.core.clock import Nanos
from neurotrade.core.events import Bar, BarInterval, Event
from neurotrade.core.ids import OrderId
from neurotrade.core.orders import Order
from neurotrade.core.types import Symbol

__all__ = [
    "BrokerPort",
    "EventStorePort",
    "MarketDataPort",
    "StoragePort",
]


@runtime_checkable
class MarketDataPort(Protocol):
    """Source of market data from outside the system.

    Implemented by the IBKR adapter live, and by a fake reading from the corpus
    in backtest. Historical fetching is separated from the local corpus
    (`StoragePort`) on purpose: this one goes to the vendor and is rate-limited,
    that one reads what we already downloaded and is free.

    Example:
        Conformance is structural — no import from this module is needed:

        >>> class FakeFeed:
        ...     async def fetch_bars(self, symbol, interval, start, end):
        ...         return []
        ...     async def is_connected(self):
        ...         return True
        >>> isinstance(FakeFeed(), MarketDataPort)
        True
    """

    async def fetch_bars(
        self,
        symbol: Symbol,
        interval: BarInterval,
        start: Nanos,
        end: Nanos,
    ) -> Sequence[Bar]:
        """Fetch historical bars from the vendor.

        Args:
            symbol: Instrument to fetch.
            interval: Bar size. 1-minute is the corpus standard.
            start: Inclusive lower bound on `ts_event`.
            end: Exclusive upper bound on `ts_event`.

        Returns:
            Bars in ascending `ts_event` order. May be shorter than the range
            asked for — venues have no data outside session hours, and a halted
            instrument produces genuine gaps.

        Raises:
            Exception: Adapter-specific. Vendors rate-limit aggressively (IBKR
                allows roughly 60 historical requests per 10 minutes), so
                implementations are expected to pace themselves rather than
                letting callers discover the limit.
        """
        ...

    async def is_connected(self) -> bool:
        """Whether the feed is currently usable.

        Checked by the circuit breakers in §6.2: a disconnected feed means stale
        prices, and trading on stale prices is worse than not trading.
        """
        ...


@runtime_checkable
class BrokerPort(Protocol):
    """Where orders go and where account state comes from.

    The only component permitted to move real money. Everything upstream
    produces proposals and records; this is where they become irreversible.

    Example:
        >>> class FakeBroker:
        ...     async def submit(self, order): pass
        ...     async def cancel(self, order_id): pass
        ...     async def is_connected(self): return True
        >>> isinstance(FakeBroker(), BrokerPort)
        True
    """

    async def submit(self, order: Order) -> None:
        """Send an order to the venue.

        Returns once the broker has **accepted** the order, which is not the
        same as it having executed. Fills arrive afterwards as `Fill` events on
        the bus — see the module docstring for why this is not `-> Fill`.

        Args:
            order: The order to place. Its `id` is derived, so re-submitting an
                identical order produces the same id and can be detected as a
                duplicate rather than doubling the position.

        Raises:
            Exception: Adapter-specific, on rejection. Rejection is normal and
                expected — insufficient margin, a locked symbol, or a hard risk
                limit at the broker — and must not be treated as a system fault.
        """
        ...

    async def cancel(self, order_id: OrderId) -> None:
        """Request cancellation of a working order.

        Cancellation is a request, not a guarantee: an order can fill in the
        gap between the decision to cancel and the venue receiving it. Callers
        must handle a fill arriving after a successful cancel call.

        Args:
            order_id: The order to cancel.
        """
        ...

    async def is_connected(self) -> bool:
        """Whether the broker session is live.

        A disconnect mid-session is a risk event, not an inconvenience: open
        positions cannot be managed and stops cannot be honoured, which is why
        §6.2 specifies a flat-on-disconnect policy.
        """
        ...


@runtime_checkable
class StoragePort(Protocol):
    """The local corpus of market data.

    Backed by Parquet on NVMe behind DuckDB today, and by object storage when
    the corpus outgrows the disk (§19). Callers never learn which, which is the
    point — the migration trigger in §19 is meant to be a deployment change.

    Synchronous by design: see the module docstring.

    Example:
        >>> class InMemoryStore:
        ...     def write_bars(self, bars): pass
        ...     def read_bars(self, symbol, interval, start, end):
        ...         return iter(())
        >>> isinstance(InMemoryStore(), StoragePort)
        True
    """

    def write_bars(self, bars: Sequence[Bar]) -> None:
        """Append bars to the corpus.

        Implementations must be idempotent: the backfill crawler is resumable
        and will re-fetch ranges it already has after an interruption. Writing
        the same bar twice must not produce two rows, or every volume feature
        computed from the corpus doubles.

        Args:
            bars: Bars to persist. May span several symbols and dates.
        """
        ...

    def read_bars(
        self,
        symbol: Symbol,
        interval: BarInterval,
        start: Nanos,
        end: Nanos,
    ) -> Iterator[Bar]:
        """Read bars back out of the corpus.

        Args:
            symbol: Instrument to read.
            interval: Bar size.
            start: Inclusive lower bound on `ts_event`.
            end: Exclusive upper bound on `ts_event`.

        Returns:
            Bars in ascending `ts_event` order, as an iterator so a multi-year
            range does not have to fit in memory at once.
        """
        ...


@runtime_checkable
class EventStorePort(Protocol):
    """The append-only log every session is replayed from.

    This is the substrate gate G1 rests on. Every market event, intent, order
    and fill is appended here, and `stream` returns them in exactly the order
    they were seen — which is what lets a session be re-run bit-for-bit.

    Synchronous by design: see the module docstring.

    Example:
        >>> class MemoryLog:
        ...     def __init__(self): self._events = []
        ...     def append(self, event): self._events.append(event)
        ...     def stream(self, start, end):
        ...         return iter(sorted(self._events, key=lambda e: e.sort_key))
        >>> isinstance(MemoryLog(), EventStorePort)
        True
    """

    def append(self, event: Event) -> None:
        """Record an event.

        Append-only: there is no update and no delete. An event that turns out
        to be wrong is corrected by appending a correction, never by editing
        history — otherwise "what did the system know at 09:47" stops having a
        single answer.

        Args:
            event: Any `Event`. The store does not interpret it beyond its
                `sort_key`, so new event types need no changes here.
        """
        ...

    def stream(self, start: Nanos, end: Nanos) -> Iterator[Event]:
        """Replay events in their original order.

        Args:
            start: Inclusive lower bound on `ts_event`.
            end: Exclusive upper bound on `ts_event`.

        Returns:
            Events ordered by `(ts_event, seq)` — the total order defined in
            `core.events`. Any other ordering, including one that merely looks
            sorted, breaks replay determinism.
        """
        ...
