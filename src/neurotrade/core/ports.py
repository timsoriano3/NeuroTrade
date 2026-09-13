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

from collections.abc import Iterator, Mapping, Sequence
from datetime import date
from typing import Protocol, runtime_checkable

from neurotrade.core.calendar import TradingSession
from neurotrade.core.clock import Nanos
from neurotrade.core.events import Bar, BarInterval, Event
from neurotrade.core.ids import OrderId
from neurotrade.core.orders import Order
from neurotrade.core.types import Symbol, Venue
from neurotrade.core.universe import Universe

__all__ = [
    "BrokerPort",
    "CalendarPort",
    "CatalogPort",
    "EventStorePort",
    "MarketDataPort",
    "StoragePort",
    "UniversePort",
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
        ...     def write_bars(self, bars, *, source, session_date): pass
        ...     def read_bars(self, symbol, interval, start, end):
        ...         return iter(())
        >>> isinstance(InMemoryStore(), StoragePort)
        True
    """

    def write_bars(self, bars: Sequence[Bar], *, source: str, session_date: date) -> None:
        """Persist one session's bars.

        Implementations must be idempotent: the backfill crawler is resumable
        and will re-fetch ranges it already has after an interruption. Writing
        the same bar twice must not produce two rows, or every volume feature
        computed from the corpus doubles.

        Args:
            bars: Bars to persist. May span symbols; all belong to one session.
            source: Which feed produced them. Recorded per row, because sources
                disagree — IEX-only data has partial volume, free samples have
                gaps — and a volume feature is only interpretable if you know
                which one a bar came from.
            session_date: The trading day, in the venue's terms. Passed rather
                than derived from `ts_event`: a US post-market bar at 19:30 ET
                is 00:30 UTC the next day, so deriving would split one session
                across two days. The caller has the venue calendar; storage
                does not.
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


@runtime_checkable
class CalendarPort(Protocol):
    """Which days a venue traded, and between which times.

    Synchronous for the same reason `StoragePort` is: an exchange calendar is a
    local computation over holiday rules, not a network round trip, and forcing
    every research script into an event loop for it would buy nothing.

    Venues disagree, which is the whole reason this takes a `Venue` rather than
    being a single global calendar. TSX trades through US Thanksgiving; NYSE
    trades through Canada Day. A crawler that assumes one calendar for both
    countries reports phantom gaps on one venue and misses real ones on the
    other.

    **Implementations must be deterministic for a given library version.** Two
    calls with the same arguments return the same sessions, and an upgrade that
    would silently change history for a date already in the corpus is a defect —
    pin the version and pin a fixture that fails the build if it moves.
    """

    def sessions(self, venue: Venue, start: date, end: date) -> tuple[date, ...]:
        """Dates the venue traded within an inclusive range.

        This is the crawler's work queue once differenced against what the
        corpus holds — see `missing_sessions` on
        `adapters.storage.duckdb_catalog.DuckDBCatalog`, which takes exactly
        this list. It lives in an adapter, so joining the two is the caller's
        job; `core/` cannot import it.

        Args:
            venue: Listing venue. `Venue.SMART` is an order route and must raise.
            start: First date to consider, inclusive.
            end: Last date to consider, inclusive.

        Returns:
            Session dates in ascending order. Empty when the venue never traded
            in the range — a holiday week, or a range before the venue existed.
            Weekends and holidays are absent, not present-and-empty.
        """
        ...

    def session(self, venue: Venue, session_date: date) -> TradingSession | None:
        """The bounds of one session, or `None` if the venue was closed.

        `None` rather than an exception because "was the venue open?" is an
        ordinary question with an ordinary negative answer, asked constantly by
        the crawler. Reserving the exception for genuinely bad input — an
        unsupported venue — keeps a holiday from looking like a fault.

        Args:
            venue: Listing venue.
            session_date: The date to look up, in the venue's own terms.

        Returns:
            The session, or `None` if that date was not a trading day.
        """
        ...


@runtime_checkable
class UniversePort(Protocol):
    """Which instruments the system is allowed to consider.

    The *source* of a universe changes with every phase while the universe
    itself does not. Phase 0 reads a hand-written file because §12.1 stage 1
    needs a crawl queue before stage 3 has built any history. Later, §5's
    Universe Selector ranks candidates nightly by relative volume, catalyst
    tags and liquidity. Both answer the same question, so both sit behind this.

    Synchronous, like `CalendarPort` and `StoragePort`: reading a local file or
    a cached ranking is not a network round trip, and making it async would push
    every research script into an event loop for nothing.

    **Implementations must answer the same thing throughout a run.** A universe
    that changed mid-crawl would make the work already done unattributable —
    the corpus would hold symbols no recorded universe contains. Re-read between
    runs, never within one.

    Example:
        Conformance is structural — no import from this module is needed:

        >>> class FixedUniverse:
        ...     def universe(self):
        ...         return Universe([Symbol("AAPL", Venue.NASDAQ)])
        >>> isinstance(FixedUniverse(), UniversePort)
        True
    """

    def universe(self) -> Universe:
        """The instruments in scope.

        Returns:
            A non-empty `Universe`. Emptiness is rejected at construction
            rather than returned, because every caller would read an empty
            universe as "nothing to do" and report success having done nothing.
        """
        ...


@runtime_checkable
class CatalogPort(Protocol):
    """What the local corpus already holds.

    Separate from `StoragePort` because the questions are different in kind.
    That one moves bars in and out; this one describes the collection, which is
    what the backfill crawler needs to decide what to fetch next and what §12.1
    stage 5's quality gate needs to decide whether the corpus can be trusted.

    Synchronous, like `StoragePort`, and for the same reason.

    **An empty corpus is not an error.** During a backfill that runs for weeks,
    "nothing yet" is the normal answer for most instruments. An implementation
    that raised would make the crawler's own progress reporting the first thing
    to break.

    Example:
        Conformance is structural — no import from this module is needed:

        >>> class EmptyCatalog:
        ...     def bar_counts(self, symbol, interval):
        ...         return {}
        >>> isinstance(EmptyCatalog(), CatalogPort)
        True
    """

    def bar_counts(self, symbol: Symbol, interval: BarInterval) -> Mapping[date, int]:
        """How many bars are held for each session of one instrument.

        A count rather than a yes-or-no, because "held" and "complete" are not
        the same thing: an interrupted fetch leaves a session with some of its
        bars, and a crawler that treated presence as completeness would leave
        those holes forever. Comparing the count against
        `TradingSession.expected_bars` is what turns it into a decision, and
        that comparison belongs to the caller, which has the calendar.

        Args:
            symbol: Instrument to describe.
            interval: Bar size. Counts are per interval — a session holding
                daily bars holds no minute bars.

        Returns:
            Session date to bar count, for sessions holding at least one bar.
            Absent means none held. Empty when the corpus holds nothing for
            this instrument, which is ordinary rather than exceptional.
        """
        ...
