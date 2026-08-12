"""Market data events.

Every event carries two timestamps and a sequence number, and the distinction
between them is what makes the system honest about time.

- ``ts_event`` — when it happened at the venue. This is the only timestamp a
  feature or strategy may look at. Using anything else leaks information that
  was not available at the moment being modelled.
- ``ts_init`` — when we built the object. In live trading that is receipt time,
  so ``ts_init - ts_event`` is our data latency. In replay it is set from the
  ``SimClock``, so a replayed session reproduces exactly.
- ``seq`` — a monotonic counter breaking ties. Two prints can share a nanosecond
  timestamp; without a tiebreaker their order depends on dict ordering or
  scheduler luck, and the replay digest stops being reproducible.

Events sort by ``(ts_event, seq)`` and nothing else. That total order is the
backbone of gate G1.

Events are immutable. They are facts about the past, and a component that could
rewrite one could rewrite history between the backtest and the live run.

**Terms used here**, for readers coming from outside markets:

- *OHLCV bar* — a summary of all trading in one time window: the first price
  (open), highest, lowest, last (close), and total shares traded (volume).
- *Bid / ask* — the best price someone will currently buy at, and the best price
  someone will currently sell at. The gap between them is the *spread*, and
  crossing it is the main hidden cost of trading.
- *Order book* — the queue of resting buy and sell orders at each price.
- *Tape* — the stream of trades that actually executed, as opposed to orders
  merely resting on the book.
- *LULD halt* — Limit Up-Limit Down: US exchanges pause trading in a stock that
  moves too far too fast, then restart it with an auction.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from neurotrade.core.clock import Nanos
from neurotrade.core.types import Price, Quantity, Side, Symbol, Venue

__all__ = [
    "Bar",
    "BarInterval",
    "Event",
    "HaltReason",
    "MarketEvent",
    "MarketSession",
    "Quote",
    "SessionBoundary",
    "TickTrade",
    "TradingHalt",
    "TradingResumed",
    "VenueEvent",
]


class BarInterval(StrEnum):
    """Bar aggregation periods.

    1-minute is the corpus standard (§12.1); the rest are aggregated from it
    rather than fetched separately, so that every timeframe derives from one
    source of truth.

    Example:
        >>> BarInterval.MIN_1.nanos
        60000000000
    """

    SEC_1 = "1s"
    SEC_5 = "5s"
    SEC_15 = "15s"
    SEC_30 = "30s"
    MIN_1 = "1m"  # the corpus standard; everything else aggregates from this
    MIN_5 = "5m"
    MIN_15 = "15m"
    MIN_30 = "30m"
    HOUR_1 = "1h"
    DAY_1 = "1d"

    @property
    def nanos(self) -> Nanos:
        """Duration of one bar in nanoseconds.

        Example:
            >>> BarInterval.MIN_5.nanos
            300000000000
        """
        return _INTERVAL_NANOS[self]


_SECOND = 1_000_000_000
_INTERVAL_NANOS: dict[BarInterval, Nanos] = {
    BarInterval.SEC_1: _SECOND,
    BarInterval.SEC_5: 5 * _SECOND,
    BarInterval.SEC_15: 15 * _SECOND,
    BarInterval.SEC_30: 30 * _SECOND,
    BarInterval.MIN_1: 60 * _SECOND,
    BarInterval.MIN_5: 300 * _SECOND,
    BarInterval.MIN_15: 900 * _SECOND,
    BarInterval.MIN_30: 1_800 * _SECOND,
    BarInterval.HOUR_1: 3_600 * _SECOND,
    BarInterval.DAY_1: 86_400 * _SECOND,
}


@dataclass(frozen=True, slots=True, kw_only=True)
class Event:
    """Anything that happened, at a knowable time, in a knowable order.

    Holds only what every event needs: the two timestamps and the tiebreaker.
    Scope — which instrument, which venue — is added by the subclasses below,
    because not every event is about a single instrument.

    Keyword-only because these records are wide and positional construction of
    several near-identical integers is a bug waiting to happen.
    """

    ts_event: Nanos  # when it happened at the venue — the only time features may read
    ts_init: Nanos  # when we constructed it; receipt time live, SimClock in replay
    seq: int = 0  # tiebreaker for events sharing a ts_event; never negative

    def __post_init__(self) -> None:
        """Validate the tiebreaker.

        Raises:
            ValueError: If `seq` is negative, which would break the total order.
        """
        if self.seq < 0:
            raise ValueError(f"seq must not be negative: {self.seq}")

    @property
    def sort_key(self) -> tuple[Nanos, int]:
        """The total order used by replay. Nothing else may affect ordering.

        Returns:
            `(ts_event, seq)` — sorting any mix of events by this reproduces
            the exact stream the live system saw.

        Example:
            >>> a = Event(ts_event=100, ts_init=100, seq=1)
            >>> b = Event(ts_event=100, ts_init=100, seq=0)
            >>> [e.seq for e in sorted([a, b], key=lambda e: e.sort_key)]
            [0, 1]
        """
        return (self.ts_event, self.seq)

    @property
    def latency_ns(self) -> int:
        """Nanoseconds between the venue timestamp and our construction of it.

        Not constrained to be positive: clock skew between our host and the
        venue can make it negative, and rejecting that would discard real data
        rather than surface a real problem. Monitor it; do not enforce it.

        Example:
            >>> Event(ts_event=1_000, ts_init=1_500).latency_ns
            500
        """
        return self.ts_init - self.ts_event


@dataclass(frozen=True, slots=True, kw_only=True)
class MarketEvent(Event):
    """An observation about one instrument."""

    symbol: Symbol  # the instrument this observation is about


@dataclass(frozen=True, slots=True, kw_only=True)
class VenueEvent(Event):
    """An observation about a venue as a whole.

    Session transitions apply to every instrument listed on a venue at once.
    Modelling them as per-instrument events would mean emitting one per symbol
    in the universe at each boundary — thousands of identical records saying the
    same thing, and a replay whose ordering depends on universe membership.
    """

    venue: Venue  # the exchange this applies to, for all its listings


@dataclass(frozen=True, slots=True, kw_only=True)
class Bar(MarketEvent):
    """An OHLCV bar: a summary of one time window's trading.

    ``ts_event`` is the bar's **close** time — the instant the bar became a
    complete, observable fact. Stamping a bar with its open time is the classic
    lookahead bug: it lets a strategy act at 09:30:00 on a bar that does not
    finish forming until 09:31:00.

    Example:
        >>> bar = Bar(
        ...     symbol=Symbol("AAPL", Venue.NASDAQ),
        ...     ts_event=60_000_000_000, ts_init=60_000_000_000,
        ...     interval=BarInterval.MIN_1,
        ...     open=Price("100"), high=Price("102"),
        ...     low=Price("99"), close=Price("101"), volume=Quantity(1_000),
        ... )
        >>> (bar.range, bar.is_up, bar.ts_open)
        (Decimal('3'), True, 0)
    """

    interval: BarInterval  # the window length this bar summarises
    open: Price  # first traded price in the window
    high: Price  # highest traded price
    low: Price  # lowest traded price
    close: Price  # last traded price; the one strategies usually act on
    volume: Quantity  # total shares traded; may be zero for illiquid names
    vwap: Price | None = None  # volume-weighted average price, if the feed gives one
    trade_count: int | None = None  # number of individual trades, if available

    def __post_init__(self) -> None:
        """Validate that the four prices describe a possible bar.

        Raises:
            ValueError: If high is below low, or open, close or vwap fall
                outside `[low, high]`, or trade_count is negative. These are
                data-quality faults that feeds do produce, and letting one
                through corrupts every level derived from it.
        """
        # Explicit base call, not `super()`. With `slots=True` the dataclass
        # decorator returns a *new* class object, while the zero-argument
        # `super()` closure still refers to the original one — so `super()`
        # raises TypeError at runtime. The module-level name already points at
        # the rebuilt class by the time this runs.
        MarketEvent.__post_init__(self)
        if self.high < self.low:
            raise ValueError(f"high {self.high} is below low {self.low}")
        if not (self.low <= self.open <= self.high):
            raise ValueError(f"open {self.open} outside range [{self.low}, {self.high}]")
        if not (self.low <= self.close <= self.high):
            raise ValueError(f"close {self.close} outside range [{self.low}, {self.high}]")
        if self.vwap is not None and not (self.low <= self.vwap <= self.high):
            raise ValueError(f"vwap {self.vwap} outside range [{self.low}, {self.high}]")
        if self.trade_count is not None and self.trade_count < 0:
            raise ValueError(f"trade_count must not be negative: {self.trade_count}")

    @property
    def ts_open(self) -> Nanos:
        """When this bar started forming.

        Derived rather than stored, so there is exactly one timestamp on the
        record and no way for the two to disagree.

        Example:
            >>> Bar(
            ...     symbol=Symbol("AAPL", Venue.NASDAQ),
            ...     ts_event=120_000_000_000, ts_init=120_000_000_000,
            ...     interval=BarInterval.MIN_1, open=Price("1"), high=Price("1"),
            ...     low=Price("1"), close=Price("1"), volume=Quantity(0),
            ... ).ts_open
            60000000000
        """
        return self.ts_event - self.interval.nanos

    @property
    def range(self) -> Decimal:
        """High minus low — how far the price travelled in the window.

        Zero for a bar that never moved. Used as a raw volatility measure and as
        the input to ATR-style stop sizing (§6.1).
        """
        return self.high - self.low

    @property
    def is_up(self) -> bool:
        """True if the bar closed above where it opened."""
        return self.close > self.open


@dataclass(frozen=True, slots=True, kw_only=True)
class Quote(MarketEvent):
    """Top-of-book bid and ask — the best prices currently available.

    Crossed and locked books are **not** rejected. A locked book (bid == ask)
    and a crossed one (bid > ask) both occur in real consolidated data, briefly,
    around fast moves and across venues with different latencies. Rejecting them
    would silently delete exactly the moments worth studying. They are flagged
    instead, so a feature can decide what to do.

    Example:
        >>> quote = Quote(
        ...     symbol=Symbol("AAPL", Venue.NASDAQ), ts_event=0, ts_init=0,
        ...     bid_price=Price("100.00"), bid_size=Quantity(900),
        ...     ask_price=Price("100.02"), ask_size=Quantity(100),
        ... )
        >>> (quote.spread, quote.mid, quote.microprice)
        (Decimal('0.02'), Decimal('100.01'), Decimal('100.018'))
    """

    bid_price: Price  # best price a buyer is currently offering
    bid_size: Quantity  # shares available at the bid
    ask_price: Price  # best price a seller is currently asking
    ask_size: Quantity  # shares available at the ask

    @property
    def spread(self) -> Decimal:
        """Ask minus bid — the immediate round-trip cost per share.

        Negative when the book is crossed.
        """
        return self.ask_price - self.bid_price

    @property
    def mid(self) -> Decimal:
        """Arithmetic midpoint of bid and ask.

        The naive reference price. `microprice` is the better estimator of where
        the next trade prints, and §5.6 makes it the reference for entries,
        exits and adverse-selection measurement.
        """
        return (self.bid_price.value + self.ask_price.value) / 2

    @property
    def microprice(self) -> Decimal:
        """Size-weighted midpoint — a better estimate of the next mid.

        Each side's price is weighted by the *opposite* side's size. The
        intuition: a large resting bid and a thin ask means buyers are queued up
        and the thin ask will be consumed first, so price is more likely to move
        up. The estimate therefore leans toward the ask.

        Returns:
            The weighted midpoint, or `mid` when both sides are empty.

        Example:
            >>> q = Quote(
            ...     symbol=Symbol("AAPL", Venue.NASDAQ), ts_event=0, ts_init=0,
            ...     bid_price=Price("10.00"), bid_size=Quantity(500),
            ...     ask_price=Price("10.02"), ask_size=Quantity(500),
            ... )
            >>> q.microprice == q.mid                # balanced book, no lean
            True
        """
        total = self.bid_size.value + self.ask_size.value
        if total == 0:
            return self.mid
        weighted = (
            self.bid_price.value * self.ask_size.value + self.ask_price.value * self.bid_size.value
        )
        return weighted / total

    @property
    def is_locked(self) -> bool:
        """True when bid equals ask — a zero spread, briefly."""
        return self.bid_price == self.ask_price

    @property
    def is_crossed(self) -> bool:
        """True when the bid is above the ask, which is transiently possible.

        Happens across venues with different latencies during fast moves. Real
        data, not corruption — but features that assume a positive spread must
        check this first.
        """
        return self.bid_price > self.ask_price


@dataclass(frozen=True, slots=True, kw_only=True)
class TickTrade(MarketEvent):
    """A single executed trade print from the tape.

    ``aggressor`` is the side that crossed the spread to make the trade happen —
    the impatient party. The tape does not carry it; it is inferred (Lee-Ready
    or bulk-volume classification, §5.6) and stays ``None`` until something does
    that inference. ``None`` means unknown, never "neither".

    Example:
        >>> trade = TickTrade(
        ...     symbol=Symbol("AAPL", Venue.NASDAQ), ts_event=0, ts_init=0,
        ...     price=Price("100.50"), size=Quantity(200),
        ... )
        >>> (trade.notional, trade.aggressor)
        (Decimal('20100.00'), None)
    """

    price: Price  # price this trade executed at
    size: Quantity  # shares exchanged; always non-zero
    aggressor: Side | None = None  # who crossed the spread; None until inferred

    def __post_init__(self) -> None:
        """Validate the print.

        Raises:
            ValueError: If size is zero — a print with no size is malformed.
        """
        MarketEvent.__post_init__(self)  # see the note in Bar.__post_init__
        if self.size.is_zero:
            raise ValueError("a trade print must have non-zero size")

    @property
    def notional(self) -> Decimal:
        """Cash value of the trade: price times size."""
        return self.price.value * self.size.value


class MarketSession(StrEnum):
    """Phases of a trading day.

    These are venue facts about when orders may execute, not judgements about
    when it is sensible to trade. The 12:00-14:00 ET liquidity lull is a
    *regime* (§5.7) and is classified separately — it happens inside REGULAR.

    Example:
        >>> MarketSession.CLOSED.is_tradable
        False
    """

    PRE = "PRE"  # pre-market, thin and wide-spread
    REGULAR = "REGULAR"  # the main session, 09:30-16:00 ET for US venues
    POST = "POST"  # after-hours, thin again
    CLOSED = "CLOSED"  # no execution possible

    @property
    def is_tradable(self) -> bool:
        """True in any phase where orders can execute at all."""
        return self is not MarketSession.CLOSED


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionBoundary(VenueEvent):
    """A venue moving from one session phase into another.

    ``session`` is the phase being *entered*. Anchored reference levels depend
    on these: pre-market high/low, the opening range and session VWAP all reset
    at a boundary, so a missed transition silently corrupts every level derived
    from it.

    Example:
        >>> SessionBoundary(
        ...     venue=Venue.NASDAQ, ts_event=0, ts_init=0,
        ...     session=MarketSession.REGULAR,
        ... ).opens_regular_trading
        True
    """

    session: MarketSession  # the phase being entered, not the one being left

    @property
    def opens_regular_trading(self) -> bool:
        """True at the open — the moment most reference levels reset."""
        return self.session is MarketSession.REGULAR


class HaltReason(StrEnum):
    """Why trading in an instrument stopped.

    LULD is the common one intraday and the one §5.2's halt-resumption strategy
    targets. News halts resolve on a different timescale and are not tradable
    the same way, so the distinction matters to more than reporting.

    Example:
        >>> HaltReason.LULD.value
        'LULD'
    """

    LULD = "LULD"  # Limit Up-Limit Down: moved too far too fast
    NEWS_PENDING = "NEWS_PENDING"  # company announcement imminent
    NEWS_DISSEMINATION = "NEWS_DISSEMINATION"  # announcement being distributed
    REGULATORY = "REGULATORY"  # regulator-imposed; can last days
    OPERATIONAL = "OPERATIONAL"  # exchange technical issue
    UNKNOWN = "UNKNOWN"  # feed did not say


@dataclass(frozen=True, slots=True, kw_only=True)
class TradingHalt(MarketEvent):
    """Trading in one instrument has stopped.

    Two things must happen on receipt, and both are risk decisions rather than
    data handling: open positions cannot be exited while halted, and stale
    quotes must stop feeding features that assume a live market.

    Example:
        >>> TradingHalt(
        ...     symbol=Symbol("AAPL", Venue.NASDAQ), ts_event=0, ts_init=0,
        ... ).reason
        <HaltReason.UNKNOWN: 'UNKNOWN'>
    """

    reason: HaltReason = HaltReason.UNKNOWN  # UNKNOWN is honest; guessing LULD is not


@dataclass(frozen=True, slots=True, kw_only=True)
class TradingResumed(MarketEvent):
    """Trading in one instrument has restarted.

    ``auction_price`` is the reopening auction print where the venue publishes
    one. It is the first real price after the halt, and the reference the
    resumption strategy measures continuation against — the pre-halt price is
    stale by definition.

    Example:
        >>> TradingResumed(
        ...     symbol=Symbol("AAPL", Venue.NASDAQ), ts_event=0, ts_init=0,
        ...     auction_price=Price("187.40"),
        ... ).auction_price
        Price(value=Decimal('187.40'))
    """

    auction_price: Price | None = None  # reopening auction print, if published
