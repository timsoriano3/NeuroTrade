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
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from neurotrade.core.clock import Nanos
from neurotrade.core.types import Price, Quantity, Side, Symbol

__all__ = [
    "Bar",
    "BarInterval",
    "MarketEvent",
    "Quote",
    "TickTrade",
]


class BarInterval(StrEnum):
    """Bar aggregation periods.

    1-minute is the corpus standard (§12.1); the rest are aggregated from it
    rather than fetched separately, so that every timeframe derives from one
    source of truth.
    """

    SEC_1 = "1s"
    SEC_5 = "5s"
    SEC_15 = "15s"
    SEC_30 = "30s"
    MIN_1 = "1m"
    MIN_5 = "5m"
    MIN_15 = "15m"
    MIN_30 = "30m"
    HOUR_1 = "1h"
    DAY_1 = "1d"

    @property
    def nanos(self) -> Nanos:
        """Duration of one bar in nanoseconds."""
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
class MarketEvent:
    """Base for anything observed from the market.

    Keyword-only because these records are wide and positional construction of
    four near-identical integers is a bug waiting to happen.
    """

    symbol: Symbol
    ts_event: Nanos
    ts_init: Nanos
    seq: int = 0

    def __post_init__(self) -> None:
        if self.seq < 0:
            raise ValueError(f"seq must not be negative: {self.seq}")

    @property
    def sort_key(self) -> tuple[Nanos, int]:
        """The total order used by replay. Nothing else may affect ordering."""
        return (self.ts_event, self.seq)

    @property
    def latency_ns(self) -> int:
        """Nanoseconds between the venue timestamp and our construction of it.

        Not constrained to be positive: clock skew between our host and the
        venue can make it negative, and rejecting that would discard real data
        rather than surface a real problem. Monitor it; do not enforce it.
        """
        return self.ts_init - self.ts_event


@dataclass(frozen=True, slots=True, kw_only=True)
class Bar(MarketEvent):
    """An OHLCV bar.

    ``ts_event`` is the bar's **close** time — the instant the bar became a
    complete, observable fact. Stamping a bar with its open time is the classic
    lookahead bug: it lets a strategy act at 09:30:00 on a bar that does not
    finish forming until 09:31:00.
    """

    interval: BarInterval
    open: Price
    high: Price
    low: Price
    close: Price
    volume: Quantity
    # IBKR supplies both on historical bars; other feeds may not.
    vwap: Price | None = None
    trade_count: int | None = None

    def __post_init__(self) -> None:
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
        """When this bar started forming."""
        return self.ts_event - self.interval.nanos

    @property
    def range(self) -> Decimal:
        """High minus low. Zero for a bar that never moved."""
        return self.high - self.low

    @property
    def is_up(self) -> bool:
        return self.close > self.open


@dataclass(frozen=True, slots=True, kw_only=True)
class Quote(MarketEvent):
    """Top-of-book bid and ask.

    Crossed and locked books are **not** rejected. A locked book (bid == ask)
    and a crossed one (bid > ask) both occur in real consolidated data, briefly,
    around fast moves and across venues with different latencies. Rejecting them
    would silently delete exactly the moments worth studying. They are flagged
    instead, so a feature can decide what to do.
    """

    bid_price: Price
    bid_size: Quantity
    ask_price: Price
    ask_size: Quantity

    @property
    def spread(self) -> Decimal:
        """Ask minus bid. Negative when the book is crossed."""
        return self.ask_price - self.bid_price

    @property
    def mid(self) -> Decimal:
        """Arithmetic midpoint.

        The naive reference price. `microprice` is the better estimator of where
        the next trade prints, and §5.6 makes it the reference for entries,
        exits and adverse-selection measurement.
        """
        return (self.bid_price.value + self.ask_price.value) / 2

    @property
    def microprice(self) -> Decimal:
        """Size-weighted midpoint — the depth-weighted estimator of the next mid.

        Weighted toward the *opposite* side's price: a large bid and a thin ask
        means buyers are queued up and price is more likely to move toward the
        ask. Falls back to `mid` when both sides are empty.
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
        return self.bid_price == self.ask_price

    @property
    def is_crossed(self) -> bool:
        return self.bid_price > self.ask_price


@dataclass(frozen=True, slots=True, kw_only=True)
class TickTrade(MarketEvent):
    """A single executed trade print from the tape.

    ``aggressor`` is the side that crossed the spread. The tape does not carry
    it; it is inferred (Lee-Ready or bulk-volume classification, §5.6) and stays
    ``None`` until something does that inference. ``None`` means unknown, never
    "neither".
    """

    price: Price
    size: Quantity
    aggressor: Side | None = None

    def __post_init__(self) -> None:
        MarketEvent.__post_init__(self)  # see the note in Bar.__post_init__
        if self.size.is_zero:
            raise ValueError("a trade print must have non-zero size")

    @property
    def notional(self) -> Decimal:
        return self.price.value * self.size.value
