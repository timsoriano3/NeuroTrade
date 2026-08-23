"""Turning events into bytes and back, without losing anything.

The event log is what gate G1 rests on: replay a session and the system must
make the same decisions it made the first time. That only holds if an event
decoded from the log is *equal* to the event that was written — not similar,
equal — so every choice here is made in favour of exactness over compactness.

**Decimals are written as strings.** A JSON number is a float, and
`json.loads("100.12345678")` gives back a float that is not that number. Writing
`"100.12345678"` and rebuilding the `Decimal` from the string is the only way the
prices in the log survive a round trip. This is the same reason the Parquet
schema uses `DECIMAL` rather than `DOUBLE`.

**Each event type registers its own codec, explicitly.** Reflecting over
dataclass fields would be shorter and would silently change the log format the
moment someone adds a field — old logs would then decode into events missing
that field, and a replay would diverge for a reason nothing announces. An
explicit codec means adding a field is a decision, taken in one place, with a
version bump if it matters.

**Nothing is optional to register.** A test walks every `Event` subclass in the
package and fails if one has no codec, so a new event type cannot reach the log
as an unserialisable object discovered at 09:31 on a Monday.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from neurotrade.core.events import (
    Bar,
    BarInterval,
    Event,
    HaltReason,
    MarketSession,
    Quote,
    SessionBoundary,
    TickTrade,
    TradingHalt,
    TradingResumed,
)
from neurotrade.core.ids import FillId, IntentId, OrderId
from neurotrade.core.intent import EntryTrigger, Intent
from neurotrade.core.orders import Fill, LiquidityFlag, Order, OrderType, TimeInForce
from neurotrade.core.types import Currency, Money, Price, Quantity, Side, Symbol, Venue

__all__ = [
    "CODEC_VERSION",
    "EventCodec",
    "UnknownEventType",
    "codec",
]

CODEC_VERSION = "1"
"""Written on every record. A log encoded under a different version is refused
rather than decoded on a guess — a replay against misread events is worse than
one that will not start."""

type Encoded = dict[str, Any]

_TYPE_KEY = "_t"
"""Kept short because it appears on every record; a session's log holds hundreds
of thousands of them."""


class UnknownEventType(ValueError):
    """Raised when a log holds a type tag nothing knows how to decode.

    Means the log was written by a newer build, or an event type was removed.
    Either way the replay must stop: silently skipping unknown records would
    produce a run that looks successful and is missing events.
    """


# ── Value helpers ────────────────────────────────────────────


def _symbol(value: Symbol) -> str:
    """Encode an instrument as `TICKER.VENUE`."""
    return str(value)


def _parse_symbol(value: str) -> Symbol:
    """Decode `TICKER.VENUE`, splitting on the last dot.

    Splitting on the *last* dot matters: real tickers contain dots — `BRK.B`
    encodes as `BRK.B.NYSE`, and splitting on the first would give a ticker of
    `BRK` on a venue of `B.NYSE`.
    """
    ticker, _, venue = value.rpartition(".")
    return Symbol(ticker, Venue(venue))


def _price(value: Price | None) -> str | None:
    return None if value is None else str(value.value)


def _parse_price(value: str | None) -> Price | None:
    return None if value is None else Price(Decimal(value))


def _money(value: Money) -> dict[str, str]:
    return {"amount": str(value.amount), "currency": value.currency.value}


def _parse_money(value: dict[str, str]) -> Money:
    return Money(Decimal(value["amount"]), Currency(value["currency"]))


def _base(event: Event) -> Encoded:
    """The three fields every event carries."""
    return {"ts_event": event.ts_event, "ts_init": event.ts_init, "seq": event.seq}


@dataclass(frozen=True, slots=True)
class _Codec:
    """How one event type is written and read."""

    tag: str  # short type marker stored on every record
    type: type[Event]  # the class it builds
    encode: Callable[[Any], Encoded]  # event -> fields, excluding the tag
    decode: Callable[[Encoded], Event]  # fields -> event


class EventCodec:
    """Registry of per-type codecs, and the JSON line format around them.

    Example:
        >>> line = codec.dumps(
        ...     Bar(symbol=AAPL, ts_event=1_000, ts_init=1_000,
        ...         interval=BarInterval.MIN_1, open=Price("100"), high=Price("101"),
        ...         low=Price("99"), close=Price("100.5"), volume=Quantity(10))
        ... )
        >>> codec.loads(line).close
        Price(value=Decimal('100.5'))
    """

    __slots__ = ("_by_tag", "_by_type")

    def __init__(self) -> None:
        self._by_tag: dict[str, _Codec] = {}
        self._by_type: dict[type[Event], _Codec] = {}

    def register(
        self,
        tag: str,
        event_type: type[Event],
        encode: Callable[[Any], Encoded],
        decode: Callable[[Encoded], Event],
    ) -> None:
        """Teach the codec about one event type.

        Args:
            tag: Short stable marker, e.g. `"bar"`. Never reused or renamed —
                it is written into every historical log and changing it makes
                those logs undecodable.
            event_type: The class.
            encode: Event to fields, excluding the type tag.
            decode: Fields back to an event.

        Raises:
            ValueError: If the tag or type is already registered.
        """
        if tag in self._by_tag:
            raise ValueError(f"event tag {tag!r} is already registered")
        if event_type in self._by_type:
            raise ValueError(f"{event_type.__name__} already has a codec")
        entry = _Codec(tag=tag, type=event_type, encode=encode, decode=decode)
        self._by_tag[tag] = entry
        self._by_type[event_type] = entry

    def encode(self, event: Event) -> Encoded:
        """Encode one event, including its type tag and codec version.

        Raises:
            UnknownEventType: If the event's class has no registered codec.
        """
        entry = self._by_type.get(type(event))
        if entry is None:
            raise UnknownEventType(
                f"{type(event).__name__} has no codec; register one in event_codec.py"
            )
        return {_TYPE_KEY: entry.tag, "v": CODEC_VERSION, **entry.encode(event)}

    def decode(self, data: Encoded) -> Event:
        """Rebuild an event from its encoded form.

        Raises:
            UnknownEventType: If the tag is unknown or the version differs.
        """
        version = data.get("v")
        if version != CODEC_VERSION:
            raise UnknownEventType(
                f"record was written by codec version {version!r}; "
                f"this build reads {CODEC_VERSION!r}"
            )
        tag = data.get(_TYPE_KEY)
        entry = self._by_tag.get(str(tag))
        if entry is None:
            raise UnknownEventType(f"no codec for event tag {tag!r}")
        return entry.decode(data)

    def dumps(self, event: Event) -> str:
        """Encode to one JSON line.

        Keys are sorted so the bytes depend only on the event's content — two
        runs that produce the same events produce the same file, which is what
        lets a replay be compared byte for byte.
        """
        return json.dumps(self.encode(event), sort_keys=True, separators=(",", ":"))

    def loads(self, line: str) -> Event:
        """Decode one JSON line."""
        return self.decode(json.loads(line))

    @property
    def registered(self) -> tuple[type[Event], ...]:
        """Every event type with a codec."""
        return tuple(self._by_type)


codec = EventCodec()
"""The one codec. Module-level because the log format is a property of the
build, not of a particular run — two stores in one process must agree."""


# ── Market data ──────────────────────────────────────────────

codec.register(
    "bar",
    Bar,
    lambda event: {
        **_base(event),
        "symbol": _symbol(event.symbol),
        "interval": event.interval.value,
        "open": _price(event.open),
        "high": _price(event.high),
        "low": _price(event.low),
        "close": _price(event.close),
        "volume": str(event.volume.value),
        "vwap": _price(event.vwap),
        "trade_count": event.trade_count,
    },
    lambda data: Bar(
        symbol=_parse_symbol(data["symbol"]),
        ts_event=data["ts_event"],
        ts_init=data["ts_init"],
        seq=data["seq"],
        interval=BarInterval(data["interval"]),
        open=Price(Decimal(data["open"])),
        high=Price(Decimal(data["high"])),
        low=Price(Decimal(data["low"])),
        close=Price(Decimal(data["close"])),
        volume=Quantity(Decimal(data["volume"])),
        vwap=_parse_price(data["vwap"]),
        trade_count=data["trade_count"],
    ),
)

codec.register(
    "quote",
    Quote,
    lambda event: {
        **_base(event),
        "symbol": _symbol(event.symbol),
        "bid_price": _price(event.bid_price),
        "bid_size": str(event.bid_size.value),
        "ask_price": _price(event.ask_price),
        "ask_size": str(event.ask_size.value),
    },
    lambda data: Quote(
        symbol=_parse_symbol(data["symbol"]),
        ts_event=data["ts_event"],
        ts_init=data["ts_init"],
        seq=data["seq"],
        bid_price=Price(Decimal(data["bid_price"])),
        bid_size=Quantity(Decimal(data["bid_size"])),
        ask_price=Price(Decimal(data["ask_price"])),
        ask_size=Quantity(Decimal(data["ask_size"])),
    ),
)

codec.register(
    "trade",
    TickTrade,
    lambda event: {
        **_base(event),
        "symbol": _symbol(event.symbol),
        "price": _price(event.price),
        "size": str(event.size.value),
        "aggressor": None if event.aggressor is None else event.aggressor.value,
    },
    lambda data: TickTrade(
        symbol=_parse_symbol(data["symbol"]),
        ts_event=data["ts_event"],
        ts_init=data["ts_init"],
        seq=data["seq"],
        price=Price(Decimal(data["price"])),
        size=Quantity(Decimal(data["size"])),
        aggressor=None if data["aggressor"] is None else Side(data["aggressor"]),
    ),
)

# ── Session and halts ────────────────────────────────────────

codec.register(
    "session",
    SessionBoundary,
    lambda event: {**_base(event), "venue": event.venue.value, "session": event.session.value},
    lambda data: SessionBoundary(
        venue=Venue(data["venue"]),
        ts_event=data["ts_event"],
        ts_init=data["ts_init"],
        seq=data["seq"],
        session=MarketSession(data["session"]),
    ),
)

codec.register(
    "halt",
    TradingHalt,
    lambda event: {**_base(event), "symbol": _symbol(event.symbol), "reason": event.reason.value},
    lambda data: TradingHalt(
        symbol=_parse_symbol(data["symbol"]),
        ts_event=data["ts_event"],
        ts_init=data["ts_init"],
        seq=data["seq"],
        reason=HaltReason(data["reason"]),
    ),
)

codec.register(
    "resumed",
    TradingResumed,
    lambda event: {
        **_base(event),
        "symbol": _symbol(event.symbol),
        "auction_price": _price(event.auction_price),
    },
    lambda data: TradingResumed(
        symbol=_parse_symbol(data["symbol"]),
        ts_event=data["ts_event"],
        ts_init=data["ts_init"],
        seq=data["seq"],
        auction_price=_parse_price(data["auction_price"]),
    ),
)

# ── System output ────────────────────────────────────────────

codec.register(
    "intent",
    Intent,
    lambda event: {
        **_base(event),
        "symbol": _symbol(event.symbol),
        "side": event.side.value,
        "entry": event.entry.value,
        "entry_price": _price(event.entry_price),
        "invalidation": _price(event.invalidation),
        "target_r": str(event.target_r),
        "horizon_ns": event.horizon_ns,
        "strategy": event.strategy,
        "strategy_version": event.strategy_version,
        "rationale": event.rationale,
    },
    lambda data: Intent(
        symbol=_parse_symbol(data["symbol"]),
        ts_event=data["ts_event"],
        ts_init=data["ts_init"],
        seq=data["seq"],
        side=Side(data["side"]),
        entry=EntryTrigger(data["entry"]),
        entry_price=_parse_price(data["entry_price"]),
        invalidation=Price(Decimal(data["invalidation"])),
        target_r=Decimal(data["target_r"]),
        horizon_ns=data["horizon_ns"],
        strategy=data["strategy"],
        strategy_version=data["strategy_version"],
        rationale=data["rationale"],
    ),
)

codec.register(
    "order",
    Order,
    lambda event: {
        **_base(event),
        "id": event.id.value,
        "intent_id": event.intent_id.value,
        "symbol": _symbol(event.symbol),
        "side": event.side.value,
        "quantity": str(event.quantity.value),
        "order_type": event.order_type.value,
        "limit_price": _price(event.limit_price),
        "stop_price": _price(event.stop_price),
        "time_in_force": event.time_in_force.value,
        "config_hash": event.config_hash,
    },
    lambda data: Order(
        id=OrderId(data["id"]),
        intent_id=IntentId(data["intent_id"]),
        symbol=_parse_symbol(data["symbol"]),
        ts_event=data["ts_event"],
        ts_init=data["ts_init"],
        seq=data["seq"],
        side=Side(data["side"]),
        quantity=Quantity(Decimal(data["quantity"])),
        order_type=OrderType(data["order_type"]),
        limit_price=_parse_price(data["limit_price"]),
        stop_price=_parse_price(data["stop_price"]),
        time_in_force=TimeInForce(data["time_in_force"]),
        config_hash=data["config_hash"],
    ),
)

codec.register(
    "fill",
    Fill,
    lambda event: {
        **_base(event),
        "id": event.id.value,
        "order_id": event.order_id.value,
        "symbol": _symbol(event.symbol),
        "side": event.side.value,
        "price": _price(event.price),
        "quantity": str(event.quantity.value),
        "commission": _money(event.commission),
        "liquidity": event.liquidity.value,
        "reference_price": _price(event.reference_price),
        "broker_exec_id": event.broker_exec_id,
    },
    lambda data: Fill(
        id=FillId(data["id"]),
        order_id=OrderId(data["order_id"]),
        symbol=_parse_symbol(data["symbol"]),
        ts_event=data["ts_event"],
        ts_init=data["ts_init"],
        seq=data["seq"],
        side=Side(data["side"]),
        price=Price(Decimal(data["price"])),
        quantity=Quantity(Decimal(data["quantity"])),
        commission=_parse_money(data["commission"]),
        liquidity=LiquidityFlag(data["liquidity"]),
        reference_price=_parse_price(data["reference_price"]),
        broker_exec_id=data["broker_exec_id"],
    ),
)
