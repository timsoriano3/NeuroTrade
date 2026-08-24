"""Tests for event serialisation.

Every test here is ultimately the same test: does an event survive the log
unchanged? Gate G1 needs replayed events to be *equal* to the originals, not
similar, because a strategy fed a price that differs in the eighth decimal makes
a different decision and the replay proves nothing.
"""

from __future__ import annotations

import inspect
import json
from decimal import Decimal

import pytest

from neurotrade.core import codec as codec_module
from neurotrade.core.codec import (
    CODEC_VERSION,
    UnknownEventType,
    codec,
)
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

AAPL = Symbol("AAPL", Venue.NASDAQ)
BRK_B = Symbol("BRK.B", Venue.NYSE)  # a real ticker containing a dot
NOW = 1_773_495_000_000_000_000

INTENT_ID = IntentId.derive(
    strategy="orb", strategy_version="1.0.0", symbol=AAPL, ts_event=NOW, seq=0
)
ORDER_ID = OrderId.derive(intent_id=INTENT_ID, ts_event=NOW)


def a_bar(**overrides: object) -> Bar:
    defaults: dict[str, object] = {
        "symbol": AAPL,
        "ts_event": NOW,
        "ts_init": NOW + 5,
        "seq": 2,
        "interval": BarInterval.MIN_1,
        "open": Price("100.12345678"),
        "high": Price("101.87654321"),
        "low": Price("99.00000001"),
        "close": Price("100.5"),
        "volume": Quantity("1234567.89"),
    }
    return Bar(**{**defaults, **overrides})  # type: ignore[arg-type]


EVERY_EVENT: list[Event] = [
    a_bar(),
    a_bar(vwap=Price("100.4"), trade_count=812),
    Quote(
        symbol=AAPL,
        ts_event=NOW,
        ts_init=NOW,
        bid_price=Price("100.00"),
        bid_size=Quantity(500),
        ask_price=Price("100.02"),
        ask_size=Quantity(300),
    ),
    TickTrade(symbol=AAPL, ts_event=NOW, ts_init=NOW, price=Price("100.5"), size=Quantity(200)),
    TickTrade(
        symbol=AAPL,
        ts_event=NOW,
        ts_init=NOW,
        price=Price("100.5"),
        size=Quantity(200),
        aggressor=Side.BUY,
    ),
    SessionBoundary(venue=Venue.NASDAQ, ts_event=NOW, ts_init=NOW, session=MarketSession.REGULAR),
    TradingHalt(symbol=AAPL, ts_event=NOW, ts_init=NOW, reason=HaltReason.LULD),
    TradingResumed(symbol=AAPL, ts_event=NOW, ts_init=NOW, auction_price=Price("187.40")),
    TradingResumed(symbol=AAPL, ts_event=NOW, ts_init=NOW),
    Intent(
        symbol=AAPL,
        ts_event=NOW,
        ts_init=NOW,
        side=Side.BUY,
        entry=EntryTrigger.LIMIT,
        entry_price=Price("100.00"),
        invalidation=Price("99.00"),
        target_r=Decimal("2.5"),
        horizon_ns=3_600_000_000_000,
        strategy="orb_stocks_in_play",
        strategy_version="1.0.0",
        rationale="5-minute opening range break on 3x relative volume",
    ),
    Intent(
        symbol=AAPL,
        ts_event=NOW,
        ts_init=NOW,
        side=Side.SELL,
        entry=EntryTrigger.MARKET,
        entry_price=None,
        invalidation=Price("101.00"),
        target_r=Decimal(2),
        horizon_ns=1_000,
        strategy="fade",
        strategy_version="2.0.1",
        rationale="failed breakout",
    ),
    Order(
        id=ORDER_ID,
        intent_id=INTENT_ID,
        symbol=AAPL,
        ts_event=NOW,
        ts_init=NOW,
        side=Side.BUY,
        quantity=Quantity(100),
        order_type=OrderType.LIMIT,
        limit_price=Price("100.05"),
        config_hash="cfg_d842e94e473f9869",
    ),
    Order(
        id=ORDER_ID,
        intent_id=INTENT_ID,
        symbol=AAPL,
        ts_event=NOW,
        ts_init=NOW,
        side=Side.SELL,
        quantity=Quantity("0.5"),
        order_type=OrderType.STOP_LIMIT,
        limit_price=Price("99.9"),
        stop_price=Price("100.0"),
        time_in_force=TimeInForce.IOC,
    ),
    Fill(
        id=FillId.derive(order_id=ORDER_ID, broker_exec_id="e.1"),
        order_id=ORDER_ID,
        symbol=AAPL,
        ts_event=NOW,
        ts_init=NOW,
        side=Side.BUY,
        price=Price("100.05"),
        quantity=Quantity(100),
        commission=Money("1.00", Currency.USD),
        liquidity=LiquidityFlag.TAKER,
        reference_price=Price("100.00"),
        broker_exec_id="e.1",
    ),
]


# ── The round trip ───────────────────────────────────────────


@pytest.mark.parametrize("event", EVERY_EVENT, ids=lambda e: type(e).__name__)
def test_every_event_survives_a_round_trip(event: Event) -> None:
    assert codec.loads(codec.dumps(event)) == event


def test_decimal_precision_survives() -> None:
    """A JSON number is a float; 100.12345678 does not survive as one."""
    restored = codec.loads(codec.dumps(a_bar()))
    assert isinstance(restored, Bar)
    assert restored.open.value == Decimal("100.12345678")
    assert restored.low.value == Decimal("99.00000001")


def test_prices_are_written_as_strings_not_numbers() -> None:
    """The encoding choice that makes exactness possible."""
    payload = json.loads(codec.dumps(a_bar()))
    assert payload["open"] == "100.12345678"
    assert isinstance(payload["open"], str)


def test_nanosecond_timestamps_survive() -> None:
    event = a_bar(ts_event=NOW + 123, ts_init=NOW + 456)
    restored = codec.loads(codec.dumps(event))
    assert (restored.ts_event, restored.ts_init) == (NOW + 123, NOW + 456)


def test_a_ticker_containing_a_dot_survives() -> None:
    """BRK.B encodes as BRK.B.NYSE; splitting on the first dot would break it."""
    restored = codec.loads(codec.dumps(a_bar(symbol=BRK_B)))
    assert isinstance(restored, Bar)
    assert restored.symbol == BRK_B
    assert restored.symbol.ticker == "BRK.B"


def test_optional_fields_stay_none() -> None:
    """None must not become zero or an empty string on the way through."""
    restored = codec.loads(codec.dumps(a_bar()))
    assert isinstance(restored, Bar)
    assert restored.vwap is None
    assert restored.trade_count is None


# ── Determinism ──────────────────────────────────────────────


def test_encoding_is_stable_across_calls() -> None:
    """Sorted keys mean the bytes depend only on content, so two runs producing
    the same events produce the same file."""
    assert codec.dumps(a_bar()) == codec.dumps(a_bar())


def test_encoding_is_one_line_per_event() -> None:
    """The log is newline-delimited; an embedded newline would corrupt it."""
    assert "\n" not in codec.dumps(EVERY_EVENT[-1])


def test_keys_are_sorted() -> None:
    payload = codec.dumps(a_bar())
    keys = list(json.loads(payload))
    assert keys == sorted(keys)


# ── Completeness ─────────────────────────────────────────────


def _event_subclasses() -> set[type[Event]]:
    """Every concrete Event subclass reachable from the domain modules."""
    from neurotrade.core import events as events_module
    from neurotrade.core import intent as intent_module
    from neurotrade.core import orders as orders_module

    found: set[type[Event]] = set()
    for module in (events_module, intent_module, orders_module):
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, Event) and obj is not Event:
                found.add(obj)
    # MarketEvent and VenueEvent are scope bases, never instantiated directly.
    return {cls for cls in found if cls.__name__ not in {"MarketEvent", "VenueEvent"}}


def test_every_event_type_has_a_codec() -> None:
    """A new event type must not reach the log as an unserialisable object.

    Without this, adding an event class and forgetting the codec fails at
    09:31 on a Monday rather than in CI.
    """
    missing = _event_subclasses() - set(codec.registered)
    assert not missing, f"no codec for: {sorted(cls.__name__ for cls in missing)}"


def test_the_sample_set_covers_every_registered_type() -> None:
    """Keeps the round-trip parametrisation honest as types are added."""
    covered = {type(event) for event in EVERY_EVENT}
    assert covered == set(codec.registered)


# ── Failure modes ────────────────────────────────────────────


def test_an_unregistered_event_is_refused() -> None:
    class Rogue(Event):
        pass

    with pytest.raises(UnknownEventType, match="has no codec"):
        codec.encode(Rogue(ts_event=1, ts_init=1))


def test_an_unknown_tag_is_refused() -> None:
    """Skipping unknown records would produce a replay that looks successful
    and is missing events."""
    with pytest.raises(UnknownEventType, match="no codec for event tag"):
        codec.decode({"_t": "from_the_future", "v": CODEC_VERSION})


def test_a_different_codec_version_is_refused() -> None:
    """Decoding on a guess is worse than refusing to start."""
    payload = json.loads(codec.dumps(a_bar()))
    payload["v"] = "0"
    with pytest.raises(UnknownEventType, match="codec version"):
        codec.decode(payload)


def test_registering_a_duplicate_tag_is_refused() -> None:
    private = codec_module.EventCodec()
    private.register("bar", Bar, lambda e: {}, lambda d: a_bar())
    with pytest.raises(ValueError, match="already registered"):
        private.register("bar", Quote, lambda e: {}, lambda d: a_bar())


def test_registering_a_type_twice_is_refused() -> None:
    private = codec_module.EventCodec()
    private.register("bar", Bar, lambda e: {}, lambda d: a_bar())
    with pytest.raises(ValueError, match="already has a codec"):
        private.register("bar2", Bar, lambda e: {}, lambda d: a_bar())


# ── Ordering is preserved through the log ────────────────────


def test_sort_keys_survive_the_round_trip() -> None:
    """Replay orders on (ts_event, seq); both must come back exactly."""
    events = [a_bar(ts_event=NOW, seq=1), a_bar(ts_event=NOW, seq=0)]
    restored = [codec.loads(codec.dumps(event)) for event in events]
    assert [event.sort_key for event in restored] == [(NOW, 1), (NOW, 0)]
