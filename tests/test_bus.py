"""Tests for the event bus.

Dispatch order is the property that matters. Everything else here is
bookkeeping; if two runs deliver the same events to the same handlers in
different orders, gate G1 is unreachable no matter how good the replay engine is.
"""

from __future__ import annotations

import pytest

from neurotrade.bus import EventBus, HandlerFailed
from neurotrade.core.events import (
    Bar,
    BarInterval,
    Event,
    MarketEvent,
    MarketSession,
    Quote,
    SessionBoundary,
    VenueEvent,
)
from neurotrade.core.types import Price, Quantity, Symbol, Venue

AAPL = Symbol("AAPL", Venue.NASDAQ)
NOW = 1_773_495_000_000_000_000
MINUTE = 60_000_000_000


def bar(index: int = 0, seq: int = 0) -> Bar:
    ts = NOW + index * MINUTE
    return Bar(
        symbol=AAPL,
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


def quote() -> Quote:
    return Quote(
        symbol=AAPL,
        ts_event=NOW,
        ts_init=NOW,
        bid_price=Price("100"),
        bid_size=Quantity(100),
        ask_price=Price("100.02"),
        ask_size=Quantity(100),
    )


def boundary() -> SessionBoundary:
    return SessionBoundary(
        venue=Venue.NASDAQ, ts_event=NOW, ts_init=NOW, session=MarketSession.REGULAR
    )


# ── Dispatch order: what G1 rests on ─────────────────────────


def test_handlers_are_called_in_subscription_order() -> None:
    calls: list[str] = []
    bus = EventBus()
    for name in ("first", "second", "third"):
        bus.subscribe(Event, lambda _event, name=name: calls.append(name))  # type: ignore[misc]
    bus.publish(bar())
    assert calls == ["first", "second", "third"]


def test_dispatch_order_is_identical_across_runs() -> None:
    """Two identical setups must deliver identically — the replay property."""

    def run() -> list[str]:
        calls: list[str] = []
        bus = EventBus()
        bus.subscribe(Bar, lambda _e: calls.append("bar"))
        bus.subscribe(Event, lambda _e: calls.append("all"))
        bus.subscribe(MarketEvent, lambda _e: calls.append("market"))
        bus.publish_all([bar(0), quote(), boundary()])
        return calls

    assert run() == run()


def test_publish_all_preserves_the_order_given() -> None:
    """The bus does not sort. Ordering belongs to whoever produced the sequence.

    A bus that re-sorted would hide a source delivering events out of order,
    which is exactly the fault a replay is meant to surface.
    """
    seen: list[int] = []
    bus = EventBus()
    bus.subscribe(Bar, lambda event: seen.append(event.ts_event))
    bus.publish_all([bar(2), bar(0), bar(1)])
    assert seen == [NOW + 2 * MINUTE, NOW, NOW + MINUTE]


# ── Type matching ────────────────────────────────────────────


def test_a_handler_only_sees_its_type() -> None:
    bars: list[Event] = []
    bus = EventBus()
    bus.subscribe(Bar, bars.append)
    bus.publish_all([bar(), quote(), boundary()])
    assert len(bars) == 1


def test_subscribing_to_a_base_catches_subclasses() -> None:
    """How `subscribe(Event, store.append)` records everything."""
    everything: list[Event] = []
    market_only: list[Event] = []
    venue_only: list[Event] = []

    bus = EventBus()
    bus.subscribe(Event, everything.append)
    bus.subscribe(MarketEvent, market_only.append)
    bus.subscribe(VenueEvent, venue_only.append)
    bus.publish_all([bar(), quote(), boundary()])

    assert len(everything) == 3
    assert len(market_only) == 2  # bar and quote
    assert len(venue_only) == 1  # the session boundary


def test_one_handler_may_subscribe_to_several_types() -> None:
    seen: list[Event] = []
    bus = EventBus()
    bus.subscribe(Bar, seen.append)
    bus.subscribe(Quote, seen.append)
    bus.publish_all([bar(), quote(), boundary()])
    assert len(seen) == 2


def test_an_event_with_no_subscriber_is_not_an_error() -> None:
    """Most events have no listener most of the time."""
    bus = EventBus()
    bus.publish(bar())
    assert bus.published == 1


# ── Recording is just a subscriber ───────────────────────────


def test_the_session_log_needs_no_special_support(tmp_path: object) -> None:
    """`bus.subscribe(Event, store.append)` is the whole recording mechanism."""
    from pathlib import Path

    from neurotrade.adapters.storage.event_store import EventStore

    assert isinstance(tmp_path, Path)
    log = tmp_path / "session.jsonl"
    with EventStore(log) as store:
        bus = EventBus()
        bus.subscribe(Event, store.append)
        bus.publish_all([bar(0), quote(), boundary()])

    assert len(list(EventStore(log).stream(0, NOW + 10 * MINUTE))) == 3


# ── Failure handling ─────────────────────────────────────────


def test_a_failing_handler_stops_dispatch() -> None:
    """Half-delivering an event leaves components disagreeing about what happened."""
    after: list[str] = []

    def explode(_event: Event) -> None:
        raise ValueError("strategy blew up")

    bus = EventBus()
    bus.subscribe(Event, explode)
    bus.subscribe(Event, lambda _e: after.append("ran"))

    with pytest.raises(HandlerFailed):
        bus.publish(bar())
    assert after == []


def test_the_failure_names_the_handler_and_the_event() -> None:
    """Otherwise the traceback says only that something in dispatch failed."""

    def rejects_everything(_event: Event) -> None:
        raise ValueError("nope")

    bus = EventBus()
    bus.subscribe(Event, rejects_everything)
    with pytest.raises(HandlerFailed, match="rejects_everything failed on Bar"):
        bus.publish(bar(seq=7))


def test_the_original_exception_is_chained() -> None:
    """The cause must stay reachable, or debugging starts from nothing."""

    def explode(_event: Event) -> None:
        raise ZeroDivisionError("division by zero")

    bus = EventBus()
    bus.subscribe(Event, explode)
    with pytest.raises(HandlerFailed) as caught:
        bus.publish(bar())
    assert isinstance(caught.value.__cause__, ZeroDivisionError)


def test_a_silent_handler_failure_would_be_invisible() -> None:
    """States the reason for raising: a swallowed error looks like no signal."""
    fired: list[Event] = []

    def strategy(event: Event) -> None:
        fired.append(event)
        raise RuntimeError("bad feature")

    bus = EventBus()
    bus.subscribe(Bar, strategy)
    with pytest.raises(HandlerFailed):
        bus.publish(bar())
    assert len(fired) == 1  # it ran, then failed — not "produced no signal"


# ── Introspection ────────────────────────────────────────────


def test_published_counts_every_event() -> None:
    bus = EventBus()
    bus.publish_all([bar(0), bar(1), quote()])
    assert bus.published == 3


def test_subscriptions_are_reported_in_call_order() -> None:
    """Two replays that dispatch differently usually differ here first."""

    def handle_bar(_event: Event) -> None: ...

    def handle_all(_event: Event) -> None: ...

    bus = EventBus()
    bus.subscribe(Bar, handle_bar)
    bus.subscribe(Event, handle_all)
    assert [name for _type, name in bus.subscriptions] == [
        "test_subscriptions_are_reported_in_call_order.<locals>.handle_bar",
        "test_subscriptions_are_reported_in_call_order.<locals>.handle_all",
    ]
    assert [event_type for event_type, _name in bus.subscriptions] == ["Bar", "Event"]


def test_length_is_the_subscriber_count() -> None:
    bus = EventBus()
    bus.subscribe(Bar, lambda _e: None)
    bus.subscribe(Event, lambda _e: None)
    assert len(bus) == 2


def test_two_buses_are_independent() -> None:
    """A test or a shadow run must not leak subscribers into the live bus."""
    seen: list[Event] = []
    first, second = EventBus(), EventBus()
    first.subscribe(Event, seen.append)
    second.publish(bar())
    assert seen == []
