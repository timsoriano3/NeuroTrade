"""The in-process event bus.

Everything that happens goes through here: market data in, intents and orders
out. One bus serves live trading and replay, and the only difference between
them is where the events come from and which `Clock` is running. That sameness
is the point — a replay that used a different dispatch path would be testing the
replay harness rather than the system.

**Dispatch is synchronous and sequential, and that is a determinism decision.**
The obvious alternative is an async bus that fans events out with
`asyncio.gather`, which is faster and unusable here: concurrent handlers
interleave in whatever order the loop happens to schedule them, so two replays
of one session produce different orderings and gate G1 becomes unachievable.
Handlers run one at a time, in subscription order, every time.

Network work still happens asynchronously — but at the edges. The live engine
awaits the broker socket, then publishes what arrived; the broker call that
results is made after dispatch returns, not inside a handler. Keeping I/O out of
dispatch is what lets dispatch be reproducible.

**A handler that raises stops the dispatch.** Swallowing the exception would
leave a strategy silently not firing, which looks exactly like a strategy with
no signal — the most expensive kind of failure, because nothing reports it.

**Recording is just a subscriber.** Writing the session log needs no special
support: subscribe the event store to every event and the log writes itself.

    bus.subscribe(Event, store.append)
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from neurotrade.core.events import Event

__all__ = ["EventBus", "Handler", "HandlerFailed"]

type Handler = Callable[[Event], None]
"""What a subscriber looks like. Returns nothing: a handler that wanted to emit
something publishes it, so the emission is itself an event on the log rather
than a value passed privately between two components."""


class HandlerFailed(RuntimeError):
    """Raised when a subscriber raises, naming the handler and the event.

    The original exception is chained. Dispatch stops here rather than
    continuing to the remaining handlers: an event half-delivered leaves
    components disagreeing about what happened, which is worse than stopping.
    """


@dataclass(frozen=True, slots=True)
class _Subscription:
    """One registration. Ordered by registration, never re-sorted."""

    event_type: type[Event]  # matched with isinstance, so bases catch subclasses
    handler: Handler  # called with the event
    name: str  # for error messages; the function's qualified name


@dataclass(slots=True)
class EventBus:
    """Routes events to subscribers, in a fixed order.

    Example:
        >>> seen = []
        >>> bus = EventBus()
        >>> bus.subscribe(Event, seen.append)
        >>> bus.publish(Event(ts_event=1_000, ts_init=1_000))
        >>> len(seen)
        1
    """

    _subscriptions: list[_Subscription] = field(default_factory=list)
    _published: int = 0

    def subscribe(self, event_type: type[Event], handler: Handler) -> None:
        """Register a handler for an event type and its subclasses.

        Subscribing to a base type catches everything below it, so
        `subscribe(MarketEvent, ...)` receives bars, quotes and prints, and
        `subscribe(Event, ...)` receives everything. That is how the session
        recorder is wired.

        Order matters and is preserved: handlers are called in the order they
        subscribed, which makes dispatch reproducible as long as subscription is.

        Args:
            event_type: The class to match, by `isinstance`.
            handler: Called with each matching event.

        Example:
            >>> bars = []
            >>> bus = EventBus()
            >>> bus.subscribe(Bar, bars.append)
            >>> bus.publish(Event(ts_event=1, ts_init=1))     # not a Bar
            >>> len(bars)
            0
        """
        self._subscriptions.append(
            _Subscription(
                event_type=event_type,
                handler=handler,
                name=getattr(handler, "__qualname__", repr(handler)),
            )
        )

    def publish(self, event: Event) -> None:
        """Deliver one event to every matching handler, in subscription order.

        Args:
            event: The event to deliver.

        Raises:
            HandlerFailed: If a handler raises. Dispatch stops at that point;
                handlers registered after it do not see the event.

        Example:
            >>> order = []
            >>> bus = EventBus()
            >>> bus.subscribe(Event, lambda e: order.append("first"))
            >>> bus.subscribe(Event, lambda e: order.append("second"))
            >>> bus.publish(Event(ts_event=1, ts_init=1))
            >>> order
            ['first', 'second']
        """
        self._published += 1
        for subscription in self._subscriptions:
            if isinstance(event, subscription.event_type):
                try:
                    subscription.handler(event)
                except Exception as error:
                    raise HandlerFailed(
                        f"{subscription.name} failed on "
                        f"{type(event).__name__}(ts_event={event.ts_event}, seq={event.seq})"
                    ) from error

    def publish_all(self, events: Iterable[Event]) -> None:
        """Publish a sequence in the order given.

        Does **not** sort. Ordering belongs to whatever produced the sequence —
        the replay engine sorts by `(ts_event, seq)` before feeding the bus, and
        a bus that re-sorted would hide a source delivering events out of order.

        Args:
            events: Events to deliver, already in the intended order.
        """
        for event in events:
            self.publish(event)

    @property
    def published(self) -> int:
        """How many events have been dispatched. Useful in progress reporting."""
        return self._published

    @property
    def subscriptions(self) -> tuple[tuple[str, str], ...]:
        """Registered handlers as `(event type, handler name)`, in call order.

        Exposed so a run can record what was listening. Two replays that
        dispatch differently usually differ here first.

        Example:
            >>> bus = EventBus()
            >>> bus.subscribe(Bar, print)
            >>> bus.subscriptions
            (('Bar', 'print'),)
        """
        return tuple(
            (subscription.event_type.__name__, subscription.name)
            for subscription in self._subscriptions
        )

    def __len__(self) -> int:
        return len(self._subscriptions)

    def __repr__(self) -> str:
        return f"EventBus({len(self._subscriptions)} subscribers, {self._published} published)"
