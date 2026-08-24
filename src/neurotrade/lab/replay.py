"""Replaying a recorded session, and proving the replay was faithful.

This is gate G1: feed a session's recorded events back through the same bus the
live system uses, under a simulated clock, and get the identical result — every
time, on every machine. A replay that could not be trusted to reproduce would
make every diagnosis of a live/backtest divergence a guess.

**The run digest is the proof.** It is a rolling hash over every event the bus
dispatched, in dispatch order. Two runs agree only if they saw the same events,
in the same order, with the same contents down to the last decimal. Comparing
digests turns "did that change alter behaviour?" into a yes/no question instead
of an afternoon of diffing logs.

Crucially the digest covers **outputs as well as inputs**. It hashes everything
published, so once strategies exist their intents enter the digest for free —
and a strategy that starts making different decisions changes the digest even
though the input data is untouched. That is the whole point.

**The clock leads each event.** Before an event is dispatched, the `SimClock` is
moved to that event's `ts_event`, so anything reading the clock during dispatch
sees the moment being modelled rather than the moment the replay is running.
Because `SimClock` refuses to move backwards, a store that yielded events out of
order fails here rather than producing quietly wrong output.

**Nothing here reaches an adapter.** The engine takes an `EventStorePort`, so a
replay can run against a file, a fixture, or an in-memory list without changing.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from neurotrade.bus import EventBus
from neurotrade.core.clock import Nanos, SimClock
from neurotrade.core.codec import CODEC_VERSION, codec
from neurotrade.core.events import Event
from neurotrade.core.ports import EventStorePort

__all__ = ["ReplayEngine", "ReplayResult", "RunDigest"]


class RunDigest:
    """A rolling hash over every event a bus dispatches.

    Used as a bus subscriber, so it observes exactly what the handlers observed,
    in the order they observed it.

    BLAKE2b rather than Python's `hash()` for the same reason identifiers use it:
    `hash()` is salted per process, so a digest built with it would differ
    between two runs of the same program and prove nothing.

    Example:
        >>> digest = RunDigest()
        >>> digest(a_bar)
        >>> len(digest.hexdigest)
        32
    """

    __slots__ = ("_count", "_hash")

    def __init__(self) -> None:
        self._hash = hashlib.blake2b(digest_size=16)
        # Seeded with the codec version so that two digests are only ever
        # compared when the encoding behind them is the same. A codec change
        # alters every digest, which is correct — the bytes really did change.
        self._hash.update(f"codec={CODEC_VERSION}\n".encode())
        self._count = 0

    def __call__(self, event: Event) -> None:
        """Fold one event into the digest. Signature matches `Handler`."""
        self._hash.update(codec.dumps(event).encode("utf-8"))
        self._hash.update(b"\n")
        self._count += 1

    @property
    def hexdigest(self) -> str:
        """The digest so far, as hex.

        Example:
            >>> a, b = RunDigest(), RunDigest()
            >>> a(a_bar); b(a_bar)
            >>> a.hexdigest == b.hexdigest
            True
        """
        return self._hash.hexdigest()

    @property
    def count(self) -> int:
        """How many events have been folded in."""
        return self._count

    def __repr__(self) -> str:
        return f"RunDigest({self._count} events, {self.hexdigest[:12]}…)"


@dataclass(frozen=True, slots=True)
class ReplayResult:
    """What a replay produced.

    Deliberately holds no timing. Wall-clock duration is useful to report and
    must never enter a comparison — it varies between runs for reasons that have
    nothing to do with behaviour, and a "digest" including it would never match.
    """

    digest: str  # rolling hash over every dispatched event, in order
    events_read: int  # events taken from the store
    events_dispatched: int  # events the bus delivered, including any reactions
    first_ts: Nanos | None  # earliest ts_event seen; None if the range was empty
    last_ts: Nanos | None  # latest ts_event seen

    @property
    def is_empty(self) -> bool:
        """Whether the replay found anything to do."""
        return self.events_read == 0

    @property
    def span_ns(self) -> int:
        """Simulated time covered, in nanoseconds. Zero for an empty replay."""
        if self.first_ts is None or self.last_ts is None:
            return 0
        return self.last_ts - self.first_ts


class ReplayEngine:
    """Drives a recorded session through the bus under a simulated clock.

    The engine owns its bus and its digest, and subscribes the digest before
    anything else. That ordering is deliberate: the digest sees an input event
    before the handlers that react to it, so the digest reads in causal order —
    bar, then the intent the bar caused.

    Example:
        >>> from neurotrade.core.clock import SimClock
        >>> class EmptyLog:
        ...     def append(self, event): ...
        ...     def stream(self, start, end): return iter(())
        >>> engine = ReplayEngine(EmptyLog(), SimClock(0))
        >>> engine.run(0, 1_000).is_empty
        True
    """

    __slots__ = ("_bus", "_clock", "_digest", "_store")

    def __init__(
        self,
        store: EventStorePort,
        clock: SimClock,
        bus: EventBus | None = None,
    ) -> None:
        """Prepare a replay.

        Args:
            store: Where the recorded session is read from. A port, so this can
                be a file-backed log, a fixture, or an in-memory list.
            clock: The simulated clock, positioned at or before the first event.
                Its starting value seeds anything derived from run start time.
            bus: An existing bus to dispatch on. A fresh one is created when
                omitted, which is the usual case — sharing a bus between two
                replays would let one run's subscribers see the other's events.
        """
        self._store = store
        self._clock = clock
        self._bus = bus if bus is not None else EventBus()
        self._digest = RunDigest()
        # First subscriber, so inputs are digested before reactions to them.
        self._bus.subscribe(Event, self._digest)

    @property
    def bus(self) -> EventBus:
        """The bus this replay dispatches on. Subscribe strategies here."""
        return self._bus

    @property
    def clock(self) -> SimClock:
        """The simulated clock. Advanced by `run`, never by a handler."""
        return self._clock

    def run(self, start: Nanos, end: Nanos) -> ReplayResult:
        """Replay a half-open time range.

        Args:
            start: Inclusive lower bound on `ts_event`.
            end: Exclusive upper bound on `ts_event`.

        Returns:
            A `ReplayResult` whose `digest` identifies this run's behaviour.

        Raises:
            ValueError: If the store yields events out of order. `SimClock`
                refuses to move backwards, so a mis-ordered source fails here
                rather than silently producing a different session.
            HandlerFailed: If a subscriber raises. The replay stops; a partially
                dispatched session would report a digest for something that
                never fully happened.

        Example:
            >>> from neurotrade.core.clock import SimClock
            >>> class OneBar:
            ...     def append(self, event): ...
            ...     def stream(self, start, end):
            ...         return iter([a_bar])
            >>> result = ReplayEngine(OneBar(), SimClock(0)).run(0, 10_000)
            >>> (result.events_read, result.first_ts)
            (1, 1000)
        """
        read = 0
        first_ts: Nanos | None = None
        last_ts: Nanos | None = None

        for event in self._store.stream(start, end):
            # Clock first: a handler reading the clock must see the moment being
            # modelled, not the moment the replay happens to be running.
            self._clock.set_time_ns(event.ts_event)
            self._bus.publish(event)

            read += 1
            if first_ts is None:
                first_ts = event.ts_event
            last_ts = event.ts_event

        return ReplayResult(
            digest=self._digest.hexdigest,
            events_read=read,
            events_dispatched=self._digest.count,
            first_ts=first_ts,
            last_ts=last_ts,
        )

    def __repr__(self) -> str:
        return f"ReplayEngine({self._store!r})"
