"""Replaying a recorded session, and proving the replay was faithful.

This is gate G1: feed a session's recorded events back through the same bus the
live system uses, under a simulated clock, and get the identical result — every
time, on every machine. A replay that could not be trusted to reproduce would
make every diagnosis of a live/backtest divergence a guess.

**The run digest is the proof.** Two runs agree only if they saw the same
events, in the same order, with the same contents down to the last decimal.
Comparing digests turns "did that change alter behaviour?" into a yes/no
question instead of an afternoon of diffing logs. It covers **outputs as well as
inputs**, so once strategies exist their intents enter the digest for free — and
a strategy that starts making different decisions changes the digest even though
the input data is untouched. That is the whole point.

**The loop itself lives in `drive.py`**, shared with `BacktestEngine`. What is
left here is only what makes a replay a replay: its source is the recorded event
log rather than the corpus.

**Nothing here reaches an adapter.** The engine takes an `EventStorePort`, so a
replay can run against a file, a fixture, or an in-memory list without changing.
"""

from __future__ import annotations

from neurotrade.bus import EventBus
from neurotrade.core.clock import Nanos, SimClock
from neurotrade.core.events import Event
from neurotrade.core.ports import EventStorePort
from neurotrade.lab.drive import RunDigest, RunResult, drive

__all__ = ["ReplayEngine", "ReplayResult", "RunDigest"]

#: What a replay produced. The same record a backtest produces, because a replay
#: and a backtest differ only in where their events came from.
ReplayResult = RunResult


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
        return drive(
            self._store.stream(start, end),
            clock=self._clock,
            bus=self._bus,
            digest=self._digest,
        )

    def __repr__(self) -> str:
        return f"ReplayEngine({self._store!r})"
