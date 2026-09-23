"""The drive loop: turning an ordered stream of events into a run.

Both things the lab runs are the same loop over a different source. A **replay**
reads a recorded session back out of the event log; a **backtest** reads bars
out of the corpus. Neither cares where its events came from — each one advances
the clock, publishes, and folds the result into a digest.

Keeping that loop in one place is not tidiness, it is the one-implementation
invariant (§3.6) applied to the lab itself. If replay and backtest each had
their own loop they could drift in exactly the way that matters most: a subtle
difference in when the clock moves, or in what gets digested, would make a
backtest result unreproducible by a replay of the same session — and that
divergence is the project's primary failure mode.

**The clock leads each event.** It is moved to `ts_event` *before* publishing,
so anything reading the clock during dispatch sees the moment being modelled
rather than the moment the run is happening. `SimClock` refuses to move
backwards, so a source that yields events out of order fails loudly here rather
than producing quietly wrong output.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass

from neurotrade.bus import EventBus
from neurotrade.core.clock import Nanos, SimClock
from neurotrade.core.codec import CODEC_VERSION, codec
from neurotrade.core.events import Event

__all__ = ["RunDigest", "RunResult", "drive"]


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
class RunResult:
    """What one drive over a source produced.

    Deliberately holds no timing. Wall-clock duration is useful to report and
    must never enter a comparison — it varies between runs for reasons that have
    nothing to do with behaviour, and a "digest" including it would never match.
    """

    digest: str  # rolling hash over every dispatched event, in order
    events_read: int  # events taken from the source
    events_dispatched: int  # events the bus delivered, including any reactions
    first_ts: Nanos | None  # earliest ts_event seen; None if the range was empty
    last_ts: Nanos | None  # latest ts_event seen

    @property
    def is_empty(self) -> bool:
        """Whether the run found anything to do."""
        return self.events_read == 0

    @property
    def span_ns(self) -> int:
        """Simulated time covered, in nanoseconds. Zero for an empty run."""
        if self.first_ts is None or self.last_ts is None:
            return 0
        return self.last_ts - self.first_ts


def drive(
    events: Iterable[Event],
    *,
    clock: SimClock,
    bus: EventBus,
    digest: RunDigest,
) -> RunResult:
    """Publish an ordered stream of events, advancing the clock ahead of each.

    Args:
        events: The source, in ascending `ts_event` order. Consumed lazily, so a
            multi-year range never has to fit in memory.
        clock: Simulated clock, moved to each event's `ts_event` before dispatch.
        bus: Where events are published.
        digest: The running hash. Passed in rather than created here because the
            caller must subscribe it *before* any handler, so that an input is
            folded in ahead of whatever it causes.

    Returns:
        A `RunResult` whose `digest` identifies this run's behaviour.

    Raises:
        ValueError: If the source yields events out of order — `SimClock` will
            not move backwards.
        HandlerFailed: If a subscriber raises. The run stops; a partially
            dispatched session would report a digest for something that never
            fully happened.

    Example:
        >>> from neurotrade.bus import EventBus
        >>> from neurotrade.core.clock import SimClock
        >>> from neurotrade.lab.drive import RunDigest, drive
        >>> bus, digest = EventBus(), RunDigest()
        >>> bus.subscribe(Bar, digest)
        >>> result = drive([a_bar], clock=SimClock(0), bus=bus, digest=digest)
        >>> (result.events_read, result.first_ts, result.is_empty)
        (1, 1000, False)
    """
    read = 0
    first_ts: Nanos | None = None
    last_ts: Nanos | None = None

    for event in events:
        # Clock first: a handler reading the clock must see the moment being
        # modelled, not the moment the run happens to be executing.
        clock.set_time_ns(event.ts_event)
        bus.publish(event)

        read += 1
        if first_ts is None:
            first_ts = event.ts_event
        last_ts = event.ts_event

    return RunResult(
        digest=digest.hexdigest,
        events_read=read,
        events_dispatched=digest.count,
        first_ts=first_ts,
        last_ts=last_ts,
    )
