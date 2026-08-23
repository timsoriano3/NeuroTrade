"""The append-only event log.

§4.1 makes every market event, signal, intent, order and fill an append-only
record, so that any session replays bit-for-bit. This is that log.

**One file per session.** The store is deliberately dumb about naming: it is
handed a path and appends to it. Deciding that a log is `2026-03-14.jsonl` or
`runs/run_7c7391c16e41c955.jsonl` is a policy question belonging to whatever
drives the session, and baking it in here would make the store hard to reuse for
the shadow environment (§10.1), which runs the same code against a different
naming scheme.

**Newline-delimited JSON, not Parquet.** The corpus is columnar because it is
read in wide date ranges and written once; this log is the opposite — written
one record at a time as events arrive, read start to finish, and heterogeneous.
Appending a line is a syscall; appending to Parquet is a read-modify-write of a
whole file.

**Line-buffered by default.** Each append reaches the operating system before
the call returns, so a crash loses nothing already accepted. §6.3 requires a
trade to be reconstructable months later, which is not true of records still
sitting in a process buffer when it dies. Backtests that write millions of
events and can simply be re-run may opt out.

**Reads sort before yielding.** The append order is already `(ts_event, seq)`
when the writer behaves, but "when the writer behaves" is not a guarantee the
replay engine can rest on. Sorting costs a session's worth of memory — a few
hundred thousand events — and buys the ordering contract the port promises.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from pathlib import Path
from types import TracebackType
from typing import IO, Self

from neurotrade.adapters.storage.event_codec import codec
from neurotrade.core.clock import Nanos
from neurotrade.core.events import Event

__all__ = ["CorruptEventLog", "EventStore"]


class CorruptEventLog(ValueError):
    """Raised when a line in the log cannot be decoded.

    Names the file and line so the damage can be inspected. Not recovered from
    automatically: a log with an unreadable record in the middle is missing
    events, and a replay that skipped them would report success while having
    simulated a different session.
    """


class EventStore:
    """An append-only log of events, on one file.

    Satisfies `EventStorePort` structurally.

    Example:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as directory:
        ...     store = EventStore(Path(directory) / "session.jsonl")
        ...     store.append(Bar(symbol=AAPL, ts_event=1_000, ts_init=1_000,
        ...                      interval=BarInterval.MIN_1, open=Price("100"),
        ...                      high=Price("101"), low=Price("99"),
        ...                      close=Price("100.5"), volume=Quantity(10)))
        ...     store.close()
        ...     [event.ts_event for event in store.stream(0, 2_000)]
        [1000]
    """

    __slots__ = ("_buffered", "_handle", "_path")

    def __init__(self, path: Path, *, buffered: bool = False) -> None:
        """Open a log for appending.

        Args:
            path: The log file. Parent directories are created. Opening for
                append never truncates, so re-opening an existing log continues
                it rather than destroying it.
            buffered: Let the OS decide when to write. Faster for backtests,
                which can be re-run; unsafe for live trading, where a record not
                yet on disk when the process dies is a trade nobody can
                reconstruct.
        """
        self._path = path
        self._buffered = buffered
        self._handle: IO[str] | None = None

    @property
    def path(self) -> Path:
        """Where this log is written."""
        return self._path

    # ── Writing ──────────────────────────────────────────────

    def append(self, event: Event) -> None:
        """Record one event.

        Append-only: there is no update and no delete. An event later found to
        be wrong is corrected by appending a correction, never by editing
        history — otherwise "what did the system know at 09:47" stops having a
        single answer.

        Args:
            event: Any event with a registered codec.

        Raises:
            UnknownEventType: If the event type has no codec.

        Example:
            >>> import tempfile
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     with EventStore(Path(directory) / "s.jsonl") as store:
            ...         store.append(Bar(symbol=AAPL, ts_event=1, ts_init=1,
            ...                          interval=BarInterval.MIN_1, open=Price("1"),
            ...                          high=Price("1"), low=Price("1"),
            ...                          close=Price("1"), volume=Quantity(1)))
            ...     len(list(EventStore(Path(directory) / "s.jsonl").stream(0, 10)))
            1
        """
        if self._handle is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            # buffering=1 is line buffering: the write reaches the OS on the
            # newline, so nothing accepted is lost if the process dies.
            self._handle = self._path.open(
                "a", encoding="utf-8", buffering=-1 if self._buffered else 1
            )
        self._handle.write(codec.dumps(event) + "\n")

    def flush(self) -> None:
        """Push buffered records to the operating system."""
        if self._handle is not None:
            self._handle.flush()

    def close(self) -> None:
        """Close the log. Safe to call more than once."""
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    # ── Reading ──────────────────────────────────────────────

    def stream(self, start: Nanos, end: Nanos) -> Iterator[Event]:
        """Replay events over a half-open range, in their original order.

        Args:
            start: Inclusive lower bound on `ts_event`.
            end: Exclusive upper bound on `ts_event`.

        Returns:
            Events ordered by `(ts_event, seq)`. Empty when the log does not
            exist — a session that has not run yet is an ordinary state, not an
            error.

        Raises:
            CorruptEventLog: If a line cannot be decoded. Not skipped: a replay
                missing events would report success having simulated a
                different session.

        Example:
            >>> import tempfile
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     list(EventStore(Path(directory) / "absent.jsonl").stream(0, 1))
            []
        """
        self.flush()
        if not self._path.exists():
            return iter(())

        events: list[Event] = []
        with self._path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    event = codec.loads(stripped)
                except Exception as error:
                    raise CorruptEventLog(
                        f"{self._path}:{number} could not be decoded: {error}"
                    ) from error
                if start <= event.ts_event < end:
                    events.append(event)

        events.sort(key=lambda event: event.sort_key)
        return iter(events)

    def __del__(self) -> None:
        """Release the file handle if the caller never closed it.

        A backstop, not the intended path — `close()` or the context manager is.
        Without it a forgotten store leaks a descriptor and emits a
        `ResourceWarning`, which the test suite treats as an error precisely so
        that leaks surface here rather than as an exhausted descriptor table
        after a few thousand sessions.
        """
        with contextlib.suppress(Exception):  # interpreter shutdown can break either
            self.close()

    def __repr__(self) -> str:
        return f"EventStore({self._path})"
