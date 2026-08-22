"""Parquet-backed implementation of `StoragePort`.

**Writes are idempotent, and that is the hard requirement.** The backfill
crawler is resumable: interrupt it mid-session and it re-fetches the range it
was working on. If a second write of the same bars appended rather than
replaced, the corpus would hold each bar twice — and a duplicated bar does not
look like corruption, it looks like double the volume. Every volume feature
computed from that day would be wrong, relative-volume rankings would put the
duplicated names at the top, and the backtest would happily trade them.

Idempotency is achieved by treating a partition as the unit of write: read what
is there, merge, drop duplicates on `(ts_event, seq)`, sort, rewrite the file.
Costlier than appending, and worth it — a corpus assembled over weeks of
interrupted crawling has to be correct by construction rather than by hoping the
crawler never crashed at a bad moment.

**`ingested_at` comes from the injected `Clock`.** Wall clock in production,
`SimClock` under replay, which keeps the no-wall-clock invariant intact and
means a replayed write produces byte-identical rows.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from neurotrade.adapters.storage.schemas import (
    BAR_SCHEMA,
    SESSION_PARTITIONING,
    Source,
    bar_to_row,
    partition_path,
    row_to_bar,
)
from neurotrade.core.clock import Clock, Nanos
from neurotrade.core.events import Bar, BarInterval
from neurotrade.core.types import Symbol

__all__ = ["ParquetStore"]


def _as_int(value: object) -> int:
    """Narrow a row value for use in a sort key.

    Rows are typed `dict[str, object]` because a Parquet row holds mixed types;
    sorting needs the integer columns actually typed as integers.
    """
    if isinstance(value, int):
        return value
    raise TypeError(f"expected an integer column, got {type(value).__name__}: {value!r}")


_DATA_FILE = "bars.parquet"
"""One file per partition. A partition is one instrument-session, which is a few
hundred rows for 1-minute bars — small enough that rewriting it whole is cheap,
and large enough that per-file overhead stays negligible."""

_DEDUPE_KEY = ("interval", "ts_event", "seq")
"""What makes a bar unique within a partition. `ts_event` alone is not enough:
one instrument can have 1-minute and 5-minute bars stamped at the same close
time, and they are different rows."""


class ParquetStore:
    """The corpus on local disk.

    Satisfies `StoragePort` structurally — no import from `core.ports`, which is
    what keeps the dependency arrow pointing from adapters to core.

    Example:
        >>> import tempfile
        >>> from neurotrade.core.clock import SimClock
        >>> with tempfile.TemporaryDirectory() as directory:
        ...     store = ParquetStore(Path(directory), SimClock(1_000))
        ...     store.write_bars([], source="ibkr", session_date=date(2026, 3, 14))
        ...     list(store.read_bars(AAPL, BarInterval.MIN_1, 0, 10_000))
        []
    """

    __slots__ = ("_clock", "_root")

    def __init__(self, root: Path, clock: Clock) -> None:
        """Create a store rooted at a directory.

        Args:
            root: Dataset root, normally `settings.storage.raw_dir / "bars"`.
                Created on first write; it does not have to exist yet.
            clock: Supplies `ingested_at`. Injected rather than read from the OS
                so that a replay writes the same bytes every time.
        """
        self._root = root
        self._clock = clock

    # ── Writing ──────────────────────────────────────────────

    def write_bars(
        self,
        bars: Sequence[Bar],
        *,
        source: str,
        session_date: date,
    ) -> None:
        """Persist one session's bars, replacing whatever was there.

        Args:
            bars: Bars to persist. May span instruments; all belong to
                `session_date`. An empty sequence is a no-op, which matters
                because a holiday or a halted symbol legitimately yields none.
            source: Feed that produced them, as a `Source` value.
            session_date: The trading day in the venue's terms.

        Raises:
            ValueError: If `source` is not a known feed. A typo would otherwise
                become an unqueryable provenance value scattered through the
                corpus.

        Example:
            >>> import tempfile
            >>> from neurotrade.core.clock import SimClock
            >>> bar = Bar(symbol=AAPL, ts_event=1_000, ts_init=1_000,
            ...           interval=BarInterval.MIN_1, open=Price("100"),
            ...           high=Price("101"), low=Price("99"), close=Price("100.5"),
            ...           volume=Quantity(10))
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     store = ParquetStore(Path(directory), SimClock(5_000))
            ...     store.write_bars([bar], source="ibkr", session_date=date(2026, 3, 14))
            ...     store.write_bars([bar], source="ibkr", session_date=date(2026, 3, 14))
            ...     len(list(store.read_bars(AAPL, BarInterval.MIN_1, 0, 10_000)))
            1
        """
        if not bars:
            return

        feed = Source(source)
        ingested_at = self._clock.now_ns()

        by_symbol: dict[Symbol, list[Bar]] = {}
        for bar in bars:
            by_symbol.setdefault(bar.symbol, []).append(bar)

        for symbol, symbol_bars in by_symbol.items():
            self._write_partition(symbol, session_date, symbol_bars, feed, ingested_at)

    def _write_partition(
        self,
        symbol: Symbol,
        session_date: date,
        bars: Sequence[Bar],
        source: Source,
        ingested_at: Nanos,
    ) -> None:
        """Merge bars into one partition and rewrite it.

        Read-merge-write rather than append. Appending is faster and wrong: the
        crawler re-fetches after an interruption, and a duplicated bar reads as
        doubled volume rather than as an error.
        """
        directory = partition_path(self._root, symbol, session_date)
        path = directory / _DATA_FILE

        incoming = [
            bar_to_row(bar, source=source, session_date=session_date, ingested_at=ingested_at)
            for bar in bars
        ]

        # Existing rows first, so that a re-fetch of the same bar keeps the
        # original `ingested_at`. The bar is identical either way; preserving
        # the first sighting makes "when did we learn this" answerable.
        existing = pq.read_table(path).to_pylist() if path.exists() else []
        merged = self._deduplicate([*existing, *incoming])

        directory.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(merged, schema=BAR_SCHEMA), path)

    @staticmethod
    def _deduplicate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
        """Drop repeats and sort, keeping the first sighting of each bar.

        Sorting on write means readers get ordered bars without sorting, and the
        file bytes depend only on content — two crawls that fetch the same
        session in different orders produce the same file.
        """
        seen: dict[tuple[object, ...], dict[str, object]] = {}
        for row in rows:
            key = tuple(row[column] for column in _DEDUPE_KEY)
            seen.setdefault(key, row)
        return sorted(
            seen.values(),
            key=lambda row: (str(row["interval"]), _as_int(row["ts_event"]), _as_int(row["seq"])),
        )

    # ── Reading ──────────────────────────────────────────────

    def read_bars(
        self,
        symbol: Symbol,
        interval: BarInterval,
        start: Nanos,
        end: Nanos,
    ) -> Iterator[Bar]:
        """Read bars for one instrument over a half-open time range.

        Args:
            symbol: Instrument to read.
            interval: Bar size.
            start: Inclusive lower bound on `ts_event`.
            end: Exclusive upper bound on `ts_event`.

        Returns:
            Bars in ascending `(ts_event, seq)` order. Empty when nothing has
            been written — a missing dataset is an ordinary state during a
            backfill, not an error.

        Example:
            >>> import tempfile
            >>> from neurotrade.core.clock import SimClock
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     store = ParquetStore(Path(directory), SimClock(0))
            ...     list(store.read_bars(AAPL, BarInterval.MIN_1, 0, 1))
            []
        """
        instrument_root = self._root / f"venue={symbol.venue.value}" / f"ticker={symbol.ticker}"
        if not instrument_root.exists():
            return iter(())

        dataset = ds.dataset(instrument_root, format="parquet", partitioning=SESSION_PARTITIONING)
        table = dataset.to_table(
            filter=(
                (ds.field("interval") == interval.value)
                & (ds.field("ts_event") >= start)
                & (ds.field("ts_event") < end)
            ),
            columns=list(BAR_SCHEMA.names),
        )
        rows = sorted(table.to_pylist(), key=lambda row: (row["ts_event"], row["seq"]))
        return (row_to_bar(row) for row in rows)

    def sessions(self, symbol: Symbol) -> tuple[date, ...]:
        """Session dates held for an instrument, sorted.

        Read straight off the directory names — no file is opened. The crawler
        uses this to decide what to fetch next, and doing that by listing
        directories keeps resumption cheap even when the corpus is large.

        Example:
            >>> import tempfile
            >>> from neurotrade.core.clock import SimClock
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     ParquetStore(Path(directory), SimClock(0)).sessions(AAPL)
            ()
        """
        instrument_root = self._root / f"venue={symbol.venue.value}" / f"ticker={symbol.ticker}"
        if not instrument_root.exists():
            return ()
        return tuple(
            sorted(
                date.fromisoformat(child.name.removeprefix("session_date="))
                for child in instrument_root.iterdir()
                if child.is_dir() and child.name.startswith("session_date=")
            )
        )

    def __repr__(self) -> str:
        return f"ParquetStore({self._root})"
