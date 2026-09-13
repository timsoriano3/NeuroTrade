"""Queries over the corpus: what is held, and what is missing.

`ParquetStore` answers "give me these bars". This answers the questions asked
*about* the corpus rather than of it — what sessions exist, where the holes are,
which feed supplied what. Those are the questions the backfill crawler needs to
decide what to fetch next, and the ones §12.1's quality gate needs to decide
whether the corpus can be trusted.

**DuckDB reads the Parquet files directly.** There is no import step and no
second copy of the data: the lake on disk *is* the database. That matters
because the corpus is the expensive asset — anything that required loading it
somewhere else before querying would be a second thing to keep in sync, and the
two would eventually disagree.

**Hive partitioning is deliberately not used here.** `ticker`, `venue` and
`session_date` are stored as real columns inside every file as well as in the
directory names, so a plain `read_parquet` glob returns them with their true
types. Asking DuckDB to infer them from directory names would work too, and
would reintroduce the string-versus-date32 ambiguity that `SESSION_PARTITIONING`
exists to pin down.

**A missing corpus is not an error.** Every query returns empty when nothing has
been written. During a backfill that runs for weeks, "nothing yet" is the normal
state for most instruments, and a catalog that raised would make the crawler's
own progress reporting the first thing to break.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import duckdb

from neurotrade.core.clock import Nanos
from neurotrade.core.events import BarInterval
from neurotrade.core.types import Symbol, Venue

__all__ = [
    "Coverage",
    "DuckDBCatalog",
    "Gap",
]


@dataclass(frozen=True, slots=True)
class Coverage:
    """What the corpus holds for one instrument-session.

    Example:
        >>> held = Coverage(
        ...     symbol=AAPL, session_date=date(2026, 3, 14), interval=BarInterval.MIN_1,
        ...     bar_count=390, first_ts=1_000, last_ts=2_000, sources=("ibkr",),
        ... )
        >>> held.is_complete(390)
        True
    """

    symbol: Symbol  # instrument
    session_date: date  # trading day, in the venue's terms
    interval: BarInterval  # bar size
    bar_count: int  # rows held for this instrument-session
    first_ts: Nanos  # earliest ts_event held
    last_ts: Nanos  # latest ts_event held
    sources: tuple[str, ...]  # feeds that contributed, sorted

    def is_complete(self, expected_bars: int) -> bool:
        """Whether the session holds at least the expected number of bars.

        Args:
            expected_bars: What a full session should contain — 390 for
                1-minute US regular hours, more if pre- and post-market are
                included.

        Returns:
            True when nothing is missing. Deliberately `>=` rather than `==`:
            extended-hours bars legitimately push a session past the regular
            count, and treating that as a fault would flag every day.
        """
        return self.bar_count >= expected_bars

    @property
    def is_mixed_source(self) -> bool:
        """Whether more than one feed contributed to this session.

        Worth knowing before trusting a volume feature: IEX-only data carries
        partial volume, so a session stitched from IEX and IBKR has a
        discontinuity that no price column reveals.
        """
        return len(self.sources) > 1


@dataclass(frozen=True, slots=True)
class Gap:
    """A run of missing bars inside a session.

    Example:
        >>> Gap(symbol=AAPL, session_date=date(2026, 3, 14), interval=BarInterval.MIN_1,
        ...     after_ts=1_000, before_ts=1_000 + 5 * 60_000_000_000).missing_bars
        4
    """

    symbol: Symbol  # instrument
    session_date: date  # trading day
    interval: BarInterval  # bar size the gap is measured against
    after_ts: Nanos  # last bar present before the hole
    before_ts: Nanos  # first bar present after the hole

    @property
    def missing_bars(self) -> int:
        """How many bars the gap could hold.

        A halt produces a genuine gap and so does a failed fetch; this number
        does not distinguish them. `TradingHalt` events do, which is why §12.1
        pairs gap detection with halt marking rather than treating every hole as
        a fault.
        """
        return (self.before_ts - self.after_ts) // self.interval.nanos - 1


class DuckDBCatalog:
    """Read-only questions about the corpus.

    Example:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as directory:
        ...     DuckDBCatalog(Path(directory)).summary()
        {'bars': 0, 'instruments': 0, 'sessions': 0}
    """

    __slots__ = ("_root",)

    def __init__(self, root: Path) -> None:
        """Open a catalog over a dataset root.

        Args:
            root: The same directory a `ParquetStore` writes to. Nothing is
                opened here; each query globs the current contents, so a catalog
                stays correct while a crawler writes underneath it.
        """
        self._root = root

    @property
    def _glob(self) -> str:
        return str(self._root / "**" / "*.parquet")

    def _query(self, sql: str, *parameters: object) -> list[tuple[object, ...]]:
        """Run a query, returning no rows when the corpus is empty.

        DuckDB raises when a glob matches no files. That is the ordinary state
        for most instruments during a multi-week backfill, so it is translated
        to an empty result rather than propagated.
        """
        if not self._root.exists() or not any(self._root.rglob("*.parquet")):
            return []
        return duckdb.sql(sql.replace("{glob}", self._glob), params=list(parameters)).fetchall()

    # ── What is held ─────────────────────────────────────────

    def coverage(
        self,
        symbol: Symbol | None = None,
        interval: BarInterval = BarInterval.MIN_1,
    ) -> tuple[Coverage, ...]:
        """Per-session coverage, oldest first.

        Args:
            symbol: Restrict to one instrument. All instruments when omitted.
            interval: Bar size.

        Returns:
            One `Coverage` per instrument-session held, ordered by instrument
            then date.

        Example:
            >>> import tempfile
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     DuckDBCatalog(Path(directory)).coverage()
            ()
        """
        clause, parameters = self._symbol_clause(symbol)
        rows = self._query(
            f"""
            SELECT venue, ticker, session_date,
                   count(*)            AS bar_count,
                   min(ts_event)       AS first_ts,
                   max(ts_event)       AS last_ts,
                   list_sort(list(DISTINCT source)) AS sources
            FROM read_parquet('{{glob}}')
            WHERE interval = ? {clause}
            GROUP BY venue, ticker, session_date
            ORDER BY venue, ticker, session_date
            """,
            interval.value,
            *parameters,
        )
        return tuple(
            Coverage(
                symbol=Symbol(str(row[1]), Venue(str(row[0]))),
                session_date=_as_date(row[2]),
                interval=interval,
                bar_count=_as_int(row[3]),
                first_ts=_as_int(row[4]),
                last_ts=_as_int(row[5]),
                sources=_as_str_tuple(row[6]),
            )
            for row in rows
        )

    def sessions_held(
        self, symbol: Symbol, interval: BarInterval = BarInterval.MIN_1
    ) -> tuple[date, ...]:
        """Session dates held for an instrument, sorted.

        Example:
            >>> import tempfile
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     DuckDBCatalog(Path(directory)).sessions_held(AAPL)
            ()
        """
        return tuple(entry.session_date for entry in self.coverage(symbol, interval))

    def bar_counts(
        self, symbol: Symbol, interval: BarInterval = BarInterval.MIN_1
    ) -> dict[date, int]:
        """Bars held per session for one instrument.

        Satisfies `CatalogPort`. A narrower query than `coverage` on purpose:
        the crawler asks this once per symbol across the whole universe, and
        the timestamps and source lists `coverage` also computes are work it
        does not need.

        Args:
            symbol: Instrument to describe.
            interval: Bar size.

        Returns:
            Session date to bar count, ascending, for sessions holding at least
            one bar. Sessions with nothing held are absent rather than zero.

        Example:
            >>> import tempfile
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     DuckDBCatalog(Path(directory)).bar_counts(AAPL)
            {}
        """
        clause, parameters = self._symbol_clause(symbol)
        rows = self._query(
            f"""
            SELECT session_date, count(*) AS bar_count
            FROM read_parquet('{{glob}}')
            WHERE interval = ? {clause}
            GROUP BY session_date
            ORDER BY session_date
            """,
            interval.value,
            *parameters,
        )
        return {_as_date(row[0]): _as_int(row[1]) for row in rows}

    def missing_sessions(
        self,
        symbol: Symbol,
        expected: list[date],
        interval: BarInterval = BarInterval.MIN_1,
    ) -> tuple[date, ...]:
        """Sessions the calendar expects that the corpus does not have.

        This is the crawler's work queue. It takes the expected sessions rather
        than deriving them, because knowing which days a venue traded requires
        an exchange calendar — holidays, half days, and the days a listing did
        not yet exist — and that is not storage's business.

        Args:
            symbol: Instrument.
            expected: Sessions the venue calendar says should exist.
            interval: Bar size.

        Returns:
            The expected sessions with no data, in order.
        """
        held = set(self.sessions_held(symbol, interval))
        return tuple(session for session in sorted(expected) if session not in held)

    # ── What is wrong ────────────────────────────────────────

    def gaps(
        self,
        symbol: Symbol,
        session_date: date,
        interval: BarInterval = BarInterval.MIN_1,
    ) -> tuple[Gap, ...]:
        """Holes inside one session, in order.

        Found by comparing each bar's timestamp with the previous one: anything
        further apart than one interval is a gap. A halt and a failed fetch look
        identical here, which is why §12.1 pairs this with halt marking.

        Args:
            symbol: Instrument.
            session_date: Trading day to examine.
            interval: Bar size, which sets the expected spacing.

        Returns:
            One `Gap` per hole.
        """
        rows = self._query(
            """
            WITH ordered AS (
                SELECT ts_event,
                       lag(ts_event) OVER (ORDER BY ts_event) AS previous_ts
                FROM read_parquet('{glob}')
                WHERE interval = ? AND ticker = ? AND venue = ? AND session_date = ?
            )
            SELECT previous_ts, ts_event
            FROM ordered
            WHERE previous_ts IS NOT NULL AND ts_event - previous_ts > ?
            ORDER BY previous_ts
            """,
            interval.value,
            symbol.ticker,
            symbol.venue.value,
            session_date,
            interval.nanos,
        )
        return tuple(
            Gap(
                symbol=symbol,
                session_date=session_date,
                interval=interval,
                after_ts=_as_int(row[0]),
                before_ts=_as_int(row[1]),
            )
            for row in rows
        )

    def duplicate_timestamps(
        self, interval: BarInterval = BarInterval.MIN_1
    ) -> tuple[tuple[Symbol, date, Nanos, int], ...]:
        """Bars sharing an instrument, interval and timestamp.

        Should always be empty: `ParquetStore` deduplicates on write. This exists
        to prove that rather than assume it, because a duplicate reads as double
        the volume and nothing downstream would flag it.
        """
        rows = self._query(
            """
            SELECT venue, ticker, session_date, ts_event, count(*) AS occurrences
            FROM read_parquet('{glob}')
            WHERE interval = ?
            GROUP BY venue, ticker, session_date, ts_event, seq
            HAVING count(*) > 1
            ORDER BY venue, ticker, ts_event
            """,
            interval.value,
        )
        return tuple(
            (
                Symbol(str(row[1]), Venue(str(row[0]))),
                _as_date(row[2]),
                _as_int(row[3]),
                _as_int(row[4]),
            )
            for row in rows
        )

    # ── Totals ───────────────────────────────────────────────

    def summary(self) -> dict[str, int]:
        """Corpus totals, for progress reporting during a backfill.

        Example:
            >>> import tempfile
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     DuckDBCatalog(Path(directory)).summary()
            {'bars': 0, 'instruments': 0, 'sessions': 0}
        """
        rows = self._query(
            """
            SELECT count(*),
                   count(DISTINCT (venue, ticker)),
                   count(DISTINCT session_date)
            FROM read_parquet('{glob}')
            """
        )
        if not rows:
            return {"bars": 0, "instruments": 0, "sessions": 0}
        bars, instruments, sessions = rows[0]
        return {
            "bars": _as_int(bars),
            "instruments": _as_int(instruments),
            "sessions": _as_int(sessions),
        }

    @staticmethod
    def _symbol_clause(symbol: Symbol | None) -> tuple[str, tuple[object, ...]]:
        """Build the optional instrument filter."""
        if symbol is None:
            return "", ()
        return "AND ticker = ? AND venue = ?", (symbol.ticker, symbol.venue.value)

    def __repr__(self) -> str:
        return f"DuckDBCatalog({self._root})"


def _as_date(value: object) -> date:
    """Narrow a DuckDB date column."""
    if isinstance(value, date):
        return value
    raise TypeError(f"expected a date column, got {type(value).__name__}: {value!r}")


def _as_int(value: object) -> int:
    """Narrow a DuckDB integer column.

    Query results are typed `object` because a row holds mixed types. Narrowing
    once here keeps the construction sites readable and rejects a surprise
    loudly instead of coercing it.
    """
    if isinstance(value, int):
        return value
    raise TypeError(f"expected an integer column, got {type(value).__name__}: {value!r}")


def _as_str_tuple(value: object) -> tuple[str, ...]:
    """Narrow a DuckDB list column to a tuple of strings."""
    if isinstance(value, list):
        return tuple(str(item) for item in value)
    raise TypeError(f"expected a list column, got {type(value).__name__}: {value!r}")
