"""What the corpus holds, what is missing, and what looks wrong (§12.1 stage 5).

The vocabulary for describing a corpus rather than trading on one. These types
live in `core` because they are domain concepts — a gap in a session is a gap
whether the bars sit in Parquet, Postgres or memory — and because `CatalogPort`
returns them, so a concrete catalog must not own them. `core/calendar.py` has
carried a comment about that misplacement since `expected_bars` was written.

**Why a quality gate is a Phase 1 deliverable and not a nicety.** §17 names
backtest overfitting as the project's primary risk, and a corpus fault is the
cheapest way to manufacture one. A session missing its last twenty minutes
makes every closing-range feature wrong in the same direction on the same days.
A duplicated print doubles a volume bar. A halted name looks like a flat,
liquid one. None of these crash anything; they produce a clean equity curve
that does not survive live.

**Nothing here decides whether a fault is fatal.** A gap can be a failed fetch
or a genuine trading halt, and no count distinguishes them — which is exactly
why §12.1 pairs gap detection with halt marking. These types report; the
operator judges.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from neurotrade.core.clock import Nanos
from neurotrade.core.events import BarInterval
from neurotrade.core.types import Symbol

__all__ = [
    "Coverage",
    "Duplicate",
    "Gap",
    "SuspectSession",
]


@dataclass(frozen=True, slots=True)
class Coverage:
    """What the corpus holds for one instrument-session.

    Example:
        >>> from neurotrade.core.types import Venue
        >>> held = Coverage(
        ...     symbol=Symbol("AAPL", Venue.NASDAQ), session_date=date(2026, 3, 14),
        ...     interval=BarInterval.MIN_1,
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
        >>> from neurotrade.core.types import Venue
        >>> Gap(symbol=Symbol("AAPL", Venue.NASDAQ), session_date=date(2026, 3, 14),
        ...     interval=BarInterval.MIN_1,
        ...     after_ts=1_000, before_ts=1_000 + 5 * 60_000_000_000).missing_bars
        4
    """

    symbol: Symbol  # instrument
    session_date: date  # trading day
    interval: BarInterval  # bar size the gap is measured against
    after_ts: Nanos  # last bar present before the hole
    before_ts: Nanos  # first bar present after the hole

    def __str__(self) -> str:
        return (
            f"{self.symbol} {self.session_date}: {self.missing_bars} bars missing "
            f"after {self.after_ts}"
        )

    @property
    def missing_bars(self) -> int:
        """How many bars the gap could hold.

        A halt produces a genuine gap and so does a failed fetch; this number
        does not distinguish them. `TradingHalt` events do, which is why §12.1
        pairs gap detection with halt marking rather than treating every hole as
        a fault.
        """
        return (self.before_ts - self.after_ts) // self.interval.nanos - 1


@dataclass(frozen=True, slots=True)
class Duplicate:
    """Two or more bars sharing an instrument, interval and timestamp.

    Always a fault, unlike a gap. One instant had one set of prices, so a
    second row for it means an ingestion ran twice or two feeds disagreed. Left
    in place it double-counts volume and makes any bar-count check pass while
    the session is still wrong.

    Example:
        >>> from neurotrade.core.types import Venue
        >>> Duplicate(symbol=Symbol("AAPL", Venue.NASDAQ), session_date=date(2026, 3, 14),
        ...          interval=BarInterval.MIN_1, ts_event=1_000, count=2).extra_rows
        1
    """

    symbol: Symbol  # instrument
    session_date: date  # trading day the duplicate falls on
    interval: BarInterval  # bar size
    ts_event: Nanos  # the timestamp held more than once
    count: int  # how many rows share it; always at least 2

    def __str__(self) -> str:
        return f"{self.symbol} {self.session_date}: ts {self.ts_event} held {self.count} times"

    @property
    def extra_rows(self) -> int:
        """Rows beyond the one that should exist."""
        return self.count - 1


@dataclass(frozen=True, slots=True)
class SuspectSession:
    """A session whose bars are present but do not look like trading.

    Covers the halt-marking half of §12.1 stage 5. A halted name still produces
    rows — the feed reports the session — but they carry no volume, or a price
    that never moves. Both read downstream as a perfectly calm, perfectly
    liquid instrument, which is the most dangerous thing a bar can pretend to
    be: realised volatility collapses toward zero and any risk model sized off
    it takes an unbounded position.

    Example:
        >>> from neurotrade.core.types import Venue
        >>> SuspectSession(symbol=Symbol("AAPL", Venue.NASDAQ),
        ...                session_date=date(2026, 3, 14), interval=BarInterval.DAY_1,
        ...                reason="zero volume", bar_count=1).reason
        'zero volume'
    """

    symbol: Symbol  # instrument
    session_date: date  # trading day
    interval: BarInterval  # bar size
    reason: str  # why it is suspect, in plain words, for the report
    bar_count: int  # rows held for the session

    def __str__(self) -> str:
        return f"{self.symbol} {self.session_date}: {self.reason} ({self.bar_count} bars)"
