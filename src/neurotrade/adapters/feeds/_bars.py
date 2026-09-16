"""Shared open-to-close, ET-to-UTC conversion for the seed feed adapters.

Both vendors' files are US/Eastern, naive, and stamp a bar at its **open**
(§12.1 stage 2, decision 6) — the same shift `adapters/ibkr/market_data.py`
applies live, done here without a socket. `firstrate.py` and `kibot.py` differ
only in how a row's raw fields are split out of a line; once split, both go
through `close_ts_ns`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
from zoneinfo import ZoneInfo

from neurotrade.core.clock import Nanos, to_datetime, to_nanos
from neurotrade.core.events import BarInterval

__all__ = ["PreparedBar", "close_ts_ns", "covered_dates"]

EASTERN = ZoneInfo("America/New_York")
"""Both vendors' files are ET, naive — no offset in the row. Attaching this via
`replace` (not `astimezone`) lets `zoneinfo` resolve the correct UTC offset for
that instant, DST included, from a plain wall-clock reading."""


@dataclass(frozen=True, slots=True)
class PreparedBar:
    """One parsed vendor row, close-stamped and ready to become a `Bar`.

    Kept separate from `Bar` itself because `ts_init` belongs to the moment a
    caller called `fetch_bars`, not the moment the file was parsed — parsing
    happens once per file and is cached, but `ts_init` must reflect every call.
    """

    ts_event: Nanos  # the bar's CLOSE, UTC nanoseconds — see close_ts_ns
    open: float
    high: float
    low: float
    close: float
    volume: float


def close_ts_ns(open_et: datetime, interval: BarInterval) -> Nanos:
    """Convert a bar's ET open time to its UTC close timestamp.

    Mirrors `IbkrMarketData._to_bar`'s shift, but starting from a naive local
    time instead of a UTC one: both vendors stamp at the open, `Bar.ts_event`
    is the close, and skipping this is the systematic lookahead bug described
    in `08-gotchas.doc.md`.

    Args:
        open_et: The bar's open, as a naive local time in US/Eastern — the
            form both vendor files give it in.
        interval: The bar's length. The shift from open to close is exactly
            one interval.

    Returns:
        `ts_event`: epoch nanoseconds, UTC, at the bar's close.

    Raises:
        ValueError: If `open_et` is not naive (already carries a `tzinfo`),
            which would mean a caller misread the file's own timezone.

    Example:
        >>> close_ts_ns(datetime(2023, 1, 3, 9, 30), BarInterval.MIN_1)
        1672756260000000000
    """
    if open_et.tzinfo is not None:
        raise ValueError(f"expected a naive local time, got {open_et!r}")
    aware = open_et.replace(tzinfo=EASTERN)
    return to_nanos(aware) + interval.nanos


def covered_dates(rows: Iterable[PreparedBar], interval: BarInterval) -> tuple[date, date] | None:
    """The first and last ET session date a set of parsed rows covers.

    Lets a caller crawl a vendor file over exactly the range it holds, instead
    of being told the window: FirstRateData's is fixed but undocumented in the
    file, and Kibot's rolls (`08-gotchas.doc.md`), so any hardcoded range is
    wrong eventually and fails by quietly ingesting less than the file has.

    The date is the **open's** date in Eastern, recovered by undoing
    `close_ts_ns`. Taking it from the UTC close instead would put a 20:00 ET
    bar on the following day, and the range would miss that session's own
    trading day at either end.

    Args:
        rows: Parsed rows, in any order.
        interval: The bar length the rows were parsed with, to undo the shift.

    Returns:
        `(first, last)` inclusive, or None if `rows` is empty.

    Example:
        >>> rows = [
        ...     PreparedBar(close_ts_ns(datetime(2023, 1, 3, 9, 30), BarInterval.MIN_1),
        ...                 1.0, 1.0, 1.0, 1.0, 1.0),
        ...     PreparedBar(close_ts_ns(datetime(2023, 1, 4, 15, 59), BarInterval.MIN_1),
        ...                 1.0, 1.0, 1.0, 1.0, 1.0),
        ... ]
        >>> covered_dates(rows, BarInterval.MIN_1)
        (datetime.date(2023, 1, 3), datetime.date(2023, 1, 4))
    """
    dates = [to_datetime(row.ts_event - interval.nanos).astimezone(EASTERN).date() for row in rows]
    if not dates:
        return None
    return min(dates), max(dates)
