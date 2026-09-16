"""Shared open-to-close, ET-to-UTC conversion for the seed feed adapters.

Both vendors' files are US/Eastern, naive, and stamp a bar at its **open**
(§12.1 stage 2, decision 6) — the same shift `adapters/ibkr/market_data.py`
applies live, done here without a socket. `firstrate.py` and `kibot.py` differ
only in how a row's raw fields are split out of a line; once split, both go
through `close_ts_ns`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from neurotrade.core.clock import Nanos, to_nanos
from neurotrade.core.events import BarInterval

__all__ = ["PreparedBar", "close_ts_ns"]

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
