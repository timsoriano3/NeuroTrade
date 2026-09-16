"""1-minute bars from a Kibot `_unadjusted` free sample file.

Satisfies `MarketDataPort` structurally, reading a file a caller already put
on disk — via `vendor_download.fetch_kibot_samples` or a manual drop — rather
than reaching a network itself. See `firstrate.py`'s module docstring for why
this feed never reaches one.

Kibot's `_unadjusted` files (decision 7) are headerless CSV, ET, naive,
`MM/DD/YYYY,HH:MM,open,high,low,close,volume`, stamped at the bar's **open**,
regular session hours only. See `_bars.py` for the open-to-close, ET-to-UTC
conversion shared with `firstrate.py`.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from pathlib import Path

from neurotrade.adapters.feeds._bars import PreparedBar, close_ts_ns, covered_dates
from neurotrade.adapters.feeds.errors import FeedError
from neurotrade.core.clock import Clock, Nanos
from neurotrade.core.events import Bar, BarInterval
from neurotrade.core.types import Price, Quantity, Symbol

__all__ = ["KibotFeed"]

_FIELD_COUNT = 7
_DATETIME_FORMAT = "%m/%d/%Y %H:%M"


class KibotFeed:
    """A `MarketDataPort` reading Kibot `_unadjusted` sample files already on disk.

    Example:
        >>> from neurotrade.core.clock import SimClock
        >>> from neurotrade.core.ports import MarketDataPort
        >>> isinstance(KibotFeed({}, SimClock(0)), MarketDataPort)
        True
    """

    __slots__ = ("_cache", "_clock", "_files")

    def __init__(self, files: Mapping[Symbol, Path], clock: Clock) -> None:
        """Create the feed.

        Args:
            files: Which file backs each symbol. Built by the caller from
                `SeedSourcesFile` plus wherever the file landed — this class
                does not know about either.
            clock: Stamps `ts_init` on every bar returned.
        """
        self._files = dict(files)
        self._clock = clock
        self._cache: dict[Symbol, tuple[PreparedBar, ...]] = {}

    async def is_connected(self) -> bool:
        """Always true: a file on disk has no connection to lose."""
        return True

    async def fetch_bars(
        self,
        symbol: Symbol,
        interval: BarInterval,
        start: Nanos,
        end: Nanos,
    ) -> Sequence[Bar]:
        """Read bars for one symbol out of its configured sample file.

        Args:
            symbol: Instrument to fetch. Must be a key in the mapping this was
                constructed with.
            interval: Must be `BarInterval.MIN_1` — the only size Kibot's
                sample files hold.
            start: Inclusive lower bound on `ts_event` (bar close).
            end: Exclusive upper bound on `ts_event`.

        Returns:
            Bars in ascending `ts_event` order.

        Raises:
            FeedError: If `symbol` was not configured, `interval` is not
                `MIN_1`, or the file does not parse.

        Example:
            >>> import asyncio
            >>> from neurotrade.core.clock import SimClock
            >>> from neurotrade.core.types import Venue
            >>> feed = KibotFeed({}, SimClock(0))
            >>> asyncio.run(feed.fetch_bars(Symbol("IBM", Venue.NYSE), BarInterval.MIN_5, 0, 1))
            Traceback (most recent call last):
                ...
            neurotrade.adapters.feeds.errors.FeedError: kibot sample feed only has 1m bars, not 5m
        """
        if interval is not BarInterval.MIN_1:
            raise FeedError(
                f"kibot sample feed only has {BarInterval.MIN_1.value} bars, not {interval.value}"
            )
        path = self._files.get(symbol)
        if path is None:
            raise FeedError(f"no kibot file configured for {symbol}")

        prepared = self._prepared(symbol, path)

        received_at = self._clock.now_ns()
        return tuple(
            Bar(
                symbol=symbol,
                ts_event=row.ts_event,
                ts_init=received_at,
                interval=interval,
                open=Price.from_float(row.open),
                high=Price.from_float(row.high),
                low=Price.from_float(row.low),
                close=Price.from_float(row.close),
                volume=Quantity.from_float(row.volume),
            )
            for row in prepared
            if start <= row.ts_event < end
        )

    def coverage(self) -> tuple[date, date] | None:
        """First and last session date the configured files hold, ET.

        What `seed ingest` crawls over, so the command needs no knowledge of
        either vendor's window — see `covered_dates` in `_bars.py` for why a
        hardcoded range goes stale. Parses every configured file, which is the
        same work `fetch_bars` would do and is cached with it.

        Returns:
            `(first, last)` inclusive, or None if no file holds a row.

        Example:
            >>> from neurotrade.core.clock import SimClock
            >>> KibotFeed({}, SimClock(0)).coverage() is None
            True
        """
        first: date | None = None
        last: date | None = None
        for symbol, path in self._files.items():
            span = covered_dates(self._prepared(symbol, path), BarInterval.MIN_1)
            if span is None:
                continue
            first = span[0] if first is None else min(first, span[0])
            last = span[1] if last is None else max(last, span[1])
        if first is None or last is None:
            return None
        return first, last

    def _prepared(self, symbol: Symbol, path: Path) -> tuple[PreparedBar, ...]:
        """Parsed rows for one file, parsed once and kept.

        Raises:
            FeedError: If the file does not parse.
        """
        prepared = self._cache.get(symbol)
        if prepared is None:
            prepared = _parse(path)
            self._cache[symbol] = prepared
        return prepared

    def __repr__(self) -> str:
        return f"KibotFeed(symbols={len(self._files)})"


def _parse(path: Path) -> tuple[PreparedBar, ...]:
    """Parse one Kibot `_unadjusted` file into ascending, close-stamped bars.

    Raises:
        FeedError: If a row does not parse.
    """
    with path.open(newline="") as handle:
        rows = [_row(path, line, fields) for line, fields in enumerate(csv.reader(handle), start=1)]
    return tuple(sorted(rows, key=lambda row: row.ts_event))


def _row(path: Path, line: int, fields: list[str]) -> PreparedBar:
    """Parse one headerless CSV row into a `PreparedBar`.

    Raises:
        FeedError: If the row has the wrong number of fields, or an
            unparseable date, time, or number.
    """
    if len(fields) != _FIELD_COUNT:
        raise FeedError(f"{path}:{line}: expected {_FIELD_COUNT} fields, got {fields!r}")
    date_, time_, open_, high, low, close, volume = fields
    try:
        # Deliberately naive: the file gives ET wall-clock time with no offset
        # of its own, and close_ts_ns is what attaches the zone.
        moment = datetime.strptime(f"{date_} {time_}", _DATETIME_FORMAT)  # noqa: DTZ007
    except ValueError as error:
        raise FeedError(f"{path}:{line}: unparseable date/time {date_!r} {time_!r}") from error
    try:
        return PreparedBar(
            ts_event=close_ts_ns(moment, BarInterval.MIN_1),
            open=float(open_),
            high=float(high),
            low=float(low),
            close=float(close),
            volume=float(volume),
        )
    except ValueError as error:
        raise FeedError(f"{path}:{line}: unparseable number in {fields!r}") from error
