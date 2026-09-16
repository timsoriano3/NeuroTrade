"""1-minute bars from a FirstRateData free sample file.

Satisfies `MarketDataPort` structurally, reading a zip a caller already put on
disk — via `vendor_download.fetch_firstrate_samples` or a manual drop
(decision 1 makes both valid) — rather than reaching a network itself.

The zip holds one CSV: `timestamp,open,high,low,close,volume`, ET, naive,
`YYYY-MM-DD HH:MM:SS`, stamped at the bar's **open**. See `_bars.py` for the
open-to-close, ET-to-UTC conversion this shares with `kibot.py`.

**No calendar trim here.** FRD's sample covers extended hours, 04:00-20:00 ET,
which the crawler's session trim (`TradingSession.holds_bar`) drops on the way
into the corpus — this feed just answers what the file holds for the range
asked, same as any `MarketDataPort` implementation.
"""

from __future__ import annotations

import csv
import io
import zipfile
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from pathlib import Path

from neurotrade.adapters.feeds._bars import PreparedBar, close_ts_ns, covered_dates
from neurotrade.adapters.feeds.errors import FeedError
from neurotrade.core.clock import Clock, Nanos
from neurotrade.core.events import Bar, BarInterval
from neurotrade.core.types import Price, Quantity, Symbol

__all__ = ["FirstRateFeed"]

_HEADER = ["timestamp", "open", "high", "low", "close", "volume"]
_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


class FirstRateFeed:
    """A `MarketDataPort` reading FirstRateData sample zips already on disk.

    Example:
        >>> from neurotrade.core.clock import SimClock
        >>> from neurotrade.core.ports import MarketDataPort
        >>> isinstance(FirstRateFeed({}, SimClock(0)), MarketDataPort)
        True
    """

    __slots__ = ("_cache", "_clock", "_files")

    def __init__(self, files: Mapping[Symbol, Path], clock: Clock) -> None:
        """Create the feed.

        Args:
            files: Which zip backs each symbol. Built by the caller from
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
            interval: Must be `BarInterval.MIN_1` — the only size the sample
                files hold.
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
            >>> feed = FirstRateFeed({}, SimClock(0))
            >>> asyncio.run(feed.fetch_bars(Symbol("AAPL", Venue.NASDAQ), BarInterval.MIN_1, 0, 1))
            Traceback (most recent call last):
                ...
            neurotrade.adapters.feeds.errors.FeedError: no firstrate file configured for AAPL.NASDAQ
        """
        if interval is not BarInterval.MIN_1:
            raise FeedError(
                f"firstrate sample feed only has {BarInterval.MIN_1.value} bars, "
                f"not {interval.value}"
            )
        path = self._files.get(symbol)
        if path is None:
            raise FeedError(f"no firstrate file configured for {symbol}")

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
            >>> FirstRateFeed({}, SimClock(0)).coverage() is None
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
        return f"FirstRateFeed(symbols={len(self._files)})"


def _parse(path: Path) -> tuple[PreparedBar, ...]:
    """Parse one FRD zip into ascending, close-stamped bars.

    Raises:
        FeedError: If the zip does not hold exactly one member, the header
            does not match, or a row does not parse.
    """
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != 1:
            raise FeedError(f"{path}: expected exactly one file in the zip, found {names}")
        with archive.open(names[0]) as member:
            text = io.TextIOWrapper(member, encoding="utf-8", newline="")
            reader = csv.reader(text)
            header = next(reader, None)
            if header != _HEADER:
                raise FeedError(f"{path}: unexpected header {header!r}, expected {_HEADER}")
            rows = [_row(path, line, fields) for line, fields in enumerate(reader, start=2)]
    return tuple(sorted(rows, key=lambda row: row.ts_event))


def _row(path: Path, line: int, fields: list[str]) -> PreparedBar:
    """Parse one CSV row into a `PreparedBar`.

    Raises:
        FeedError: If the row has the wrong number of fields, or an
            unparseable timestamp or number.
    """
    if len(fields) != len(_HEADER):
        raise FeedError(f"{path}:{line}: expected {len(_HEADER)} fields, got {fields!r}")
    timestamp, open_, high, low, close, volume = fields
    try:
        # Deliberately naive: the file gives ET wall-clock time with no offset
        # of its own, and close_ts_ns is what attaches the zone.
        moment = datetime.strptime(timestamp, _TIMESTAMP_FORMAT)  # noqa: DTZ007
    except ValueError as error:
        raise FeedError(f"{path}:{line}: unparseable timestamp {timestamp!r}") from error
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
