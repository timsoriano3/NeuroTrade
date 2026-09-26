"""Splits and dividends from Yahoo Finance (§12.1 stage 5).

Satisfies `CorporateActionsPort`. Yahoo is the source because it is free, it
covers `.TO` listings, and the daily corpus already comes from it — adjusting
Yahoo bars with Yahoo actions keeps one provenance rather than two that can
disagree about a date.

**Its weakness is silent omission.** A feed that fails to report a split does
not raise; it returns a short list, and the prices it serves alongside are all
perfectly plausible. Nothing here can detect that, which is why the check lives
downstream in `unexplained_gaps` and runs over the corpus rather than over the
response. Treat this adapter as a *lead* on corporate actions and the gap scan
as the audit.

**Dates come back tz-aware and are converted to the venue's date.** Yahoo
stamps a split at 09:30 in the listing venue's local time — the session open on
which the new basis takes effect. Taking `.date()` of the UTC instant would put
an ET-morning split on the same calendar day (13:30 UTC), but the same is not
true everywhere, and a date that drifts by one session applies the split to the
wrong bar. The local date is used directly.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Final, Protocol

from neurotrade.adapters.feeds.errors import FeedError
from neurotrade.adapters.feeds.yfinance_daily import yahoo_ticker
from neurotrade.core.actions import ONE, CorporateAction
from neurotrade.core.types import Symbol, tidy_decimal

__all__ = [
    "ActionRow",
    "ActionsDownloader",
    "YFinanceActions",
    "YahooActionsDownloader",
]

_MONEY_DECIMALS: Final = 6
"""Decimal places a dividend is rounded to.

Yahoo answers float64, so a 27c dividend arrives as `0.27000000000000002`.
`Decimal(repr(...))` would keep all of it, and the value then fails to compare
equal to the `0.27` the same feed reports next week. Six places is well below a
cent and well above any real dividend's precision."""

_RATIO_DECIMALS: Final = 8
"""Decimal places a split ratio is rounded to. Ratios are usually small whole
numbers, but reverse splits are fractional (a 1-for-10 arrives as 0.1) and
odd ratios like 3-for-2 arrive as 1.5."""


@dataclass(frozen=True, slots=True)
class ActionRow:
    """One action row as Yahoo gave it, before conversion to the domain type.

    The seam between the network and the domain, mirroring `DailyRow`: tests
    build these by hand, so the conversion is exercised without reaching Yahoo.

    Example:
        >>> ActionRow(date(2020, 8, 31), split_ratio=4.0, dividend=0.0).split_ratio
        4.0
    """

    effective_date: date  # in the VENUE's timezone, not UTC — see the module docstring
    split_ratio: float  # 0.0 when the row is dividend-only, as Yahoo reports it
    dividend: float  # 0.0 when the row is split-only


class ActionsDownloader(Protocol):
    """How the adapter reaches Yahoo. Injected so tests never do.

    Conformance is structural, so a plain function of the right shape is a
    downloader:

    Example:
        >>> def none(ticker):
        ...     return ()
        >>> none("AAPL")
        ()
    """

    def __call__(self, ticker: str) -> Sequence[ActionRow]:
        """Fetch every action Yahoo holds for one symbol.

        The whole history is returned rather than a range. Yahoo serves it in
        one call regardless, splitting it costs an extra request per range, and
        an `AdjustmentSeries` wants the full set anyway — a 2014 split still
        applies to a 2013 bar.

        Args:
            ticker: Yahoo's own symbol, suffix included — `SHOP.TO`, not `SHOP`.

        Returns:
            Rows in ascending date order.

        Raises:
            FeedError: If Yahoo refuses or answers with something unparseable.
        """
        ...


class YahooActionsDownloader:
    """The real downloader: `yfinance`, one `Ticker.get_actions` call per symbol.

    Imports `yfinance` lazily for the same reason `YahooDownloader` does — the
    package costs a second of import time that `neurotrade --help` should not
    pay.

    **No timeout, unlike the daily downloader.** `get_actions` takes only
    `period` on yfinance 1.7.0 — passing `timeout` raises `TypeError` and every
    symbol fails. The daily feed's `history()` accepts one because it forwards
    `**kwargs`; this endpoint does not, so there is nothing to pass a ceiling
    to. Verified by `inspect.signature` against the installed version.

    Example:
        >>> isinstance(YahooActionsDownloader(), YahooActionsDownloader)
        True
    """

    __slots__ = ()

    def __call__(self, ticker: str) -> Sequence[ActionRow]:
        """Fetch one symbol's full action history.

        Args:
            ticker: Yahoo's symbol, suffix included.

        Returns:
            Rows in ascending date order. Empty when the name has never split
            or paid — the common case, and not an error.

        Raises:
            FeedError: If Yahoo refuses the request, or gives a row this cannot
                read.
        """
        import yfinance

        # Same two settings as the daily feed, and for the same reasons: an
        # exception hidden behind an empty frame would enter the corpus as "no
        # actions", and yfinance's DEBUG tracing would bury a crawl's progress.
        yfinance.config.debug.hide_exceptions = False
        logging.getLogger("yfinance").setLevel(logging.WARNING)

        try:
            frame = yfinance.Ticker(ticker).get_actions(period="max")
        except Exception as error:
            raise FeedError(f"yahoo actions request failed for {ticker}: {error}") from error

        if frame is None or frame.empty:
            return ()

        # A name that has only ever paid dividends comes back WITHOUT a
        # "Stock Splits" column, and one that has only ever split comes back
        # without "Dividends" — yfinance returns the columns it has, not a
        # fixed shape. Indexing blindly raises KeyError on exactly the names
        # with the cleanest history (SPY, TSLA), so a missing column means
        # zero, not an error.
        has_splits = "Stock Splits" in frame.columns
        has_dividends = "Dividends" in frame.columns
        if not (has_splits or has_dividends):
            raise FeedError(f"yahoo actions for {ticker} carry neither column")

        rows: list[ActionRow] = []
        for stamp, record in frame.iterrows():
            # `stamp` is tz-aware in the listing venue's zone. `.date()` on it
            # is the LOCAL date, which is the one the split takes effect on.
            try:
                rows.append(
                    ActionRow(
                        effective_date=stamp.date(),
                        split_ratio=float(record["Stock Splits"]) if has_splits else 0.0,
                        dividend=float(record["Dividends"]) if has_dividends else 0.0,
                    )
                )
            except (AttributeError, KeyError, TypeError, ValueError) as error:
                raise FeedError(
                    f"unreadable action row for {ticker} at {stamp}: {error}"
                ) from error
        rows.sort(key=lambda row: row.effective_date)
        return tuple(rows)


class YFinanceActions:
    """`CorporateActionsPort` over a Yahoo downloader.

    Caches per symbol: the downloader returns the whole history in one call, so
    a second question about a different date range costs nothing.

    Example:
        >>> from neurotrade.core.types import Venue
        >>> import asyncio
        >>> feed = YFinanceActions(lambda ticker: (ActionRow(date(2020, 8, 31), 4.0, 0.0),))
        >>> actions = asyncio.run(
        ...     feed.fetch_actions(Symbol("AAPL", Venue.NASDAQ), date(2020, 1, 1), date(2021, 1, 1))
        ... )
        >>> str(actions[0])
        'AAPL.NASDAQ 2020-08-31 4:1 split'
    """

    __slots__ = ("_cache", "_download")

    def __init__(self, download: ActionsDownloader | None = None) -> None:
        """Create the feed.

        Args:
            download: How to reach Yahoo. Defaults to the real downloader.
        """
        self._download = download if download is not None else YahooActionsDownloader()
        self._cache: dict[Symbol, tuple[CorporateAction, ...]] = {}

    async def fetch_actions(
        self, symbol: Symbol, start: date, end: date
    ) -> Sequence[CorporateAction]:
        """Actions effective within `[start, end]`, inclusive at both ends.

        Args:
            symbol: The instrument to fetch for.
            start: First effective date of interest.
            end: Last effective date of interest.

        Returns:
            The actions in range, oldest first.

        Raises:
            ValueError: If `end` precedes `start`.
            FeedError: If Yahoo refuses or answers unreadably.
        """
        if end < start:
            raise ValueError(f"end {end} is before start {start}")
        history = await self._history(symbol)
        return tuple(action for action in history if start <= action.effective_date <= end)

    async def _history(self, symbol: Symbol) -> tuple[CorporateAction, ...]:
        """The full history for a symbol, fetched once and remembered."""
        cached = self._cache.get(symbol)
        if cached is not None:
            return cached
        ticker = yahoo_ticker(symbol)
        # The downloader is synchronous and does network IO, so it goes to a
        # thread rather than blocking the loop the crawl runs on.
        rows = await asyncio.to_thread(self._download, ticker)
        history = tuple(_to_action(symbol, row) for row in rows)
        self._cache[symbol] = history
        return history


def _to_action(symbol: Symbol, row: ActionRow) -> CorporateAction:
    """Convert one Yahoo row to the domain type.

    Yahoo reports "no split on this row" as `0.0`, not `1.0`. Passing that
    through would make `CorporateAction` raise on a non-positive ratio — which
    it should, since a zero ratio is meaningless — so the identity is
    substituted here, at the boundary that knows the convention.

    Raises:
        FeedError: If the row carries a non-finite number. NaN reaches
            `Decimal` intact and then poisons every factor computed from it,
            silently, because NaN comparisons are all false.
    """
    for name, value in (("split_ratio", row.split_ratio), ("dividend", row.dividend)):
        if not math.isfinite(value):
            raise FeedError(f"{symbol} action on {row.effective_date} has non-finite {name}")

    ratio = ONE if row.split_ratio == 0.0 else _decimal(row.split_ratio, _RATIO_DECIMALS)
    return CorporateAction(
        symbol=symbol,
        effective_date=row.effective_date,
        split_ratio=ratio,
        dividend=_decimal(row.dividend, _MONEY_DECIMALS),
    )


def _decimal(value: float, places: int) -> Decimal:
    """Round a Yahoo float to a fixed scale.

    Via `repr(float(...))` rather than `Decimal(value)`: the direct conversion
    keeps the full binary expansion, so `0.27` becomes
    `0.27000000000000001776...` and never compares equal to the same dividend
    read back from Parquet. See `08-gotchas.doc.md` on `np.float64`.

    Example:
        >>> _decimal(0.27000000000000002, 6)
        Decimal('0.27')
        >>> _decimal(4.0, 8)
        Decimal('4')
    """
    return tidy_decimal(Decimal(repr(float(value))), places)
