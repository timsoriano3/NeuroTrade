"""Historical bars from IBKR, converted into domain events.

Three things here are not obvious, and each was verified against a live Gateway
rather than taken from documentation.

**IBKR stamps a bar at its OPEN; we stamp at its CLOSE.** A 390-bar US regular
session comes back running 13:30 to 19:59 UTC — the last bar is stamped at the
open of the final minute, not at 20:00. `Bar.ts_event` is the moment a bar became
an observable fact, so every timestamp gains one interval on the way in. Skip
that and every strategy acts one bar early, on every bar, forever: a systematic
lookahead that no test of the strategy itself would reveal.

**Venue names differ.** NASDAQ, NYSE and ARCA pass through unchanged; TSX is
`TSE` to IBKR. The mapping is explicit because a wrong exchange silently
qualifies a *different listing* — the same ticker in another currency at another
price.

**Prices arrive as floats.** They go through `Price.from_float`, which routes via
`repr` and is the marked boundary where precision was last trusted. Everything
downstream is exact.

Requests are paced (§12.1): IBKR permits about 60 historical requests per ten
minutes, and exceeding it locks out further requests rather than returning a
retryable error.

**Requests time out as errors, never as empty answers.** A Gateway can be up and
logged in while its data farms are unreachable (IBKR's weekend maintenance, for
one). Contract lookup then never answers, and `ib_async`'s historical request
times out by quietly returning no bars — indistinguishable from a halt. A
crawler would record every session as "nothing traded" and never notice the
feed was dead, so both cases raise `MarketDataError` here instead.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Protocol

from ib_async import Stock

from neurotrade.adapters.ibkr.connection import IbkrConnection
from neurotrade.adapters.ibkr.pacing import HistoricalPacer
from neurotrade.core.clock import Clock, Nanos, to_datetime
from neurotrade.core.events import Bar, BarInterval
from neurotrade.core.types import Price, Quantity, Symbol, Venue

__all__ = ["IBKR_EXCHANGE", "IbkrMarketData", "MarketDataError"]

logging.getLogger("ib_async").setLevel(logging.WARNING)

IBKR_EXCHANGE: dict[Venue, str] = {
    Venue.NASDAQ: "NASDAQ",
    Venue.NYSE: "NYSE",
    Venue.AMEX: "AMEX",
    Venue.ARCA: "ARCA",
    Venue.BATS: "BATS",
    Venue.TSX: "TSE",  # IBKR's name for the Toronto Stock Exchange
    Venue.TSXV: "VENTURE",  # and for TSX Venture
}
"""Our venue names to IBKR's. Mostly identity, which is exactly why the two
exceptions are worth stating: a wrong exchange does not error, it qualifies a
different listing of the same ticker."""

_BAR_SIZE: dict[BarInterval, str] = {
    BarInterval.SEC_1: "1 secs",
    BarInterval.SEC_5: "5 secs",
    BarInterval.SEC_15: "15 secs",
    BarInterval.SEC_30: "30 secs",
    BarInterval.MIN_1: "1 min",
    BarInterval.MIN_5: "5 mins",
    BarInterval.MIN_15: "15 mins",
    BarInterval.MIN_30: "30 mins",
    BarInterval.HOUR_1: "1 hour",
    BarInterval.DAY_1: "1 day",
}
"""IBKR's bar size strings. Irregular — "1 secs", "5 mins", "1 hour" — so this is
a table rather than a format string."""

_SECONDS_PER_DAY = 86_400

_MAX_DAYS: dict[BarInterval, int] = {
    BarInterval.SEC_1: 1,
    BarInterval.SEC_5: 7,
    BarInterval.SEC_15: 14,
    BarInterval.SEC_30: 28,
    BarInterval.MIN_1: 30,
    BarInterval.MIN_5: 100,
    BarInterval.MIN_15: 365,
    BarInterval.MIN_30: 365,
    BarInterval.HOUR_1: 365,
    BarInterval.DAY_1: 3_650,
}
"""How much history one request may cover, per bar size.

Verified against a live Gateway rather than taken from the documentation: one
minute bars return 390 for "1 D", 1,560 for "1 W" and 8,190 for "1 M"; daily
bars return 2,512 for "10 Y". Asking for materially more does not error — the
request simply never answers, and the caller sits through a timeout. Refusing
immediately turns a sixty-second hang into an actionable message.

Chunking a longer range is the crawler's job, not this adapter's: one call here
is one request to IBKR, which is also what makes pacing countable."""


class _BarData(Protocol):
    """The fields we read off an `ib_async` historical bar."""

    date: datetime  # the bar's OPEN, with formatDate=2
    open: float
    high: float
    low: float
    close: float
    volume: float
    average: float  # VWAP over the bar
    barCount: int  # trades in the bar


@dataclass(frozen=True, slots=True)
class VwapDrop:
    """How often, and how badly, a symbol's VWAP failed to match its own bar.

    IBKR's `average` is computed at finer precision than the tick-rounded high
    and low it ships alongside, so it can land a fraction of a cent outside the
    bar's range. `worst_excess` is what separates that rounding artefact from a
    feed that is actually wrong: a few thousandths is the former, a whole cent
    or more is worth a human.
    """

    count: int  # bars whose VWAP fell outside [low, high]
    worst_excess: Decimal  # largest distance outside the range, in price units


class MarketDataError(RuntimeError):
    """Raised when a request cannot be made or its answer cannot be trusted.

    Distinct from "no data": an instrument that did not trade returns an empty
    sequence, which is an ordinary outcome during a backfill, not a fault.
    """


class IbkrMarketData:
    """Historical bars from IBKR. Satisfies `MarketDataPort`.

    Example:
        >>> from neurotrade.config import IbkrSettings
        >>> from neurotrade.core.clock import LiveClock
        >>> feed = IbkrMarketData(IbkrConnection(IbkrSettings()), LiveClock())
        >>> feed.pacer.headroom > 0
        True
    """

    __slots__ = ("_clock", "_connection", "_pacer", "_use_rth", "_vwap_drops")

    def __init__(
        self,
        connection: IbkrConnection,
        clock: Clock,
        *,
        pacer: HistoricalPacer | None = None,
        use_rth: bool = True,
    ) -> None:
        """Create the feed.

        Args:
            connection: An `IbkrConnection`. Opened on first use if needed.
            clock: Drives pacing and stamps `ts_init`.
            pacer: Rate limiter. A default one is built when omitted.
            use_rth: Regular trading hours only. True by default because
                extended-hours bars are thin and wide, and mixing them into a
                session silently changes what a volume or range feature means.
        """
        self._connection = connection
        self._clock = clock
        self._pacer = pacer if pacer is not None else HistoricalPacer(clock)
        self._use_rth = use_rth
        self._vwap_drops: dict[Symbol, VwapDrop] = {}

    def vwap_drops(self) -> Mapping[Symbol, VwapDrop]:
        """Bars whose VWAP was discarded for falling outside their own range.

        Never empty for free: either IBKR's rounding is showing or the feed
        disagrees with itself, and the second case wants a human. See `_to_bar`
        for why the bar is kept and only the VWAP is dropped.

        Example:
            >>> from neurotrade.config import IbkrSettings
            >>> from neurotrade.core.clock import LiveClock
            >>> feed = IbkrMarketData(IbkrConnection(IbkrSettings()), LiveClock())
            >>> feed.vwap_drops()
            {}
        """
        return dict(self._vwap_drops)

    @property
    def pacer(self) -> HistoricalPacer:
        """The rate limiter, exposed so a crawler can report headroom."""
        return self._pacer

    async def is_connected(self) -> bool:
        """Whether the feed is usable right now.

        Checked by §6.2's circuit breakers: a disconnected feed means stale
        prices, and trading on stale prices is worse than not trading.
        """
        return self._connection.is_connected

    async def fetch_bars(
        self,
        symbol: Symbol,
        interval: BarInterval,
        start: Nanos,
        end: Nanos,
    ) -> Sequence[Bar]:
        """Fetch historical bars for one instrument.

        Args:
            symbol: Instrument to fetch.
            interval: Bar size.
            start: Inclusive lower bound on `ts_event` (bar close).
            end: Exclusive upper bound on `ts_event`.

        Returns:
            Bars in ascending `ts_event` order, each stamped at its **close**.
            Empty when the instrument did not trade in the range — a holiday, a
            halt, or a listing that did not exist yet. That is an ordinary
            outcome, not an error.

        Raises:
            MarketDataError: If the instrument cannot be resolved, the interval
                is unsupported, IBKR refuses the request, or it goes unanswered
                for `request_timeout_seconds`.
        """
        if interval not in _BAR_SIZE:
            raise MarketDataError(f"no IBKR bar size for {interval.value}")
        if end <= start:
            return ()

        await self._connection.connect()
        contract = await self._qualify(symbol)

        # Built before pacing: a range we will refuse should not consume quota
        # or make the caller wait for a request that is never sent.
        duration = self._duration(interval, start, end)

        wait = self._pacer.wait_seconds()
        if wait > 0:
            # Waiting is cheaper than a pacing violation, which locks historical
            # requests out entirely rather than returning a retryable error.
            await asyncio.sleep(wait)

        self._pacer.record()
        timeout = self._connection.settings.request_timeout_seconds
        sent_at = self._clock.now_ns()
        try:
            # ib_async's own timeout rather than asyncio.wait_for: on expiry it
            # cancels the request at IBKR, where cancelling our coroutine would
            # leave it running there, holding one of the few concurrent slots.
            raw = await self._connection.ib.reqHistoricalDataAsync(
                contract,
                endDateTime=to_datetime(end),
                durationStr=duration,
                barSizeSetting=_BAR_SIZE[interval],
                whatToShow="TRADES",
                useRTH=self._use_rth,
                formatDate=2,  # UTC-aware datetimes rather than local strings
                timeout=timeout,
            )
        except Exception as error:
            raise MarketDataError(f"historical request failed for {symbol}: {error}") from error

        # The price of that choice: a timeout comes back as an empty list, the
        # same as a day with no trades. Only the elapsed time tells them apart.
        if not raw and self._clock.now_ns() - sent_at >= int(timeout * 1_000_000_000):
            raise MarketDataError(
                f"historical request for {symbol} went unanswered for {timeout}s — "
                f"is Gateway connected to its data farms?"
            )

        received_at = self._clock.now_ns()
        bars = [
            self._to_bar(entry, symbol, interval, received_at) for entry in raw if entry is not None
        ]
        # IBKR returns whole bars overlapping the range, so trim to what was
        # asked for rather than handing back bars outside it.
        return tuple(bar for bar in bars if start <= bar.ts_event < end)

    async def _qualify(self, symbol: Symbol) -> Stock:
        """Resolve a `Symbol` into an IBKR contract.

        Qualifying is not optional: an unqualified contract can match several
        listings, and IBKR then picks one. Which one is not something to leave
        to chance when the alternatives differ in currency and price.

        Raises:
            MarketDataError: If the instrument does not resolve, resolves
                ambiguously, or the lookup goes unanswered.
        """
        exchange = IBKR_EXCHANGE.get(symbol.venue)
        if exchange is None:
            raise MarketDataError(f"no IBKR exchange mapped for {symbol.venue.value}")

        stock = Stock(
            symbol.ticker,
            "SMART",  # route through SMART, but pin the listing below
            symbol.currency.value,
            primaryExchange=exchange,
        )
        timeout = self._connection.settings.request_timeout_seconds
        try:
            qualified = await asyncio.wait_for(
                self._connection.ib.qualifyContractsAsync(stock), timeout
            )
        except TimeoutError as error:
            # Contract lookups are not paced and hold no concurrent slot, so
            # abandoning one locally costs nothing at IBKR.
            raise MarketDataError(
                f"resolving {symbol} went unanswered for {timeout}s — "
                f"is Gateway connected to its data farms?"
            ) from error
        except Exception as error:
            raise MarketDataError(f"could not resolve {symbol}: {error}") from error

        if not qualified:
            raise MarketDataError(f"{symbol} did not resolve to any IBKR contract")
        resolved: Stock = qualified[0]
        return resolved

    def _to_bar(
        self,
        entry: _BarData,
        symbol: Symbol,
        interval: BarInterval,
        received_at: Nanos,
    ) -> Bar:
        """Convert one IBKR bar, shifting its timestamp to the close.

        The shift is the important line. IBKR stamps a bar at its open; a
        `Bar` is stamped at the moment it became observable, which is its close.
        Storing IBKR's timestamp unchanged would let a strategy act on a bar one
        interval before it finished forming.

        A VWAP that contradicts the bar's own range is dropped here rather than
        allowed to reject the bar — see `_checked_vwap`.
        """
        opened_at = int(entry.date.replace(tzinfo=entry.date.tzinfo or UTC).timestamp())
        close_ns = (opened_at * 1_000_000_000) + interval.nanos

        high = Price.from_float(entry.high)
        low = Price.from_float(entry.low)
        # `average` is IBKR's VWAP for the bar; zero means it did not trade.
        vwap = Price.from_float(entry.average) if entry.average > 0 else None

        return Bar(
            symbol=symbol,
            ts_event=close_ns,
            ts_init=received_at,
            interval=interval,
            open=Price.from_float(entry.open),
            high=high,
            low=low,
            close=Price.from_float(entry.close),
            volume=Quantity.from_float(entry.volume),
            vwap=self._checked_vwap(symbol, vwap, low=low, high=high),
            trade_count=entry.barCount if entry.barCount >= 0 else None,
        )

    def _checked_vwap(
        self, symbol: Symbol, vwap: Price | None, *, low: Price, high: Price
    ) -> Price | None:
        """Drop a VWAP that falls outside its own bar, and record that it did.

        `Bar` rejects a VWAP outside `[low, high]`, and rightly so. IBKR ships
        one anyway: `average` carries more decimals than the tick-rounded high
        and low, so it can sit a few thousandths above the high. Letting that
        raise cost a whole session of bars — 390 for JPM on 2026-09-15 — over a
        field nothing in the corpus needs yet.

        The VWAP is dropped rather than clamped to the bound: a clamped price is
        a number no trade printed, and `vwap` is already optional. What was
        dropped is counted in `vwap_drops` so the scale of it stays visible.
        """
        if vwap is None or low <= vwap <= high:
            return vwap

        excess = vwap.value - high.value if vwap > high else low.value - vwap.value
        seen = self._vwap_drops.get(symbol)
        self._vwap_drops[symbol] = VwapDrop(
            count=1 if seen is None else seen.count + 1,
            worst_excess=excess if seen is None else max(excess, seen.worst_excess),
        )
        return None

    def _duration(self, interval: BarInterval, start: Nanos, end: Nanos) -> str:
        """IBKR's duration string covering the requested range.

        Whole days, rounded up: IBKR rejects durations outside its accepted
        units, and asking for slightly more than needed is safe because the
        result is trimmed to the range afterwards.

        Raises:
            MarketDataError: If the range exceeds what one request may cover.
                IBKR does not reject an oversized request — it never answers it,
                so the caller waits out a timeout and learns nothing. Naming the
                limit and telling the caller to chunk is more useful.
        """
        seconds = (end - start) / 1_000_000_000
        days = max(1, -(-int(seconds) // _SECONDS_PER_DAY))
        limit = _MAX_DAYS[interval]
        if days > limit:
            raise MarketDataError(
                f"{days} days of {interval.value} bars exceeds the {limit} day limit "
                f"for one IBKR request; fetch it in chunks"
            )
        return f"{days} D"

    def __repr__(self) -> str:
        return f"IbkrMarketData({self._connection!r})"
