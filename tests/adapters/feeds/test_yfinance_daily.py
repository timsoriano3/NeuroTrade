"""Tests for `YFinanceDailyFeed` and its downloader.

Nothing here reaches Yahoo. Rows are hand-written `DailyRow`s, and the one test
that exercises the real `YahooDownloader` replaces `yfinance.Ticker` with a fake
so the call's shape — the exclusive end date, `auto_adjust=False` — is pinned
without a network.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import ClassVar

import pytest

from neurotrade.adapters.calendar.venue_calendar import VenueCalendar
from neurotrade.adapters.feeds.errors import FeedError
from neurotrade.adapters.feeds.yfinance_daily import (
    DailyRow,
    YahooDownloader,
    YFinanceDailyFeed,
    yahoo_ticker,
)
from neurotrade.core.clock import SimClock, to_nanos
from neurotrade.core.events import Bar, BarInterval
from neurotrade.core.ports import MarketDataPort
from neurotrade.core.types import Price, Quantity, Symbol, Venue

AAPL = Symbol("AAPL", Venue.NASDAQ)
SHOP = Symbol("SHOP", Venue.TSX)
FOREVER = (0, 2_000_000_000_000_000_000)

# A full NASDAQ session and the Independence Day holiday beside it.
JULY_2 = date(2024, 7, 2)
JULY_4 = date(2024, 7, 4)


class _Recorder:
    """A downloader returning fixed rows and counting how often it was asked."""

    def __init__(self, rows: Sequence[DailyRow]) -> None:
        self.rows = tuple(rows)
        self.calls: list[tuple[str, date, date]] = []

    def __call__(self, ticker: str, *, start: date, end: date) -> Sequence[DailyRow]:
        self.calls.append((ticker, start, end))
        return self.rows


def _feed(
    rows: Sequence[DailyRow],
    *,
    clock: SimClock | None = None,
    start: date = date(2024, 7, 1),
    end: date = date(2024, 7, 5),
) -> tuple[YFinanceDailyFeed, _Recorder]:
    downloader = _Recorder(rows)
    feed = YFinanceDailyFeed(
        VenueCalendar(),
        clock if clock is not None else SimClock(0),
        start=start,
        end=end,
        downloader=downloader,
    )
    return feed, downloader


def _fetch(feed: YFinanceDailyFeed, symbol: Symbol = AAPL) -> Sequence[Bar]:
    return asyncio.run(feed.fetch_bars(symbol, BarInterval.DAY_1, *FOREVER))


# ── Conformance ──────────────────────────────────────────────


def test_satisfies_the_market_data_port() -> None:
    feed, _ = _feed(())
    assert isinstance(feed, MarketDataPort)


def test_is_connected_is_always_true() -> None:
    feed, _ = _feed(())
    assert asyncio.run(feed.is_connected()) is True


# ── Yahoo's names for our symbols ────────────────────────────


@pytest.mark.parametrize(
    ("symbol", "expected"),
    [
        (Symbol("AAPL", Venue.NASDAQ), "AAPL"),
        (Symbol("SPY", Venue.ARCA), "SPY"),
        (Symbol("TD", Venue.NYSE), "TD"),
        (Symbol("TD", Venue.TSX), "TD.TO"),
        (Symbol("SHOP", Venue.TSX), "SHOP.TO"),
        (Symbol("WELL", Venue.TSXV), "WELL.V"),
        # Yahoo writes share classes and preferreds with a hyphen.
        (Symbol("BRK.B", Venue.NYSE), "BRK-B"),
        (Symbol("BCE.PR.A", Venue.TSX), "BCE-PR-A.TO"),
    ],
)
def test_yahoo_ticker(symbol: Symbol, expected: str) -> None:
    assert yahoo_ticker(symbol) == expected


def test_the_downloader_is_asked_for_yahoos_name_not_ours() -> None:
    feed, downloader = _feed(())
    _fetch(feed, SHOP)
    assert downloader.calls == [("SHOP.TO", date(2024, 7, 1), date(2024, 7, 5))]


# ── The daily bar is stamped at the session close ────────────


def test_a_row_is_stamped_at_the_venue_close_not_at_midnight() -> None:
    feed, _ = _feed([DailyRow(JULY_2, 100.0, 102.0, 99.0, 101.0, 1_000.0)])
    (bar,) = _fetch(feed)
    assert bar.ts_event == to_nanos(datetime(2024, 7, 2, 20, 0, tzinfo=UTC))
    assert bar.interval is BarInterval.DAY_1


def test_an_early_close_moves_the_stamp_with_it() -> None:
    # 2024-07-03 was a 13:00 ET half day ahead of Independence Day. The stamp
    # follows the calendar, which is the whole reason it comes from there.
    feed, _ = _feed([DailyRow(date(2024, 7, 3), 100.0, 102.0, 99.0, 101.0, 1_000.0)])
    (bar,) = _fetch(feed)
    assert bar.ts_event == to_nanos(datetime(2024, 7, 3, 17, 0, tzinfo=UTC))


def test_a_toronto_row_is_stamped_on_torontos_own_close() -> None:
    feed, _ = _feed([DailyRow(JULY_2, 73.0, 75.0, 72.0, 74.0, 4_000.0)])
    (bar,) = _fetch(feed, SHOP)
    assert bar.ts_event == to_nanos(datetime(2024, 7, 2, 20, 0, tzinfo=UTC))
    assert bar.symbol == SHOP


# ── Conversion into domain types ─────────────────────────────


def test_prices_and_volume_become_exact_decimals() -> None:
    feed, _ = _feed([DailyRow(JULY_2, 172.02, 173.07, 170.33, 171.21, 51_861_100.0)])
    (bar,) = _fetch(feed)
    assert (bar.open, bar.high, bar.low, bar.close) == (
        Price(Decimal("172.02")),
        Price(Decimal("173.07")),
        Price(Decimal("170.33")),
        Price(Decimal("171.21")),
    )
    assert bar.volume == Quantity(Decimal("51861100"))


def test_ts_init_comes_from_the_clock_at_each_call_not_from_the_download() -> None:
    clock = SimClock(0)
    feed, _ = _feed([DailyRow(JULY_2, 100.0, 102.0, 99.0, 101.0, 1_000.0)], clock=clock)

    clock.advance_ns(5_000)
    (first,) = _fetch(feed)
    clock.advance_ns(7_000)
    (second,) = _fetch(feed)

    assert (first.ts_init, second.ts_init) == (5_000, 12_000)
    assert first.ts_event == second.ts_event


# ── One download per symbol, sliced per cell ─────────────────


def test_the_symbol_is_downloaded_once_however_many_cells_ask() -> None:
    feed, downloader = _feed([DailyRow(JULY_2, 100.0, 102.0, 99.0, 101.0, 1_000.0)])
    for _ in range(3):
        _fetch(feed)
    assert len(downloader.calls) == 1


def test_each_cell_gets_only_the_bars_inside_its_window() -> None:
    rows = [
        DailyRow(date(2024, 7, 1), 100.0, 102.0, 99.0, 101.0, 1_000.0),
        DailyRow(JULY_2, 101.0, 103.0, 100.0, 102.0, 1_100.0),
        DailyRow(date(2024, 7, 5), 102.0, 104.0, 101.0, 103.0, 1_200.0),
    ]
    feed, _ = _feed(rows)
    july_2_close = to_nanos(datetime(2024, 7, 2, 20, 0, tzinfo=UTC))

    inside = asyncio.run(feed.fetch_bars(AAPL, BarInterval.DAY_1, july_2_close, july_2_close + 1))
    assert [bar.ts_event for bar in inside] == [july_2_close]

    # The lower bound is inclusive and the upper exclusive, as the port says.
    excluded = asyncio.run(feed.fetch_bars(AAPL, BarInterval.DAY_1, july_2_close + 1, FOREVER[1]))
    assert july_2_close not in [bar.ts_event for bar in excluded]


def test_bars_come_back_in_ascending_order_whatever_order_yahoo_gave() -> None:
    rows = [
        DailyRow(date(2024, 7, 5), 102.0, 104.0, 101.0, 103.0, 1_200.0),
        DailyRow(date(2024, 7, 1), 100.0, 102.0, 99.0, 101.0, 1_000.0),
        DailyRow(JULY_2, 101.0, 103.0, 100.0, 102.0, 1_100.0),
    ]
    feed, _ = _feed(rows)
    stamps = [bar.ts_event for bar in _fetch(feed)]
    assert stamps == sorted(stamps)


# ── Rejection ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "interval", [BarInterval.MIN_1, BarInterval.MIN_5, BarInterval.HOUR_1, BarInterval.SEC_1]
)
def test_only_daily_bars_are_served(interval: BarInterval) -> None:
    feed, _ = _feed(())
    with pytest.raises(FeedError, match=r"only has 1d bars"):
        asyncio.run(feed.fetch_bars(AAPL, interval, *FOREVER))


def test_a_backwards_range_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match=r"range runs backwards: 2024-07-05 to 2024-07-01"):
        YFinanceDailyFeed(
            VenueCalendar(),
            SimClock(0),
            start=date(2024, 7, 5),
            end=date(2024, 7, 1),
        )


def test_a_row_on_a_day_the_venue_was_shut_is_dropped_and_reported() -> None:
    rows = [
        DailyRow(JULY_2, 100.0, 102.0, 99.0, 101.0, 1_000.0),
        DailyRow(JULY_4, 100.0, 102.0, 99.0, 101.0, 1_000.0),  # Independence Day
    ]
    feed, _ = _feed(rows)
    bars = _fetch(feed)

    assert len(bars) == 1
    assert feed.unmatched() == {AAPL: (JULY_4,)}


def test_nothing_is_reported_unmatched_when_every_row_has_a_session() -> None:
    feed, _ = _feed([DailyRow(JULY_2, 100.0, 102.0, 99.0, 101.0, 1_000.0)])
    _fetch(feed)
    assert feed.unmatched() == {}


def test_an_impossible_bar_is_refused_by_the_domain_type() -> None:
    # Close above high. The feed does not re-check what `Bar` already checks;
    # this pins that the check is reached rather than bypassed by the float
    # conversion.
    feed, _ = _feed([DailyRow(JULY_2, 100.0, 102.0, 99.0, 103.0, 1_000.0)])
    with pytest.raises(ValueError, match=r"close 103\.0 outside range \[99\.0, 102\.0\]"):
        _fetch(feed)


# ── The real downloader, with yfinance faked out ─────────────


class _FakeFrame:
    """The slice of a pandas frame `YahooDownloader` actually touches."""

    def __init__(self, stamps: Sequence[datetime], columns: dict[str, Sequence[float]]) -> None:
        self.index = tuple(stamps)
        self._columns = columns

    def __getitem__(self, name: str) -> Sequence[float]:
        return self._columns[name]


class _FakeTicker:
    """Records the call `YahooDownloader` makes and answers with a fixed frame."""

    calls: ClassVar[list[dict[str, object]]] = []
    frame: ClassVar[_FakeFrame | None] = None

    def __init__(self, ticker: str) -> None:
        self.ticker = ticker

    def history(self, **kwargs: object) -> _FakeFrame:
        _FakeTicker.calls.append({"ticker": self.ticker, **kwargs})
        assert _FakeTicker.frame is not None
        return _FakeTicker.frame


@pytest.fixture
def fake_yfinance(monkeypatch: pytest.MonkeyPatch) -> type[_FakeTicker]:
    import yfinance

    _FakeTicker.calls = []
    monkeypatch.setattr(yfinance, "Ticker", _FakeTicker)
    return _FakeTicker


def test_the_downloader_asks_yahoo_for_an_exclusive_end_and_unadjusted_prices(
    fake_yfinance: type[_FakeTicker],
) -> None:
    fake_yfinance.frame = _FakeFrame(
        [datetime(2024, 7, 2, tzinfo=UTC)],
        {
            "Open": [100.0],
            "High": [102.0],
            "Low": [99.0],
            "Close": [101.0],
            "Volume": [1_000.0],
        },
    )

    rows = YahooDownloader(timeout=1.0)("AAPL", start=date(2024, 7, 1), end=date(2024, 7, 2))

    (call,) = fake_yfinance.calls
    assert call["ticker"] == "AAPL"
    assert call["start"] == date(2024, 7, 1)
    # Ours is inclusive, Yahoo's is exclusive: the last session must survive.
    assert call["end"] == date(2024, 7, 3)
    assert call["auto_adjust"] is False
    assert call["interval"] == "1d"
    assert rows == (DailyRow(date(2024, 7, 2), 100.0, 102.0, 99.0, 101.0, 1_000.0),)


def test_a_non_finite_value_from_yahoo_is_refused_rather_than_guessed(
    fake_yfinance: type[_FakeTicker],
) -> None:
    fake_yfinance.frame = _FakeFrame(
        [datetime(2024, 7, 2, tzinfo=UTC)],
        {
            "Open": [100.0],
            "High": [float("nan")],
            "Low": [99.0],
            "Close": [101.0],
            "Volume": [1_000.0],
        },
    )

    with pytest.raises(FeedError, match=r"non-finite values on 2024-07-02"):
        YahooDownloader()("AAPL", start=date(2024, 7, 1), end=date(2024, 7, 2))


def test_a_refusal_from_yahoo_surfaces_as_a_feed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import yfinance

    class _Angry:
        def __init__(self, ticker: str) -> None:
            pass

        def history(self, **kwargs: object) -> None:
            raise RuntimeError("HTTP Error 404")

    monkeypatch.setattr(yfinance, "Ticker", _Angry)

    with pytest.raises(FeedError, match=r"yfinance failed for AAPL: HTTP Error 404"):
        YahooDownloader()("AAPL", start=date(2024, 7, 1), end=date(2024, 7, 2))


# ── Yahoo's float noise ──────────────────────────────────────


def test_a_float64_close_is_rounded_back_to_the_price_that_traded() -> None:
    """Yahoo sends $169.34 as 169.33999633789062. Kept whole, that is fourteen
    decimal places — more than the corpus column holds, and never equal to
    IBKR's own 169.34 when the two are cross-checked."""
    feed, _ = _feed(
        [
            DailyRow(
                JULY_2,
                169.33999633789062,
                172.02999877929688,
                167.61999511718750,
                170.69000244140625,
                56294400.0,
            )
        ]
    )
    (bar,) = _fetch(feed)

    assert (bar.open, bar.high, bar.low, bar.close) == (
        Price(Decimal("169.34")),
        Price(Decimal("172.03")),
        Price(Decimal("167.62")),
        Price(Decimal("170.69")),
    )
    assert bar.volume == Quantity(Decimal("56294400"))


def test_a_sub_dollar_tick_survives_the_rounding() -> None:
    # Below $1.00 the minimum tick is $0.0001, so four places must be kept.
    feed, _ = _feed([DailyRow(JULY_2, 0.0032, 0.0035, 0.0031, 0.0034, 1_000.0)])
    (bar,) = _fetch(feed)
    assert bar.close == Price(Decimal("0.0034"))
