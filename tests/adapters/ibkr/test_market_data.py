"""Tests for the IBKR historical bar feed.

The timestamp test is the one that matters. IBKR stamps a bar at its open and we
stamp at its close, so a missing shift makes every strategy act one bar early —
systematically, invisibly, and in a way no test of the strategy itself would
catch.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest
from pydantic import ValidationError

from neurotrade.adapters.ibkr.connection import IbkrConnection
from neurotrade.adapters.ibkr.market_data import (
    IBKR_EXCHANGE,
    IbkrMarketData,
    MarketDataError,
    VwapDrop,
)
from neurotrade.adapters.ibkr.pacing import HistoricalPacer
from neurotrade.config import IbkrSettings
from neurotrade.core.clock import SimClock
from neurotrade.core.events import Bar, BarInterval
from neurotrade.core.ports import MarketDataPort
from neurotrade.core.types import Price, Symbol, Venue
from tests.adapters.ibkr.conftest import FakeBarData, FakeContract, FakeIB, a_bar

AAPL = Symbol("AAPL", Venue.NASDAQ)
SHOP = Symbol("SHOP", Venue.TSX)

MINUTE = 60_000_000_000
OPEN_NS = int(datetime(2026, 3, 16, 13, 30, tzinfo=UTC).timestamp()) * 1_000_000_000
FOREVER = OPEN_NS + 1_000 * MINUTE


def a_feed(**fake: object) -> tuple[IbkrMarketData, FakeIB]:
    ib = FakeIB(**fake)  # type: ignore[arg-type]
    connection = IbkrConnection(IbkrSettings(), ib=ib)
    return IbkrMarketData(connection, SimClock(OPEN_NS)), ib


# ── Conformance ──────────────────────────────────────────────


def test_satisfies_the_market_data_port() -> None:
    feed, _ = a_feed()
    assert isinstance(feed, MarketDataPort)


# ── The timestamp shift: what this module exists to get right ─


async def test_a_bar_is_stamped_at_its_close_not_its_open() -> None:
    """IBKR stamps at the open. Storing that unchanged is a one-bar lookahead."""
    feed, _ = a_feed(bars=[a_bar(0)])
    bars = await feed.fetch_bars(AAPL, BarInterval.MIN_1, OPEN_NS, FOREVER)
    assert len(bars) == 1
    # IBKR said 13:30 (the open); the bar became observable at 13:31.
    assert bars[0].ts_event == OPEN_NS + MINUTE
    assert bars[0].ts_open == OPEN_NS


async def test_the_shift_follows_the_interval() -> None:
    """A five-minute bar becomes observable five minutes after it opens."""
    feed, _ = a_feed(bars=[a_bar(0)])
    bars = await feed.fetch_bars(AAPL, BarInterval.MIN_5, OPEN_NS, FOREVER)
    assert bars[0].ts_event == OPEN_NS + 5 * MINUTE


async def test_a_full_session_ends_at_the_close() -> None:
    """A 390-bar US session runs 13:30-19:59 at the open, 13:31-20:00 at close.

    The last bar landing on 20:00 rather than 19:59 is the whole point.
    """
    feed, _ = a_feed(bars=[a_bar(i) for i in range(390)])
    bars = await feed.fetch_bars(AAPL, BarInterval.MIN_1, OPEN_NS, FOREVER)
    assert len(bars) == 390
    assert bars[0].ts_event == OPEN_NS + MINUTE
    assert bars[-1].ts_event == OPEN_NS + 390 * MINUTE  # 20:00 UTC


# ── Conversion ───────────────────────────────────────────────


async def test_prices_arrive_exact() -> None:
    """Floats in, decimals out, via the marked from_float boundary."""
    feed, _ = a_feed(bars=[a_bar(0, close=315.93)])
    bars = await feed.fetch_bars(AAPL, BarInterval.MIN_1, OPEN_NS, FOREVER)
    assert bars[0].close.value == Decimal("315.93")
    assert isinstance(bars[0].close, Price)


async def test_vwap_and_trade_count_are_carried_through() -> None:
    feed, _ = a_feed(bars=[a_bar(0, close=100.4, bar_count=1779)])
    bars = await feed.fetch_bars(AAPL, BarInterval.MIN_1, OPEN_NS, FOREVER)
    assert bars[0].vwap == Price("100.4")
    assert bars[0].trade_count == 1779


async def test_a_bar_that_did_not_trade_has_no_vwap() -> None:
    """IBKR reports average=0 for an empty bar. Zero is not a price."""
    feed, _ = a_feed(bars=[a_bar(0, average=0.0, volume=0.0)])
    bars = await feed.fetch_bars(AAPL, BarInterval.MIN_1, OPEN_NS, FOREVER)
    assert bars[0].vwap is None
    assert bars[0].volume.is_zero


# ── A VWAP that contradicts its own bar ──────────────────────
#
# Live, 2026-09-15: IBKR returned JPM with average=344.576 against a high of
# 344.57, `Bar` refused it, and the whole 390-bar session was lost. These bars
# are built by hand because `a_bar` pins average to close, which is the one
# thing that cannot be true here.


def a_bar_with_vwap(minute: int, *, average: float, high: float, low: float) -> FakeBarData:
    """A bar whose VWAP is set independently of its range."""
    return FakeBarData(
        date=datetime(2026, 3, 16, 13, 30 + minute, tzinfo=UTC),
        open=low,
        high=high,
        low=low,
        close=high,
        volume=1000.0,
        average=average,
        barCount=42,
    )


async def test_a_vwap_above_the_high_is_dropped_rather_than_rejecting_the_bar() -> None:
    feed, _ = a_feed(bars=[a_bar_with_vwap(0, average=344.576, high=344.57, low=344.43)])
    bars = await feed.fetch_bars(AAPL, BarInterval.MIN_1, OPEN_NS, FOREVER)
    assert len(bars) == 1
    assert bars[0].vwap is None
    assert bars[0].high == Price("344.57")


async def test_a_vwap_below_the_low_is_dropped_too() -> None:
    feed, _ = a_feed(bars=[a_bar_with_vwap(0, average=344.42, high=344.57, low=344.43)])
    bars = await feed.fetch_bars(AAPL, BarInterval.MIN_1, OPEN_NS, FOREVER)
    assert bars[0].vwap is None


async def test_one_bad_vwap_does_not_cost_the_rest_of_the_session() -> None:
    """The regression: a single contradictory field used to lose every bar."""
    feed, _ = a_feed(
        bars=[
            a_bar(0),
            a_bar_with_vwap(1, average=344.576, high=344.57, low=344.43),
            a_bar(2),
        ]
    )
    bars = await feed.fetch_bars(AAPL, BarInterval.MIN_1, OPEN_NS, FOREVER)
    assert len(bars) == 3
    assert [bar.vwap is None for bar in bars] == [False, True, False]


async def test_a_dropped_vwap_is_counted_with_its_worst_excess() -> None:
    feed, _ = a_feed(
        bars=[
            a_bar_with_vwap(0, average=344.576, high=344.57, low=344.43),
            a_bar_with_vwap(1, average=344.60, high=344.57, low=344.43),
        ]
    )
    await feed.fetch_bars(AAPL, BarInterval.MIN_1, OPEN_NS, FOREVER)
    assert feed.vwap_drops() == {AAPL: VwapDrop(count=2, worst_excess=Decimal("0.03"))}


async def test_a_vwap_inside_the_range_is_carried_through_and_counted_nowhere() -> None:
    feed, _ = a_feed(bars=[a_bar_with_vwap(0, average=344.50, high=344.57, low=344.43)])
    bars = await feed.fetch_bars(AAPL, BarInterval.MIN_1, OPEN_NS, FOREVER)
    assert bars[0].vwap == Price("344.5")
    assert feed.vwap_drops() == {}


async def test_a_vwap_exactly_on_a_bound_is_not_a_drop() -> None:
    feed, _ = a_feed(bars=[a_bar_with_vwap(0, average=344.57, high=344.57, low=344.43)])
    bars = await feed.fetch_bars(AAPL, BarInterval.MIN_1, OPEN_NS, FOREVER)
    assert bars[0].vwap == Price("344.57")
    assert feed.vwap_drops() == {}


async def test_ts_init_records_when_we_received_it() -> None:
    """ts_init - ts_event is our data latency, and comes from the clock."""
    feed, _ = a_feed(bars=[a_bar(0)])
    bars = await feed.fetch_bars(AAPL, BarInterval.MIN_1, OPEN_NS, FOREVER)
    assert bars[0].ts_init == OPEN_NS  # the SimClock's position


# ── Venue mapping ────────────────────────────────────────────


def test_tsx_maps_to_ibkrs_own_name() -> None:
    """A wrong exchange does not error — it qualifies a different listing."""
    assert IBKR_EXCHANGE[Venue.TSX] == "TSE"
    assert IBKR_EXCHANGE[Venue.TSXV] == "VENTURE"


def test_us_venues_pass_through_unchanged() -> None:
    for venue in (Venue.NASDAQ, Venue.NYSE, Venue.ARCA):
        assert IBKR_EXCHANGE[venue] == venue.value


def test_smart_is_not_a_listing_venue() -> None:
    """It is a router; `Symbol` refuses it, so it needs no mapping."""
    assert Venue.SMART not in IBKR_EXCHANGE


async def test_the_contract_is_pinned_to_the_listing() -> None:
    """Routing goes through SMART; the listing is pinned by primaryExchange."""
    feed, ib = a_feed(bars=[a_bar(0)], contracts=[FakeContract(symbol="SHOP")])
    await feed.fetch_bars(SHOP, BarInterval.MIN_1, OPEN_NS, FOREVER)
    assert ib.historical_calls  # it got as far as requesting


# ── Range handling ───────────────────────────────────────────


async def test_bars_outside_the_range_are_trimmed() -> None:
    """IBKR returns whole bars overlapping the range, not exactly the range."""
    feed, _ = a_feed(bars=[a_bar(i) for i in range(10)])
    bars = await feed.fetch_bars(
        AAPL, BarInterval.MIN_1, OPEN_NS + 3 * MINUTE, OPEN_NS + 6 * MINUTE
    )
    assert [b.ts_event for b in bars] == [OPEN_NS + i * MINUTE for i in (3, 4, 5)]


async def test_an_empty_range_asks_for_nothing() -> None:
    feed, ib = a_feed(bars=[a_bar(0)])
    assert await feed.fetch_bars(AAPL, BarInterval.MIN_1, FOREVER, OPEN_NS) == ()
    assert not ib.historical_calls  # no request was made


async def test_no_data_is_not_an_error() -> None:
    """A holiday, a halt, or a listing that did not exist yet."""
    feed, _ = a_feed(bars=[])
    assert await feed.fetch_bars(AAPL, BarInterval.MIN_1, OPEN_NS, FOREVER) == ()


# ── Failure modes ────────────────────────────────────────────


async def test_an_unresolvable_instrument_raises() -> None:
    feed, _ = a_feed(contracts=[])
    with pytest.raises(MarketDataError, match="did not resolve"):
        await feed.fetch_bars(AAPL, BarInterval.MIN_1, OPEN_NS, FOREVER)


async def test_a_refused_request_names_the_instrument() -> None:
    feed, _ = a_feed(historical_error=RuntimeError("pacing violation"))
    with pytest.raises(MarketDataError, match=r"AAPL\.NASDAQ"):
        await feed.fetch_bars(AAPL, BarInterval.MIN_1, OPEN_NS, FOREVER)


# ── Timeouts ─────────────────────────────────────────────────


async def test_the_history_request_carries_the_configured_timeout() -> None:
    ib = FakeIB(bars=[a_bar(0)])
    connection = IbkrConnection(IbkrSettings(request_timeout_seconds=12.5), ib=ib)
    await IbkrMarketData(connection, SimClock(OPEN_NS)).fetch_bars(
        AAPL, BarInterval.MIN_1, OPEN_NS, FOREVER
    )
    assert ib.historical_calls[0]["timeout"] == 12.5


def _timing_out_feed(bars: list[FakeBarData]) -> IbkrMarketData:
    """A feed whose history request takes exactly the whole timeout, as
    `ib_async` does when it gives up waiting."""
    clock = SimClock(OPEN_NS)

    def elapse(kwargs: dict[str, Any]) -> None:
        clock.advance_ns(int(kwargs["timeout"] * 1_000_000_000))

    ib = FakeIB(bars=bars, on_historical=elapse)
    return IbkrMarketData(IbkrConnection(IbkrSettings(), ib=ib), clock)


async def test_a_history_request_that_times_out_is_an_error_not_an_empty_day() -> None:
    """ib_async answers a timeout with no bars. Recording that as "nothing
    traded" would let a crawler with a dead feed report every session empty."""
    with pytest.raises(MarketDataError, match="went unanswered"):
        await _timing_out_feed([]).fetch_bars(AAPL, BarInterval.MIN_1, OPEN_NS, FOREVER)


async def test_bars_that_arrive_slowly_are_still_bars() -> None:
    """Elapsed time alone is not a timeout; only elapsed time with no answer."""
    bars = await _timing_out_feed([a_bar(0)]).fetch_bars(AAPL, BarInterval.MIN_1, OPEN_NS, FOREVER)
    assert len(bars) == 1


async def test_an_unanswered_contract_lookup_is_an_error() -> None:
    ib = FakeIB(qualify_hangs=True)
    connection = IbkrConnection(IbkrSettings(request_timeout_seconds=0.05), ib=ib)
    feed = IbkrMarketData(connection, SimClock(OPEN_NS))
    with pytest.raises(MarketDataError, match=r"resolving AAPL\.NASDAQ went unanswered"):
        await feed.fetch_bars(AAPL, BarInterval.MIN_1, OPEN_NS, FOREVER)
    assert not ib.historical_calls  # never got as far as asking for bars


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_a_request_timeout_must_be_positive(bad: float) -> None:
    """Zero would disable the timeout, which is the hang this setting exists to end."""
    with pytest.raises(ValidationError):
        IbkrSettings(request_timeout_seconds=bad)


# ── Pacing ───────────────────────────────────────────────────


async def test_every_request_is_counted() -> None:
    """IBKR locks out rather than erroring, so the count must be kept."""
    feed, _ = a_feed(bars=[a_bar(0)])
    assert feed.pacer.in_window == 0
    await feed.fetch_bars(AAPL, BarInterval.MIN_1, OPEN_NS, FOREVER)
    assert feed.pacer.in_window == 1


def test_the_window_slides_rather_than_resetting() -> None:
    """A fixed window would allow double the limit across a boundary."""
    clock = SimClock(0)
    pacer = HistoricalPacer(clock, max_requests=2, window_seconds=10)
    pacer.record()
    clock.advance_ns(9 * 1_000_000_000)
    pacer.record()
    assert pacer.wait_seconds() == pytest.approx(1.0)  # until the first ages out
    clock.advance_ns(2 * 1_000_000_000)
    assert pacer.wait_seconds() == 0.0
    assert pacer.in_window == 1


def test_headroom_is_reported() -> None:
    clock = SimClock(0)
    pacer = HistoricalPacer(clock, max_requests=3, window_seconds=10)
    pacer.record()
    assert pacer.headroom == 2


def test_the_default_stays_under_ibkrs_documented_limit() -> None:
    """Their limit is 60 per ten minutes, enforced on their clock, not ours."""
    from neurotrade.adapters.ibkr.pacing import DEFAULT_MAX_REQUESTS

    assert DEFAULT_MAX_REQUESTS < 60


@pytest.mark.parametrize("bad", [0, -1])
def test_a_nonsensical_limit_is_rejected(bad: int) -> None:
    with pytest.raises(ValueError):
        HistoricalPacer(SimClock(0), max_requests=bad)


# ── Interval support ─────────────────────────────────────────


@pytest.mark.parametrize("interval", list(BarInterval))
async def test_every_interval_is_supported(interval: BarInterval) -> None:
    """A missing bar size would fail only for the interval nobody tested."""
    feed, _ = a_feed(bars=[a_bar(0)])
    await feed.fetch_bars(AAPL, interval, OPEN_NS, OPEN_NS + 6 * 60 * MINUTE)


# ── Request size ─────────────────────────────────────────────


async def test_an_oversized_range_is_refused_immediately() -> None:
    """IBKR does not reject an oversized request — it never answers it.

    Found by running against a live Gateway: a range of a few decades made the
    call sit through a sixty-second timeout and report nothing useful.
    """
    feed, ib = a_feed(bars=[a_bar(0)])
    a_decade = OPEN_NS + 3_650 * 86_400 * 1_000_000_000
    with pytest.raises(MarketDataError, match="exceeds the 30 day limit"):
        await feed.fetch_bars(AAPL, BarInterval.MIN_1, OPEN_NS, a_decade)
    assert not ib.historical_calls  # nothing was sent


async def test_a_refused_range_costs_no_pacing_quota() -> None:
    """Otherwise a caller with a bad range burns the rate limit learning that."""
    feed, _ = a_feed(bars=[a_bar(0)])
    a_decade = OPEN_NS + 3_650 * 86_400 * 1_000_000_000
    with pytest.raises(MarketDataError):
        await feed.fetch_bars(AAPL, BarInterval.MIN_1, OPEN_NS, a_decade)
    assert feed.pacer.in_window == 0


async def test_daily_bars_may_cover_years() -> None:
    """The limit is per bar size: a decade of daily bars is one request."""
    feed, _ = a_feed(bars=[a_bar(0)])
    nine_years = OPEN_NS + 3_285 * 86_400 * 1_000_000_000
    await feed.fetch_bars(AAPL, BarInterval.DAY_1, OPEN_NS, nine_years)


# ── Against a real Gateway ───────────────────────────────────


@pytest.mark.ibkr
async def test_real_bars_come_back_stamped_at_the_close() -> None:
    """The claim this module rests on, checked against IBKR itself.

    Run with `uv run pytest -m ibkr` and Gateway logged in.
    """
    from neurotrade.config import Profile, load_settings

    settings = load_settings(Profile.PAPER).ibkr
    connection = IbkrConnection(IbkrSettings(**{**settings.model_dump(), "client_id": 78}))
    clock = SimClock(OPEN_NS)
    feed = IbkrMarketData(connection, clock)
    try:
        # One trading day, ending now. The range is what a crawler would ask
        # for; anything longer is chunked by the caller.
        now = int(datetime.now(UTC).timestamp()) * 1_000_000_000
        bars = await feed.fetch_bars(AAPL, BarInterval.MIN_1, now - 86_400 * 1_000_000_000, now)
    finally:
        connection.disconnect()

    assert bars, "no bars returned — is the market data subscription shared?"
    assert all(isinstance(bar, Bar) for bar in bars)
    # Consecutive one-minute bars, each stamped one minute after the last.
    gaps = {bars[i + 1].ts_event - bars[i].ts_event for i in range(len(bars) - 1)}
    assert gaps == {MINUTE}, f"unexpected spacing: {sorted(gaps)}"
    # A close-stamped session ends on a minute boundary at or after the close.
    assert bars[-1].ts_event % MINUTE == 0
