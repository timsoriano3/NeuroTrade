"""Tests for `FirstRateFeed`, the file-backed feed over FRD's sample zip.

Rows are hand-written in FRD's own format; nothing here is a real vendor
download (`08-gotchas.doc.md`, `11-seed-data.plan.md` decision 4).
"""

from __future__ import annotations

import zipfile
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from neurotrade.adapters.feeds.errors import FeedError
from neurotrade.adapters.feeds.firstrate import FirstRateFeed
from neurotrade.core.clock import SimClock
from neurotrade.core.events import BarInterval
from neurotrade.core.ports import MarketDataPort
from neurotrade.core.types import Price, Symbol, Venue

AAPL = Symbol("AAPL", Venue.NASDAQ)
FOREVER = (0, 2_000_000_000_000_000_000)
_HEADER = "timestamp,open,high,low,close,volume\n"


def _zip(tmp_path: Path, name: str, rows: list[str], *, header: str | None = _HEADER) -> Path:
    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("AAPL.csv", (header or "") + "".join(rows))
    return path


# ── Conformance ──────────────────────────────────────────────


def test_satisfies_the_market_data_port() -> None:
    assert isinstance(FirstRateFeed({}, SimClock(0)), MarketDataPort)


# ── The open-to-close shift, both sides of DST ────────────────


async def test_a_bar_is_stamped_at_its_close_not_its_open(tmp_path: Path) -> None:
    path = _zip(tmp_path, "AAPL.zip", ["2023-01-03 09:30:00,100,101,99,100.5,1000\n"])
    feed = FirstRateFeed({AAPL: path}, SimClock(0))
    bars = await feed.fetch_bars(AAPL, BarInterval.MIN_1, *FOREVER)
    assert len(bars) == 1
    # 09:30 ET open in January (EST, UTC-5) is 14:30 UTC; the close is a minute later.
    assert bars[0].ts_open == 1_672_756_200_000_000_000
    assert bars[0].ts_event == 1_672_756_260_000_000_000


async def test_dst_summer_bar_shifts_by_the_daylight_offset(tmp_path: Path) -> None:
    """July: ET is EDT (UTC-4), not EST (UTC-5) — the same 09:30 wall-clock
    open lands an hour earlier in UTC than the January case above."""
    path = _zip(tmp_path, "AAPL.zip", ["2023-07-03 09:30:00,100,101,99,100.5,1000\n"])
    feed = FirstRateFeed({AAPL: path}, SimClock(0))
    bars = await feed.fetch_bars(AAPL, BarInterval.MIN_1, *FOREVER)
    assert bars[0].ts_open == 1_688_391_000_000_000_000
    assert bars[0].ts_event == 1_688_391_060_000_000_000


# ── Conversion and ordering ────────────────────────────────────


async def test_prices_are_exact_decimals(tmp_path: Path) -> None:
    path = _zip(tmp_path, "AAPL.zip", ["2023-01-03 09:30:00,100,101,99,100.5,1000\n"])
    feed = FirstRateFeed({AAPL: path}, SimClock(0))
    bars = await feed.fetch_bars(AAPL, BarInterval.MIN_1, *FOREVER)
    assert bars[0].close == Price("100.5")
    assert bars[0].volume.value == Decimal("1000")


async def test_rows_come_back_ascending_regardless_of_file_order(tmp_path: Path) -> None:
    path = _zip(
        tmp_path,
        "AAPL.zip",
        [
            "2023-01-03 09:31:00,101,102,100,101.5,1100\n",
            "2023-01-03 09:30:00,100,101,99,100.5,1000\n",
        ],
    )
    feed = FirstRateFeed({AAPL: path}, SimClock(0))
    bars = await feed.fetch_bars(AAPL, BarInterval.MIN_1, *FOREVER)
    assert [bar.ts_event for bar in bars] == sorted(bar.ts_event for bar in bars)


async def test_range_is_trimmed_to_start_inclusive_end_exclusive(tmp_path: Path) -> None:
    path = _zip(
        tmp_path,
        "AAPL.zip",
        [
            "2023-01-03 09:30:00,100,101,99,100.5,1000\n",
            "2023-01-03 09:31:00,101,102,100,101.5,1100\n",
            "2023-01-03 09:32:00,102,103,101,102.5,1200\n",
        ],
    )
    feed = FirstRateFeed({AAPL: path}, SimClock(0))
    first_close = 1_672_756_260_000_000_000
    second_close = first_close + BarInterval.MIN_1.nanos
    bars = await feed.fetch_bars(AAPL, BarInterval.MIN_1, first_close, second_close + 1)
    assert [bar.ts_event for bar in bars] == [first_close, second_close]


# ── Rejections ─────────────────────────────────────────────────


async def test_unconfigured_symbol_raises(tmp_path: Path) -> None:
    feed = FirstRateFeed({}, SimClock(0))
    with pytest.raises(FeedError, match="no firstrate file configured"):
        await feed.fetch_bars(AAPL, BarInterval.MIN_1, *FOREVER)


async def test_unsupported_interval_raises(tmp_path: Path) -> None:
    path = _zip(tmp_path, "AAPL.zip", ["2023-01-03 09:30:00,100,101,99,100.5,1000\n"])
    feed = FirstRateFeed({AAPL: path}, SimClock(0))
    with pytest.raises(FeedError, match="only has 1m bars"):
        await feed.fetch_bars(AAPL, BarInterval.MIN_5, *FOREVER)


async def test_wrong_header_raises(tmp_path: Path) -> None:
    path = _zip(
        tmp_path,
        "AAPL.zip",
        ["2023-01-03 09:30:00,100,101,99,100.5,1000\n"],
        header="date,open,high,low,close,vol\n",
    )
    feed = FirstRateFeed({AAPL: path}, SimClock(0))
    with pytest.raises(FeedError, match="unexpected header"):
        await feed.fetch_bars(AAPL, BarInterval.MIN_1, *FOREVER)


async def test_bad_row_is_rejected(tmp_path: Path) -> None:
    path = _zip(tmp_path, "AAPL.zip", ["2023-01-03 09:30:00,not-a-number,101,99,100.5,1000\n"])
    feed = FirstRateFeed({AAPL: path}, SimClock(0))
    with pytest.raises(FeedError, match="unparseable number"):
        await feed.fetch_bars(AAPL, BarInterval.MIN_1, *FOREVER)


async def test_wrong_column_count_is_rejected(tmp_path: Path) -> None:
    path = _zip(tmp_path, "AAPL.zip", ["2023-01-03 09:30:00,100,101,99,100.5\n"])
    feed = FirstRateFeed({AAPL: path}, SimClock(0))
    with pytest.raises(FeedError, match="expected 6 fields"):
        await feed.fetch_bars(AAPL, BarInterval.MIN_1, *FOREVER)


async def test_zip_with_more_than_one_member_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "AAPL.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("AAPL.csv", _HEADER)
        archive.writestr("extra.csv", _HEADER)
    feed = FirstRateFeed({AAPL: path}, SimClock(0))
    with pytest.raises(FeedError, match="expected exactly one file"):
        await feed.fetch_bars(AAPL, BarInterval.MIN_1, *FOREVER)


# ── ts_init ──────────────────────────────────────────────────


async def test_ts_init_comes_from_the_clock(tmp_path: Path) -> None:
    path = _zip(tmp_path, "AAPL.zip", ["2023-01-03 09:30:00,100,101,99,100.5,1000\n"])
    clock = SimClock(42)
    feed = FirstRateFeed({AAPL: path}, clock)
    bars = await feed.fetch_bars(AAPL, BarInterval.MIN_1, *FOREVER)
    assert bars[0].ts_init == 42


# ── coverage() ──────────────────────────────────────────────────


async def test_coverage_reports_the_session_dates_the_file_holds(tmp_path: Path) -> None:
    path = _zip(
        tmp_path,
        "AAPL.zip",
        [
            "2023-01-03 09:30:00,100,101,99,100.5,1000\n",
            "2023-01-05 15:59:00,100,101,99,100.5,1000\n",
        ],
    )
    feed = FirstRateFeed({AAPL: path}, SimClock(0))
    assert feed.coverage() == (date(2023, 1, 3), date(2023, 1, 5))


async def test_coverage_dates_a_bar_by_its_et_open_not_its_utc_close(tmp_path: Path) -> None:
    """The last extended-hours bar opens at 19:59 ET, which is 00:59 UTC the
    next day. Dating it by the UTC close would put the file's span a day long
    and make `seed ingest` crawl a session the file has no rows for."""
    path = _zip(tmp_path, "AAPL.zip", ["2023-01-03 19:59:00,100,101,99,100.5,1000\n"])
    feed = FirstRateFeed({AAPL: path}, SimClock(0))
    assert feed.coverage() == (date(2023, 1, 3), date(2023, 1, 3))


async def test_coverage_spans_every_configured_file(tmp_path: Path) -> None:
    aapl = _zip(tmp_path, "AAPL.zip", ["2023-01-03 09:30:00,100,101,99,100.5,1000\n"])
    msft = _zip(tmp_path, "MSFT.zip", ["2023-02-06 09:30:00,100,101,99,100.5,1000\n"])
    feed = FirstRateFeed({AAPL: aapl, Symbol("MSFT", Venue.NASDAQ): msft}, SimClock(0))
    assert feed.coverage() == (date(2023, 1, 3), date(2023, 2, 6))


async def test_coverage_is_none_when_no_file_holds_a_row(tmp_path: Path) -> None:
    path = _zip(tmp_path, "AAPL.zip", [])
    assert FirstRateFeed({AAPL: path}, SimClock(0)).coverage() is None


async def test_coverage_rejects_an_unparseable_file(tmp_path: Path) -> None:
    path = _zip(tmp_path, "AAPL.zip", ["not,a,bar\n"])
    with pytest.raises(FeedError):
        FirstRateFeed({AAPL: path}, SimClock(0)).coverage()
