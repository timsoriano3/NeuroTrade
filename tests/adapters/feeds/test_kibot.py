"""Tests for `KibotFeed`, the file-backed feed over Kibot's `_unadjusted` file.

Rows are hand-written in Kibot's own headerless format; nothing here is a real
vendor download (`08-gotchas.doc.md`, `11-seed-data.plan.md` decision 4).
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from neurotrade.adapters.feeds.errors import FeedError
from neurotrade.adapters.feeds.kibot import KibotFeed
from neurotrade.core.clock import SimClock
from neurotrade.core.events import BarInterval
from neurotrade.core.ports import MarketDataPort
from neurotrade.core.types import Price, Symbol, Venue

IBM = Symbol("IBM", Venue.NYSE)
FOREVER = (0, 2_000_000_000_000_000_000)


def _write(tmp_path: Path, rows: list[str]) -> Path:
    path = tmp_path / "IBM_unadjusted.txt"
    path.write_text("".join(rows))
    return path


# ── Conformance ──────────────────────────────────────────────


def test_satisfies_the_market_data_port() -> None:
    assert isinstance(KibotFeed({}, SimClock(0)), MarketDataPort)


# ── Headerless parsing, and the open-to-close shift ───────────


async def test_a_bar_is_stamped_at_its_close_not_its_open(tmp_path: Path) -> None:
    """No header row — the first line is already data, unlike FRD's file."""
    path = _write(tmp_path, ["01/03/2023,09:30,100,101,99,100.5,1000\n"])
    feed = KibotFeed({IBM: path}, SimClock(0))
    bars = await feed.fetch_bars(IBM, BarInterval.MIN_1, *FOREVER)
    assert len(bars) == 1
    assert bars[0].ts_open == 1_672_756_200_000_000_000
    assert bars[0].ts_event == 1_672_756_260_000_000_000


async def test_dst_summer_bar_shifts_by_the_daylight_offset(tmp_path: Path) -> None:
    path = _write(tmp_path, ["07/03/2023,09:30,100,101,99,100.5,1000\n"])
    feed = KibotFeed({IBM: path}, SimClock(0))
    bars = await feed.fetch_bars(IBM, BarInterval.MIN_1, *FOREVER)
    assert bars[0].ts_open == 1_688_391_000_000_000_000
    assert bars[0].ts_event == 1_688_391_060_000_000_000


# ── Conversion and ordering ────────────────────────────────────


async def test_prices_are_exact_decimals(tmp_path: Path) -> None:
    path = _write(tmp_path, ["01/03/2023,09:30,100,101,99,100.5,1000\n"])
    feed = KibotFeed({IBM: path}, SimClock(0))
    bars = await feed.fetch_bars(IBM, BarInterval.MIN_1, *FOREVER)
    assert bars[0].close == Price("100.5")
    assert bars[0].volume.value == Decimal("1000")


async def test_rows_come_back_ascending_regardless_of_file_order(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        [
            "01/03/2023,09:31,101,102,100,101.5,1100\n",
            "01/03/2023,09:30,100,101,99,100.5,1000\n",
        ],
    )
    feed = KibotFeed({IBM: path}, SimClock(0))
    bars = await feed.fetch_bars(IBM, BarInterval.MIN_1, *FOREVER)
    assert [bar.ts_event for bar in bars] == sorted(bar.ts_event for bar in bars)


async def test_range_is_trimmed_to_start_inclusive_end_exclusive(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        [
            "01/03/2023,09:30,100,101,99,100.5,1000\n",
            "01/03/2023,09:31,101,102,100,101.5,1100\n",
            "01/03/2023,09:32,102,103,101,102.5,1200\n",
        ],
    )
    feed = KibotFeed({IBM: path}, SimClock(0))
    first_close = 1_672_756_260_000_000_000
    second_close = first_close + BarInterval.MIN_1.nanos
    bars = await feed.fetch_bars(IBM, BarInterval.MIN_1, first_close, second_close + 1)
    assert [bar.ts_event for bar in bars] == [first_close, second_close]


# ── Rejections ─────────────────────────────────────────────────


async def test_unconfigured_symbol_raises(tmp_path: Path) -> None:
    feed = KibotFeed({}, SimClock(0))
    with pytest.raises(FeedError, match="no kibot file configured"):
        await feed.fetch_bars(IBM, BarInterval.MIN_1, *FOREVER)


async def test_unsupported_interval_raises(tmp_path: Path) -> None:
    path = _write(tmp_path, ["01/03/2023,09:30,100,101,99,100.5,1000\n"])
    feed = KibotFeed({IBM: path}, SimClock(0))
    with pytest.raises(FeedError, match="only has 1m bars"):
        await feed.fetch_bars(IBM, BarInterval.MIN_5, *FOREVER)


async def test_bad_row_is_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, ["01/03/2023,09:30,not-a-number,101,99,100.5,1000\n"])
    feed = KibotFeed({IBM: path}, SimClock(0))
    with pytest.raises(FeedError, match="unparseable number"):
        await feed.fetch_bars(IBM, BarInterval.MIN_1, *FOREVER)


async def test_wrong_column_count_is_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, ["01/03/2023,09:30,100,101,99,100.5\n"])
    feed = KibotFeed({IBM: path}, SimClock(0))
    with pytest.raises(FeedError, match="expected 7 fields"):
        await feed.fetch_bars(IBM, BarInterval.MIN_1, *FOREVER)


async def test_bad_date_is_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, ["not-a-date,09:30,100,101,99,100.5,1000\n"])
    feed = KibotFeed({IBM: path}, SimClock(0))
    with pytest.raises(FeedError, match="unparseable date/time"):
        await feed.fetch_bars(IBM, BarInterval.MIN_1, *FOREVER)


# ── ts_init ──────────────────────────────────────────────────


async def test_ts_init_comes_from_the_clock(tmp_path: Path) -> None:
    path = _write(tmp_path, ["01/03/2023,09:30,100,101,99,100.5,1000\n"])
    clock = SimClock(42)
    feed = KibotFeed({IBM: path}, clock)
    bars = await feed.fetch_bars(IBM, BarInterval.MIN_1, *FOREVER)
    assert bars[0].ts_init == 42


# ── coverage() ──────────────────────────────────────────────────


async def test_coverage_reports_the_session_dates_the_file_holds(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        [
            "06/15/2026,09:30,100,101,99,100.5,1000\n",
            "06/17/2026,15:59,100,101,99,100.5,1000\n",
        ],
    )
    feed = KibotFeed({IBM: path}, SimClock(0))
    assert feed.coverage() == (date(2026, 6, 15), date(2026, 6, 17))


async def test_coverage_is_none_when_the_file_holds_no_row(tmp_path: Path) -> None:
    assert KibotFeed({IBM: _write(tmp_path, [])}, SimClock(0)).coverage() is None
