"""Tests for the Yahoo corporate-actions adapter.

The network is never reached: `YFinanceActions` takes its downloader by
injection, so every test here drives the conversion from Yahoo's shape to the
domain's. That conversion is where the adapter can be wrong — a `0.0` ratio
meaning "no split", a float that is nearly but not quite the dividend, a NaN
that would poison every factor derived from it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import date
from decimal import Decimal

import pytest

from neurotrade.adapters.feeds.errors import FeedError
from neurotrade.adapters.feeds.yfinance_actions import ActionRow, YFinanceActions
from neurotrade.core.actions import ONE, CorporateAction
from neurotrade.core.types import Symbol, Venue

AAPL = Symbol("AAPL", Venue.NASDAQ)
SHOP = Symbol("SHOP", Venue.TSX)

WIDE = (date(2000, 1, 1), date(2030, 1, 1))


def fetch(
    rows: Sequence[ActionRow],
    symbol: Symbol = AAPL,
    span: tuple[date, date] = WIDE,
) -> Sequence[CorporateAction]:
    """Run one fetch against a fixed downloader."""
    feed = YFinanceActions(lambda ticker: rows)
    return asyncio.run(feed.fetch_actions(symbol, *span))


# ── Yahoo's conventions ──────────────────────────────────────────────────────


def test_a_zero_ratio_means_no_split_not_an_invalid_one() -> None:
    """Yahoo writes 0.0 on a dividend-only row; `CorporateAction` refuses 0."""
    (action,) = fetch([ActionRow(date(2023, 6, 1), split_ratio=0.0, dividend=0.27)])
    assert action.split_ratio == ONE
    assert action.dividend == Decimal("0.27")


def test_a_float_dividend_is_rounded_to_a_comparable_decimal() -> None:
    """0.27 arrives as 0.27000000000000002 and must not stay that way."""
    (action,) = fetch([ActionRow(date(2023, 6, 1), 0.0, 0.27000000000000002)])
    assert action.dividend == Decimal("0.27")
    assert str(action.dividend) == "0.27"


def test_a_whole_split_ratio_keeps_plain_notation() -> None:
    (action,) = fetch([ActionRow(date(2020, 8, 31), 4.0, 0.0)])
    assert str(action.split_ratio) == "4"


def test_a_fractional_reverse_split_survives() -> None:
    """A 1-for-10 arrives as 0.1 and is a legitimate ratio."""
    (action,) = fetch([ActionRow(date(2023, 6, 1), 0.1, 0.0)])
    assert action.split_ratio == Decimal("0.1")


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_number_is_refused(bad: float) -> None:
    """NaN reaches Decimal intact, then makes every comparison silently false."""
    with pytest.raises(FeedError, match="non-finite"):
        fetch([ActionRow(date(2023, 6, 1), bad, 0.0)])


def test_a_non_finite_dividend_is_refused() -> None:
    with pytest.raises(FeedError, match="non-finite dividend"):
        fetch([ActionRow(date(2023, 6, 1), 0.0, float("nan"))])


# ── Range and identity ───────────────────────────────────────────────────────


def test_actions_outside_the_range_are_dropped() -> None:
    rows = [ActionRow(date(2014, 6, 9), 7.0, 0.0), ActionRow(date(2020, 8, 31), 4.0, 0.0)]
    actions = fetch(rows, span=(date(2020, 1, 1), date(2021, 1, 1)))
    assert [action.effective_date for action in actions] == [date(2020, 8, 31)]


def test_the_range_is_inclusive_at_both_ends() -> None:
    rows = [ActionRow(date(2020, 8, 31), 4.0, 0.0)]
    assert len(fetch(rows, span=(date(2020, 8, 31), date(2020, 8, 31)))) == 1


def test_an_inverted_range_is_refused() -> None:
    feed = YFinanceActions(lambda ticker: ())
    with pytest.raises(ValueError, match="is before start"):
        asyncio.run(feed.fetch_actions(AAPL, date(2021, 1, 1), date(2020, 1, 1)))


def test_actions_carry_the_symbol_they_were_asked_for() -> None:
    (action,) = fetch([ActionRow(date(2020, 8, 31), 4.0, 0.0)], symbol=SHOP)
    assert action.symbol == SHOP


def test_no_actions_is_an_empty_result_not_an_error() -> None:
    """Most names go years without one; silence is the common case."""
    assert fetch([]) == ()


# ── Caching ──────────────────────────────────────────────────────────────────


def test_the_downloader_is_called_once_per_symbol() -> None:
    """The whole history arrives in one call, so a second range is free."""
    calls: list[str] = []

    def download(ticker: str) -> Sequence[ActionRow]:
        calls.append(ticker)
        return (ActionRow(date(2020, 8, 31), 4.0, 0.0),)

    feed = YFinanceActions(download)

    async def twice() -> None:
        await feed.fetch_actions(AAPL, date(2019, 1, 1), date(2021, 1, 1))
        await feed.fetch_actions(AAPL, date(2020, 1, 1), date(2020, 12, 31))

    asyncio.run(twice())
    assert calls == ["AAPL"]


def test_the_canadian_suffix_reaches_the_downloader() -> None:
    """Yahoo cannot tell SHOP.TSX from SHOP.NYSE without it."""
    calls: list[str] = []

    def download(ticker: str) -> Sequence[ActionRow]:
        calls.append(ticker)
        return ()

    asyncio.run(YFinanceActions(download).fetch_actions(SHOP, *WIDE))
    assert calls == ["SHOP.TO"]
