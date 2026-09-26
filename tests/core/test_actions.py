"""Tests for corporate actions and price adjustment.

Heavier on rejection and on the point-in-time boundary than on happy paths. The
happy path here is arithmetic — a 4:1 split quarters the price — and it is not
where this goes wrong. What goes wrong is an action applied to the wrong side
of its effective date, a future action leaking into a past decision, or a
factor that is numerically right and unstorable.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from neurotrade.core.actions import (
    ONE,
    AdjustmentSeries,
    CorporateAction,
    PriceGap,
    adjust_bars,
    unexplained_gaps,
)
from neurotrade.core.events import Bar, BarInterval
from neurotrade.core.types import Price, Quantity, Symbol, Venue

AAPL = Symbol("AAPL", Venue.NASDAQ)
MSFT = Symbol("MSFT", Venue.NASDAQ)

DAY = 86_400_000_000_000
# 2023-06-01 00:00:00 UTC, so a bar's UTC date is easy to reason about in tests.
BASE_NS = 1_685_577_600_000_000_000


def bar(ts: int, price: str, *, volume: str = "1000", symbol: Symbol = AAPL) -> Bar:
    """A flat daily bar — every price equal — so adjustment is easy to read."""
    value = Price(price)
    return Bar(
        symbol=symbol,
        ts_event=ts,
        ts_init=ts,
        interval=BarInterval.DAY_1,
        open=value,
        high=value,
        low=value,
        close=value,
        volume=Quantity(volume),
    )


def split(day: date, ratio: str, *, symbol: Symbol = AAPL) -> CorporateAction:
    return CorporateAction(symbol=symbol, effective_date=day, split_ratio=Decimal(ratio))


def dividend(day: date, amount: str, *, symbol: Symbol = AAPL) -> CorporateAction:
    return CorporateAction(symbol=symbol, effective_date=day, dividend=Decimal(amount))


# ── CorporateAction ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("ratio", ["0", "-1", "-0.5"])
def test_non_positive_split_ratio_is_refused(ratio: str) -> None:
    """A zero or negative ratio makes every factor before it meaningless."""
    with pytest.raises(ValueError, match="must be positive"):
        CorporateAction(AAPL, date(2020, 8, 31), split_ratio=Decimal(ratio))


def test_negative_dividend_is_refused() -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        CorporateAction(AAPL, date(2020, 8, 31), dividend=Decimal("-0.5"))


def test_a_row_may_carry_both_a_split_and_a_dividend() -> None:
    """Feeds report them together when they fall on the same date."""
    action = CorporateAction(AAPL, date(2020, 8, 31), Decimal(4), Decimal("0.22"))
    assert (action.is_split, action.is_dividend) == (True, True)


def test_identity_ratio_is_not_a_split() -> None:
    assert not CorporateAction(AAPL, date(2020, 8, 31)).is_split


# ── AdjustmentSeries: the effective-date boundary ────────────────────────────


def test_action_applies_to_bars_before_its_effective_date() -> None:
    series = AdjustmentSeries(AAPL, [split(date(2020, 8, 31), "4")])
    assert series.price_factor(date(2020, 8, 28), as_of=date(2020, 9, 30)) == Decimal("0.25")


def test_action_does_not_apply_to_its_own_effective_date() -> None:
    """The bar on the effective date already trades on the new basis.

    Getting this inclusive would halve the split session itself, which shifts
    the whole series by one bar and corrupts any opening range built from it.
    """
    series = AdjustmentSeries(AAPL, [split(date(2020, 8, 31), "4")])
    assert series.price_factor(date(2020, 8, 31), as_of=date(2020, 9, 30)) == ONE


def test_action_does_not_apply_to_bars_after_it() -> None:
    series = AdjustmentSeries(AAPL, [split(date(2020, 8, 31), "4")])
    assert series.price_factor(date(2020, 9, 1), as_of=date(2020, 9, 30)) == ONE


# ── AdjustmentSeries: point-in-time ──────────────────────────────────────────


def test_an_action_after_as_of_is_invisible() -> None:
    """The core PIT guarantee: a 2021 split cannot touch a 2020 decision."""
    series = AdjustmentSeries(AAPL, [split(date(2021, 6, 1), "4")])
    assert series.price_factor(date(2020, 8, 28), as_of=date(2020, 12, 31)) == ONE
    assert series.price_factor(date(2020, 8, 28), as_of=date(2021, 12, 31)) == Decimal("0.25")


def test_as_of_before_the_observation_is_refused() -> None:
    """Asking for a basis earlier than the price itself is a caller bug."""
    series = AdjustmentSeries(AAPL, [])
    with pytest.raises(ValueError, match="is before observed"):
        series.price_factor(date(2020, 9, 1), as_of=date(2020, 8, 1))


def test_successive_splits_compound() -> None:
    series = AdjustmentSeries(AAPL, [split(date(2014, 6, 9), "7"), split(date(2020, 8, 31), "4")])
    assert series.price_factor(date(2014, 1, 2), as_of=date(2021, 1, 4)) == Decimal("0.03571429")


def test_a_reverse_split_raises_the_factor_above_one() -> None:
    """A 1-for-10 arrives as a ratio of 0.1 and scales old prices up by 10."""
    series = AdjustmentSeries(AAPL, [split(date(2023, 6, 1), "0.1")])
    assert series.price_factor(date(2023, 5, 31), as_of=date(2023, 6, 30)) == Decimal("10")


def test_actions_for_another_symbol_are_refused() -> None:
    """Applying one company's split to another's prices must be impossible."""
    with pytest.raises(ValueError, match=r"action for MSFT\.NASDAQ in series for AAPL\.NASDAQ"):
        AdjustmentSeries(AAPL, [split(date(2020, 8, 31), "4", symbol=MSFT)])


def test_unordered_actions_are_sorted_on_construction() -> None:
    series = AdjustmentSeries(AAPL, [split(date(2020, 8, 31), "4"), split(date(2014, 6, 9), "7")])
    assert [action.effective_date for action in series.actions] == [
        date(2014, 6, 9),
        date(2020, 8, 31),
    ]


# ── AdjustmentSeries: dividends and total return ─────────────────────────────


def test_price_factor_ignores_dividends() -> None:
    """A stop is hit because the price traded there; a dividend fills nothing."""
    series = AdjustmentSeries(
        AAPL,
        [dividend(date(2023, 6, 1), "1")],
        prior_close={date(2023, 6, 1): Price("100")},
    )
    assert series.price_factor(date(2023, 5, 31), as_of=date(2023, 6, 30)) == ONE


def test_total_return_factor_removes_the_ex_dividend_drop() -> None:
    series = AdjustmentSeries(
        AAPL,
        [dividend(date(2023, 6, 1), "1")],
        prior_close={date(2023, 6, 1): Price("100")},
    )
    assert series.total_return_factor(date(2023, 5, 31), as_of=date(2023, 6, 30)) == Decimal("0.99")


def test_a_dividend_without_a_prior_close_is_skipped_not_guessed() -> None:
    """A wrong denominator is a silent drift through every downstream return."""
    series = AdjustmentSeries(AAPL, [dividend(date(2023, 6, 1), "1")])
    assert series.total_return_factor(date(2023, 5, 31), as_of=date(2023, 6, 30)) == ONE


def test_a_dividend_at_or_above_the_prior_close_is_refused() -> None:
    """Not a large dividend — bad data. The factor would be zero or negative."""
    series = AdjustmentSeries(
        AAPL,
        [dividend(date(2023, 6, 1), "100")],
        prior_close={date(2023, 6, 1): Price("100")},
    )
    with pytest.raises(ValueError, match="is not below"):
        series.total_return_factor(date(2023, 5, 31), as_of=date(2023, 6, 30))


# ── adjust_bars ──────────────────────────────────────────────────────────────


def test_prices_fall_and_volume_rises_across_a_split() -> None:
    """Leaving volume alone would put a 4x step into every volume feature."""
    series = AdjustmentSeries(AAPL, [split(date(2023, 6, 2), "4")])
    (adjusted,) = adjust_bars([bar(BASE_NS, "400")], series, as_of=date(2023, 6, 30))
    assert adjusted.close == Price("100")
    assert adjusted.volume == Quantity("4000")


def test_a_bar_needing_no_adjustment_is_returned_unchanged() -> None:
    series = AdjustmentSeries(AAPL, [])
    original = bar(BASE_NS, "400")
    (adjusted,) = adjust_bars([original], series, as_of=date(2023, 6, 30))
    assert adjusted is original


def test_volume_follows_the_split_not_the_dividend() -> None:
    """A cash payment leaves the share count untouched."""
    series = AdjustmentSeries(
        AAPL,
        [dividend(date(2023, 6, 2), "1")],
        prior_close={date(2023, 6, 2): Price("100")},
    )
    (adjusted,) = adjust_bars(
        [bar(BASE_NS, "100")], series, as_of=date(2023, 6, 30), total_return=True
    )
    assert adjusted.close == Price("99")
    assert adjusted.volume == Quantity("1000")


def test_adjusting_with_another_symbols_series_is_refused() -> None:
    series = AdjustmentSeries(MSFT, [])
    with pytest.raises(ValueError, match=r"bar for AAPL\.NASDAQ adjusted with series for MSFT"):
        adjust_bars([bar(BASE_NS, "100")], series, as_of=date(2023, 6, 30))


def test_an_adjusted_price_fits_the_corpus_scale() -> None:
    """A 3:1 split makes a price that does not terminate.

    `decimal128(18, 8)` refuses a value with more places than it can hold, so
    an unquantized factor would make the bar unstorable rather than rounded.
    """
    series = AdjustmentSeries(AAPL, [split(date(2023, 6, 2), "3")])
    (adjusted,) = adjust_bars([bar(BASE_NS, "100")], series, as_of=date(2023, 6, 30))
    assert adjusted.close.value == adjusted.close.value.quantize(Decimal("0.00000001"))


def test_an_adjusted_bar_still_satisfies_its_own_invariants() -> None:
    """Scaling every price by one factor must preserve high >= low."""
    series = AdjustmentSeries(AAPL, [split(date(2023, 6, 2), "4")])
    original = Bar(
        symbol=AAPL,
        ts_event=BASE_NS,
        ts_init=BASE_NS,
        interval=BarInterval.DAY_1,
        open=Price("400"),
        high=Price("420"),
        low=Price("390"),
        close=Price("410"),
        volume=Quantity("1000"),
        vwap=Price("405"),
    )
    (adjusted,) = adjust_bars([original], series, as_of=date(2023, 6, 30))
    assert adjusted.high == Price("105")
    assert adjusted.vwap == Price("101.25")


# ── unexplained_gaps ─────────────────────────────────────────────────────────


def test_a_known_split_leaves_no_gap() -> None:
    """The whole point: a reported split explains its own price move."""
    bars = [bar(BASE_NS, "400"), bar(BASE_NS + DAY, "100")]
    series = AdjustmentSeries(AAPL, [split(date(2023, 6, 2), "4")])
    assert unexplained_gaps(bars, series, as_of=date(2023, 6, 30)) == ()


def test_an_unreported_split_is_caught() -> None:
    """The failure mode this exists for: the feed omitted the action."""
    bars = [bar(BASE_NS, "400"), bar(BASE_NS + DAY, "100")]
    (gap,) = unexplained_gaps(bars, AdjustmentSeries(AAPL, []), as_of=date(2023, 6, 30))
    assert gap.session_date == date(2023, 6, 2)
    assert round(gap.implied_split, 2) == Decimal("4.00")


def test_an_ordinary_move_is_not_a_gap() -> None:
    bars = [bar(BASE_NS, "100"), bar(BASE_NS + DAY, "108")]
    assert unexplained_gaps(bars, AdjustmentSeries(AAPL, []), as_of=date(2023, 6, 30)) == ()


def test_bars_are_sorted_before_comparison() -> None:
    """Out-of-order input must not invent a gap out of the ordering."""
    bars = [bar(BASE_NS + DAY, "108"), bar(BASE_NS, "100")]
    assert unexplained_gaps(bars, AdjustmentSeries(AAPL, []), as_of=date(2023, 6, 30)) == ()


@pytest.mark.parametrize("threshold", ["0", "-0.1"])
def test_a_non_positive_threshold_is_refused(threshold: str) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        unexplained_gaps(
            [bar(BASE_NS, "100")],
            AdjustmentSeries(AAPL, []),
            as_of=date(2023, 6, 30),
            threshold=Decimal(threshold),
        )


def test_a_tighter_threshold_catches_a_smaller_move() -> None:
    bars = [bar(BASE_NS, "100"), bar(BASE_NS + DAY, "80")]
    assert unexplained_gaps(bars, AdjustmentSeries(AAPL, []), as_of=date(2023, 6, 30)) == ()
    found = unexplained_gaps(
        bars, AdjustmentSeries(AAPL, []), as_of=date(2023, 6, 30), threshold=Decimal("0.1")
    )
    assert len(found) == 1


def test_a_reverse_split_gap_is_caught_upward() -> None:
    """A 1-for-10 shows a 10x jump, not a fall — `abs` is what catches it."""
    bars = [bar(BASE_NS, "1"), bar(BASE_NS + DAY, "10")]
    (gap,) = unexplained_gaps(bars, AdjustmentSeries(AAPL, []), as_of=date(2023, 6, 30))
    assert gap.ratio == Decimal("10")


# ── PriceGap ─────────────────────────────────────────────────────────────────


def test_implied_split_names_the_suspect_ratio() -> None:
    gap = PriceGap(AAPL, date(2020, 8, 31), Price("500"), Price("125"), Decimal("0.25"))
    assert gap.implied_split == Decimal("4")
