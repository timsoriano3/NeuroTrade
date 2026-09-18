"""Tests for triple-barrier labelling and uniqueness weighting.

This is where being wrong is silent. A label that is subtly optimistic does not
raise, does not fail a type check, and produces a backtest that looks better
than the strategy is — so the tests that matter most here are the ones pinning
the pessimistic choices: intrabar touches, stop-wins-ties, and costs on both
legs.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from neurotrade.core.costs import CostModel, FeeSchedule, FlooredSpread
from neurotrade.core.events import Bar, BarInterval
from neurotrade.core.types import Price, Quantity, Side, Symbol, Venue
from neurotrade.lab.labelling import (
    BarrierTouch,
    Label,
    average_uniqueness,
    concurrency,
    label_barriers,
    triple_barrier,
)

AAPL = Symbol("AAPL", Venue.NASDAQ)
SIZE = Quantity(1000)

# The cheapest model the cost layer permits. NOT free: `FlooredSpread` floors at
# one tick whatever the estimate says, because nothing trades tighter than a
# tick. That floor is deliberate, so a genuinely costless backtest is
# unrepresentable here — which is the point.
FREE = CostModel(
    spreads=FlooredSpread(fraction=Decimal("0"), minimum=Decimal("0")),
    fees=FeeSchedule(per_share=Decimal("0"), minimum=Decimal("0")),
)
REAL = CostModel(spreads=FlooredSpread(), fees=FeeSchedule())


def bar(index: int, high: str, low: str, close: str) -> Bar:
    return Bar(
        symbol=AAPL,
        ts_event=index,
        ts_init=index,
        interval=BarInterval.MIN_1,
        open=Price(close),
        high=Price(high),
        low=Price(low),
        close=Price(close),
        volume=Quantity(1000),
    )


def flat(index: int, price: str = "100") -> Bar:
    return bar(index, price, price, price)


def label(
    bars: list[Bar],
    *,
    side: Side = Side.BUY,
    costs: CostModel = FREE,
    max_bars: int = 5,
    target: str = "0.02",
    stop: str = "0.02",
) -> BarrierTouch | None:
    return triple_barrier(
        bars,
        0,
        side=side,
        profit_target=Decimal(target),
        stop_loss=Decimal(stop),
        max_bars=max_bars,
        costs=costs,
        quantity=SIZE,
    )


# ── Which barrier, and when ──────────────────────────────────────────────────


def test_the_profit_barrier_is_hit_when_the_high_reaches_it() -> None:
    touch = label([flat(0), bar(1, "103", "100", "103")])
    assert touch is not None
    assert touch.label is Label.PROFIT


def test_the_stop_is_hit_intrabar_not_at_the_close() -> None:
    """A stop is hit when the low trades through it.

    Testing closes understates how often stops are hit, always flatteringly:
    this bar closes back at 100 but traded to 97 on the way.
    """
    touch = label([flat(0), bar(1, "100", "97", "100")])
    assert touch is not None
    assert touch.label is Label.STOP


def test_the_earliest_barrier_wins_across_bars() -> None:
    bars = [flat(0), bar(1, "103", "100", "103"), bar(2, "100", "90", "90")]
    touch = label(bars)
    assert touch is not None
    assert (touch.label, touch.bars_held) == (Label.PROFIT, 1)


def test_the_stop_wins_when_both_fall_inside_one_bar() -> None:
    """Unknowable from OHLC; the conservative reading is the only honest one."""
    touch = label([flat(0), bar(1, "105", "95", "100")])
    assert touch is not None
    assert touch.label is Label.STOP
    assert touch.ambiguous


def test_an_unambiguous_touch_is_not_flagged() -> None:
    touch = label([flat(0), bar(1, "103", "100", "103")])
    assert touch is not None
    assert not touch.ambiguous


def test_neither_barrier_touched_times_out_at_the_close() -> None:
    touch = label([flat(0), flat(1, "100.5"), flat(2, "100.5")], max_bars=2)
    assert touch is not None
    assert (touch.label, touch.bars_held) == (Label.TIMEOUT, 2)
    assert touch.exit == Price("100.5")


def test_the_vertical_barrier_bounds_the_walk() -> None:
    """A touch after the time limit must not be seen."""
    bars = [flat(0), flat(1), flat(2), bar(3, "200", "100", "200")]
    touch = label(bars, max_bars=2)
    assert touch is not None
    assert touch.label is Label.TIMEOUT


# ── Lookahead and the entry bar ──────────────────────────────────────────────


def test_the_entry_bar_itself_cannot_trigger_a_barrier() -> None:
    """A decision on a bar's close cannot be filled inside that same bar."""
    bars = [bar(0, "200", "50", "100"), flat(1)]
    touch = label(bars)
    assert touch is not None
    assert touch.label is Label.TIMEOUT


def test_no_bars_after_the_entry_yields_no_label() -> None:
    """None, not a timeout — an unlabelled decision is not an outcome.

    Conflating them puts fabricated outcomes at the most recent dates, which is
    exactly where a model is most likely to be trusted.
    """
    assert label([flat(0)]) is None


# ── Short positions ──────────────────────────────────────────────────────────


def test_a_short_profits_when_the_price_falls() -> None:
    touch = label([flat(0), bar(1, "100", "97", "97")], side=Side.SELL)
    assert touch is not None
    assert touch.label is Label.PROFIT


def test_a_short_stops_out_when_the_price_rises() -> None:
    touch = label([flat(0), bar(1, "103", "100", "103")], side=Side.SELL)
    assert touch is not None
    assert touch.label is Label.STOP


# ── Costs inside the label ───────────────────────────────────────────────────


def test_costs_are_charged_on_both_legs() -> None:
    """A round trip crosses the spread twice and pays commission twice."""
    free = label([flat(0), bar(1, "103", "100", "103")], costs=FREE)
    priced = label([flat(0), bar(1, "103", "100", "103")], costs=REAL)
    assert free is not None and priced is not None
    assert priced.realised_return < free.realised_return


def test_a_profitable_label_can_still_lose_money_after_costs() -> None:
    """The case a cost-blind label hides, and why `is_win` is not `label == PROFIT`."""
    # A 5bp target on a $100 name: 5c gross per share, against a 5c spread
    # crossed twice (2.5c a side) plus half a cent a share of commission each
    # way — 6bp of cost against 5bp of edge.
    touch = label(
        [flat(0), bar(1, "100.05", "100", "100.05")], costs=REAL, target="0.0005", stop="0.02"
    )
    assert touch is not None
    assert touch.label is Label.PROFIT
    assert not touch.is_win
    assert touch.realised_return < 0


def test_the_cheapest_possible_model_still_charges_a_tick() -> None:
    """A 2% gross win comes back as 1.99% because a tick is still crossed twice.

    `FlooredSpread` floors at one tick regardless of its estimate, so no
    configuration produces a costless backtest.
    """
    touch = label([flat(0), bar(1, "103", "100", "103")], costs=FREE)
    assert touch is not None
    assert touch.is_win
    assert touch.realised_return == Decimal("0.01990000")


# ── Rejection ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"target": "0"}, "profit_target 0 must be positive"),
        ({"target": "-0.01"}, "must be positive"),
        ({"stop": "0"}, "stop_loss 0 must be positive"),
        ({"max_bars": 0}, "max_bars 0 must be at least 1"),
    ],
)
def test_invalid_barriers_are_refused(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        label([flat(0), flat(1)], **kwargs)  # type: ignore[arg-type]


def test_an_entry_index_outside_the_series_is_refused() -> None:
    with pytest.raises(ValueError, match="is outside the 2 bars given"):
        triple_barrier(
            [flat(0), flat(1)],
            5,
            side=Side.BUY,
            profit_target=Decimal("0.02"),
            stop_loss=Decimal("0.02"),
            max_bars=5,
            costs=FREE,
            quantity=SIZE,
        )


# ── label_barriers ───────────────────────────────────────────────────────────


def test_many_entries_are_labelled_in_order() -> None:
    bars = [flat(i) for i in range(10)]
    labelled = label_barriers(
        bars,
        [0, 2, 4],
        side=Side.BUY,
        profit_target=Decimal("0.02"),
        stop_loss=Decimal("0.02"),
        max_bars=3,
        costs=FREE,
        quantity=SIZE,
    )
    assert [index for index, _ in labelled] == [0, 2, 4]


def test_entries_too_close_to_the_end_are_skipped_not_faked() -> None:
    bars = [flat(i) for i in range(3)]
    labelled = label_barriers(
        bars,
        [0, 2],
        side=Side.BUY,
        profit_target=Decimal("0.02"),
        stop_loss=Decimal("0.02"),
        max_bars=3,
        costs=FREE,
        quantity=SIZE,
    )
    assert [index for index, _ in labelled] == [0]


# ── Concurrency and uniqueness ───────────────────────────────────────────────


def test_concurrency_counts_live_labels_per_bar() -> None:
    assert concurrency([(0, 2), (1, 3)], length=5) == (1, 2, 2, 1, 0)


def test_fully_overlapping_labels_each_score_one_half() -> None:
    """Two labels sharing every bar are nearly one observation."""
    assert average_uniqueness([(0, 1), (0, 1)], length=2) == (0.5, 0.5)


def test_disjoint_labels_are_fully_unique() -> None:
    assert average_uniqueness([(0, 0), (1, 1)], length=2) == (1.0, 1.0)


def test_partial_overlap_scores_between_the_two() -> None:
    weights = average_uniqueness([(0, 2), (2, 4)], length=5)
    assert all(0.5 < weight < 1.0 for weight in weights)


def test_uniqueness_falls_as_more_labels_pile_onto_the_same_bars() -> None:
    """The inflation §8's weighting exists to remove."""
    two = average_uniqueness([(0, 1)] * 2, length=2)[0]
    ten = average_uniqueness([(0, 1)] * 10, length=2)[0]
    assert ten < two


@pytest.mark.parametrize("span", [(2, 1), (-1, 1), (0, 9)])
def test_an_impossible_span_is_refused(span: tuple[int, int]) -> None:
    with pytest.raises(ValueError):
        concurrency([span], length=5)
