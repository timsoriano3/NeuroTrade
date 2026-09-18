"""Tests for the cost model.

The cases that matter are the ones where a cost could come out too low. A
backtest that understates costs does not fail — it produces a clean equity
curve and loses money live (§17), so every test here that pins a floor is
guarding against that direction specifically.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from neurotrade.core.costs import (
    CostModel,
    FeeSchedule,
    FlooredSpread,
    SpreadSource,
    TradeCost,
    tick_size,
)
from neurotrade.core.types import Currency, Money, Price, Quantity, Side

USD = Currency.USD


def model(**kwargs: object) -> CostModel:
    return CostModel(spreads=FlooredSpread(), fees=FeeSchedule(), **kwargs)  # type: ignore[arg-type]


# ── tick_size ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("price", "expected"),
    [("0.50", "0.0001"), ("0.9999", "0.0001"), ("1.00", "0.01"), ("400", "0.01")],
)
def test_tick_size_switches_at_one_dollar(price: str, expected: str) -> None:
    assert tick_size(Price(price)) == Decimal(expected)


# ── FlooredSpread ────────────────────────────────────────────────────────────


def test_a_proportional_estimate_is_used_when_it_clears_the_floors() -> None:
    assert FlooredSpread(fraction=Decimal("0.0005")).spread(Price("100")) == Decimal("0.05")


def test_the_minimum_floor_binds_on_a_cheap_name() -> None:
    """A thin name must never price as though it were SPY."""
    assert FlooredSpread(fraction=Decimal("0.0005")).spread(Price("2")) == Decimal("0.01")


def test_the_tick_floor_binds_below_the_minimum() -> None:
    thin = FlooredSpread(fraction=Decimal("0"), minimum=Decimal("0"))
    assert thin.spread(Price("0.50")) == Decimal("0.0001")


def test_volatility_widens_the_spread_when_configured() -> None:
    estimator = FlooredSpread(fraction=Decimal("0.0005"), volatility_multiplier=Decimal("0.01"))
    assert estimator.spread(Price("100"), volatility=0.5) > estimator.spread(Price("100"))


def test_volatility_is_ignored_when_the_multiplier_is_zero() -> None:
    estimator = FlooredSpread(fraction=Decimal("0.0005"))
    assert estimator.spread(Price("100"), volatility=0.5) == estimator.spread(Price("100"))


@pytest.mark.parametrize("field", ["fraction", "minimum", "volatility_multiplier"])
def test_a_negative_spread_term_is_refused(field: str) -> None:
    """A negative spread pays the strategy to trade — a manufactured edge."""
    with pytest.raises(ValueError, match="must not be negative"):
        FlooredSpread(**{field: Decimal("-1")})


def test_the_floored_estimator_satisfies_the_port() -> None:
    assert isinstance(FlooredSpread(), SpreadSource)


# ── FeeSchedule ──────────────────────────────────────────────────────────────


def test_commission_is_per_share_when_that_clears_the_minimum() -> None:
    assert FeeSchedule().commission(Quantity(1000), Price("50")).amount == Decimal("5.00000000")


def test_the_minimum_binds_on_a_small_order() -> None:
    assert FeeSchedule().commission(Quantity(10), Price("50")).amount == Decimal("1.00000000")


def test_the_cap_binds_before_the_minimum_on_a_cheap_small_order() -> None:
    """100 shares of a $0.50 name is $50 of value; a $1 minimum would be 2%."""
    assert FeeSchedule().commission(Quantity(100), Price("0.50")).amount == Decimal("0.50000000")


def test_a_zero_quantity_fill_costs_nothing() -> None:
    """The minimum applies to an order that traded, not to one that did not."""
    assert FeeSchedule().commission(Quantity(0), Price("50")).amount == 0


@pytest.mark.parametrize("field", ["per_share", "minimum", "maximum_fraction"])
def test_a_negative_fee_term_is_refused(field: str) -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        FeeSchedule(**{field: Decimal("-1")})  # type: ignore[arg-type]


# ── CostModel ────────────────────────────────────────────────────────────────


def test_a_taker_pays_half_the_spread() -> None:
    cost = model().cost(Side.BUY, Quantity(1000), Price("100"))
    # 5 bps of 100 is 0.05 full spread; half of that times 1000 shares.
    assert cost.spread.amount == Decimal("25.00000000")


def test_a_maker_pays_no_spread() -> None:
    """A resting order earns the half spread; modelled as zero, never negative.

    A negative spread cost would let a strategy's entire edge be the modelled
    rebate.
    """
    cost = model().cost(Side.BUY, Quantity(1000), Price("100"), is_maker=True)
    assert cost.spread.amount == 0


def test_slippage_is_zero_without_a_volume_reference() -> None:
    """Optimistic, and stated rather than hidden."""
    assert model().cost(Side.BUY, Quantity(100), Price("50"), volatility=0.3).slippage.amount == 0


def test_slippage_grows_with_participation() -> None:
    small = model().cost(
        Side.BUY, Quantity(100), Price("100"), volatility=0.3, average_volume=Quantity(1_000_000)
    )
    large = model().cost(
        Side.BUY, Quantity(50_000), Price("100"), volatility=0.3, average_volume=Quantity(1_000_000)
    )
    assert large.slippage.amount > small.slippage.amount


def test_slippage_grows_less_than_proportionally_with_size() -> None:
    """The square-root impact shape: 100x the size is not 100x the impact."""
    base = model().cost(
        Side.BUY, Quantity(100), Price("100"), volatility=0.3, average_volume=Quantity(1_000_000)
    )
    hundred = model().cost(
        Side.BUY, Quantity(10_000), Price("100"), volatility=0.3, average_volume=Quantity(1_000_000)
    )
    per_share_small = base.slippage.amount / 100
    per_share_large = hundred.slippage.amount / 10_000
    assert per_share_large > per_share_small
    assert per_share_large < per_share_small * 100


def test_zero_average_volume_is_refused() -> None:
    """Treating it as zero slippage would make illiquid names look cheapest."""
    with pytest.raises(ValueError, match="undefined against zero average volume"):
        model().cost(
            Side.BUY, Quantity(100), Price("50"), volatility=0.3, average_volume=Quantity(0)
        )


def test_a_zero_quantity_fill_has_no_cost_at_all() -> None:
    cost = model().cost(Side.BUY, Quantity(0), Price("50"))
    assert cost.total.amount == 0


def test_every_component_is_a_positive_cost() -> None:
    cost = model().cost(
        Side.SELL, Quantity(5_000), Price("100"), volatility=0.3, average_volume=Quantity(1_000_000)
    )
    assert cost.spread.amount > 0
    assert cost.commission.amount > 0
    assert cost.slippage.amount > 0


# ── TradeCost ────────────────────────────────────────────────────────────────


def test_total_sums_the_three_parts() -> None:
    cost = TradeCost(
        spread=Money(Decimal("2.50"), USD),
        commission=Money(Decimal("1.00"), USD),
        slippage=Money(Decimal("0.50"), USD),
    )
    assert cost.total.amount == Decimal("4.00")


def test_per_share_cost_of_a_zero_quantity_fill_is_refused() -> None:
    """Returning zero would read as "free"."""
    cost = TradeCost(
        spread=Money(Decimal("1"), USD),
        commission=Money(Decimal("1"), USD),
        slippage=Money(Decimal("0"), USD),
    )
    with pytest.raises(ValueError, match="undefined for a zero-quantity fill"):
        cost.per_share(Quantity(0))
