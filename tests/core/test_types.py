"""Tests for the core value objects.

Heavier on rejection cases than on happy paths: the point of these types is to
make a class of bug unrepresentable, so what matters is that the bad values are
actually refused.
"""

from __future__ import annotations

from decimal import Decimal
from typing import assert_type

import pytest

from neurotrade.core.types import (
    Currency,
    Money,
    Price,
    Quantity,
    Side,
    Symbol,
    Venue,
)

# ── Side ─────────────────────────────────────────────────────


def test_side_opposite_round_trips() -> None:
    assert Side.BUY.opposite is Side.SELL
    assert Side.SELL.opposite.opposite is Side.SELL


def test_side_sign() -> None:
    assert Side.BUY.sign == 1
    assert Side.SELL.sign == -1


# ── Venue / Symbol ───────────────────────────────────────────


@pytest.mark.parametrize(
    ("venue", "currency"),
    [
        (Venue.NASDAQ, Currency.USD),
        (Venue.NYSE, Currency.USD),
        (Venue.TSX, Currency.CAD),
        (Venue.TSXV, Currency.CAD),
    ],
)
def test_venue_currency(venue: Venue, currency: Currency) -> None:
    assert venue.currency is currency


def test_symbol_str_is_venue_qualified() -> None:
    assert str(Symbol("AAPL", Venue.NASDAQ)) == "AAPL.NASDAQ"


def test_same_ticker_on_two_venues_is_two_instruments() -> None:
    """TD is Toronto-Dominion on TSX and Tandem Diabetes on NASDAQ."""
    assert Symbol("TD", Venue.TSX) != Symbol("TD", Venue.NASDAQ)
    assert Symbol("TD", Venue.TSX).currency is Currency.CAD
    assert Symbol("TD", Venue.NASDAQ).currency is Currency.USD


@pytest.mark.parametrize("ticker", ["", " AAPL", "AAPL ", "aapl", "Aapl"])
def test_symbol_rejects_malformed_tickers(ticker: str) -> None:
    with pytest.raises(ValueError):
        Symbol(ticker, Venue.NASDAQ)


def test_symbol_rejects_smart_as_listing_venue() -> None:
    """SMART is IBKR's router, not a place a security is listed."""
    with pytest.raises(ValueError, match="order route"):
        Symbol("AAPL", Venue.SMART)


def test_symbol_is_hashable_and_usable_as_a_key() -> None:
    universe = {Symbol("AAPL", Venue.NASDAQ): 1, Symbol("SHOP", Venue.TSX): 2}
    assert universe[Symbol("AAPL", Venue.NASDAQ)] == 1


# ── Price ────────────────────────────────────────────────────


def test_price_from_str_is_exact() -> None:
    assert Price("0.1").value == Decimal("0.1")


def test_price_rejects_float_construction() -> None:
    """Decimal(0.1) is 0.1000000000000000055511151231257827."""
    with pytest.raises(TypeError, match="float"):
        Price(0.1)  # type: ignore[arg-type]


def test_price_from_float_avoids_binary_artefacts() -> None:
    assert Price.from_float(0.1).value == Decimal("0.1")


@pytest.mark.parametrize("bad", ["0", "-1", "-0.01"])
def test_price_rejects_non_positive(bad: str) -> None:
    with pytest.raises(ValueError, match="positive"):
        Price(bad)


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "-Infinity"])
def test_price_rejects_non_finite(bad: str) -> None:
    with pytest.raises(ValueError):
        Price(bad)


def test_price_difference_is_not_a_price() -> None:
    """A spread can be negative and can be zero; a price level cannot.

    `assert_type` is checked by mypy, not at runtime — a stronger guarantee
    than an isinstance check, which mypy correctly points out can never fail.
    """
    delta = Price("10.50") - Price("10.75")
    assert_type(delta, Decimal)
    assert delta == Decimal("-0.25")


def test_prices_order_and_compare_exactly() -> None:
    assert Price("10.10") < Price("10.20")
    assert Price("10.10") == Price("10.1000")


def test_decimal_arithmetic_has_no_float_drift() -> None:
    """The reason this module exists at all."""
    total = sum((Price(f"0.{n:02d}") - Price("0.01") for n in range(1, 11)), Decimal(0))
    assert total == Decimal("0.45")
    assert 0.1 + 0.2 != 0.3  # what we are avoiding


# ── Quantity ─────────────────────────────────────────────────


def test_quantity_allows_zero_and_fractional_shares() -> None:
    assert Quantity(0).is_zero
    assert Quantity("0.5").value == Decimal("0.5")


def test_quantity_rejects_negative() -> None:
    with pytest.raises(ValueError, match="negative"):
        Quantity("-1")


def test_quantity_addition_and_subtraction() -> None:
    assert (Quantity(3) + Quantity(4)).value == Decimal(7)
    assert (Quantity(7) - Quantity(4)).value == Decimal(3)


def test_quantity_subtraction_below_zero_is_rejected() -> None:
    """Closing more than is held is a bug, not a short."""
    with pytest.raises(ValueError, match="negative"):
        Quantity(3) - Quantity(4)


# ── Money ────────────────────────────────────────────────────


def test_money_may_be_negative() -> None:
    assert Money("-100", Currency.USD).amount == Decimal("-100")


def test_money_of_price_and_quantity() -> None:
    assert Money.of(Price("10.25"), Quantity(4), Currency.USD).amount == Decimal("41.00")


def test_money_arithmetic_within_a_currency() -> None:
    usd = Currency.USD
    assert (Money("10", usd) + Money("5", usd)).amount == Decimal(15)
    assert (Money("10", usd) - Money("15", usd)).amount == Decimal(-5)
    assert (-Money("10", usd)).amount == Decimal(-10)
    assert (Money("10", usd) * 3).amount == Decimal(30)


@pytest.mark.parametrize("op", ["add", "sub", "lt", "gt", "le", "ge"])
def test_money_refuses_to_mix_currencies(op: str) -> None:
    """Both currencies are live at once when trading US and TSX names."""
    usd, cad = Money("10", Currency.USD), Money("10", Currency.CAD)
    with pytest.raises(ValueError, match="convert explicitly"):
        getattr(usd, f"__{op}__")(cad)


def test_money_comparison_within_a_currency() -> None:
    usd = Currency.USD
    assert Money("10", usd) < Money("11", usd)
    assert Money("10", usd) >= Money("10", usd)


def test_money_zero() -> None:
    assert Money.zero(Currency.CAD).amount == Decimal(0)
    assert Money.zero(Currency.CAD).currency is Currency.CAD


# ── Immutability ─────────────────────────────────────────────


@pytest.mark.parametrize(
    ("obj", "field"),
    [
        (Symbol("AAPL", Venue.NASDAQ), "ticker"),
        (Price("1"), "value"),
        (Quantity(1), "value"),
        (Money("1", Currency.USD), "amount"),
    ],
)
def test_value_objects_are_frozen(obj: object, field: str) -> None:
    with pytest.raises(AttributeError):
        setattr(obj, field, "mutated")
