"""Tests for Position.

Position is a fold, so the tests are mostly about sequences of fills. The
reversal case and the gross/net fee separation are the two that carry real risk
of being quietly wrong.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from neurotrade.core.ids import FillId, IntentId, OrderId
from neurotrade.core.orders import Fill
from neurotrade.core.position import Position
from neurotrade.core.types import Currency, Money, Price, Quantity, Side, Symbol, Venue

AAPL = Symbol("AAPL", Venue.NASDAQ)
SHOP = Symbol("SHOP", Venue.TSX)
NOW = 1_773_495_000_000_000_000

_INTENT = IntentId.derive(
    strategy="orb", strategy_version="1.0.0", symbol=AAPL, ts_event=NOW, seq=0
)
_ORDER = OrderId.derive(intent_id=_INTENT, ts_event=NOW)
_counter = iter(range(1_000_000))


def fill(
    side: Side,
    price: str,
    quantity: int | str,
    *,
    symbol: Symbol = AAPL,
    commission: str = "0",
) -> Fill:
    exec_id = f"e.{next(_counter)}"
    currency = symbol.currency
    return Fill(
        id=FillId.derive(order_id=_ORDER, broker_exec_id=exec_id),
        order_id=_ORDER,
        symbol=symbol,
        ts_event=NOW,
        ts_init=NOW,
        side=side,
        price=Price(price),
        quantity=Quantity(quantity),
        commission=Money(commission, currency),
        broker_exec_id=exec_id,
    )


def usd(amount: str) -> Money:
    return Money(amount, Currency.USD)


# ── Flat ─────────────────────────────────────────────────────


def test_a_flat_position_has_no_side_and_no_average() -> None:
    flat = Position.flat(AAPL)
    assert flat.is_flat
    assert flat.side is None
    assert flat.average_price is None
    assert flat.signed_quantity == 0
    assert flat.realised_pnl == usd("0")


def test_flat_position_has_no_unrealised_pnl() -> None:
    assert Position.flat(AAPL).unrealised_pnl(Price("999")) == usd("0")


def test_side_and_quantity_must_agree() -> None:
    with pytest.raises(ValueError, match="side must be None exactly when flat"):
        Position(
            symbol=AAPL,
            side=Side.BUY,
            quantity=Quantity(0),
            realised_pnl=usd("0"),
            fees=usd("0"),
        )


# ── Opening and adding ───────────────────────────────────────


def test_first_fill_opens_the_position() -> None:
    position = Position.flat(AAPL).apply(fill(Side.BUY, "100.00", 100))
    assert position.is_long
    assert position.quantity == Quantity(100)
    assert position.average_price == Price("100.00")
    assert position.signed_quantity == Decimal(100)


def test_a_short_has_negative_signed_quantity() -> None:
    position = Position.flat(AAPL).apply(fill(Side.SELL, "100.00", 100))
    assert position.is_short
    assert position.signed_quantity == Decimal(-100)


def test_adding_weights_the_average_price() -> None:
    """100 @ 100 then 100 @ 102 → average 101."""
    position = (
        Position.flat(AAPL)
        .apply(fill(Side.BUY, "100.00", 100))
        .apply(fill(Side.BUY, "102.00", 100))
    )
    assert position.quantity == Quantity(200)
    assert position.average_price == Price("101.00")


def test_average_price_is_size_weighted_not_a_plain_mean() -> None:
    """300 @ 100 then 100 @ 108 → 102, not 104."""
    position = (
        Position.flat(AAPL)
        .apply(fill(Side.BUY, "100.00", 300))
        .apply(fill(Side.BUY, "108.00", 100))
    )
    assert position.average_price == Price("102.00")


def test_adding_realises_nothing() -> None:
    position = (
        Position.flat(AAPL)
        .apply(fill(Side.BUY, "100.00", 100))
        .apply(fill(Side.BUY, "102.00", 100))
    )
    assert position.realised_pnl == usd("0")


# ── Closing ──────────────────────────────────────────────────


def test_closing_a_winning_long_realises_the_gain() -> None:
    position = (
        Position.flat(AAPL)
        .apply(fill(Side.BUY, "100.00", 100))
        .apply(fill(Side.SELL, "102.00", 100))
    )
    assert position.is_flat
    assert position.realised_pnl == usd("200.00")


def test_closing_a_losing_long_realises_the_loss() -> None:
    position = (
        Position.flat(AAPL)
        .apply(fill(Side.BUY, "100.00", 100))
        .apply(fill(Side.SELL, "99.00", 100))
    )
    assert position.realised_pnl == usd("-100.00")


def test_closing_a_winning_short_realises_the_gain() -> None:
    """Shorts profit when price falls — the sign must flip."""
    position = (
        Position.flat(AAPL)
        .apply(fill(Side.SELL, "100.00", 100))
        .apply(fill(Side.BUY, "98.00", 100))
    )
    assert position.is_flat
    assert position.realised_pnl == usd("200.00")


def test_closing_a_losing_short_realises_the_loss() -> None:
    position = (
        Position.flat(AAPL)
        .apply(fill(Side.SELL, "100.00", 100))
        .apply(fill(Side.BUY, "101.00", 100))
    )
    assert position.realised_pnl == usd("-100.00")


def test_partial_close_leaves_the_average_price_alone() -> None:
    """Selling half does not change what the other half cost."""
    position = (
        Position.flat(AAPL)
        .apply(fill(Side.BUY, "100.00", 100))
        .apply(fill(Side.SELL, "104.00", 40))
    )
    assert position.quantity == Quantity(60)
    assert position.average_price == Price("100.00")
    assert position.realised_pnl == usd("160.00")


def test_scaling_out_in_stages() -> None:
    position = Position.flat(AAPL).apply(fill(Side.BUY, "100.00", 300))
    for price, size in (("101.00", 100), ("102.00", 100), ("103.00", 100)):
        position = position.apply(fill(Side.SELL, price, size))
    assert position.is_flat
    assert position.realised_pnl == usd("600.00")  # 100 + 200 + 300


# ── Reversal ─────────────────────────────────────────────────


def test_reversing_through_zero_realises_then_reopens() -> None:
    """Sell 150 against a 100 long: close 100, open 50 short at the fill price."""
    position = (
        Position.flat(AAPL)
        .apply(fill(Side.BUY, "100.00", 100))
        .apply(fill(Side.SELL, "102.00", 150))
    )
    assert position.is_short
    assert position.quantity == Quantity(50)
    assert position.average_price == Price("102.00")
    assert position.realised_pnl == usd("200.00")


def test_reversal_does_not_net_into_one_averaged_position() -> None:
    """The bug this case exists to prevent.

    Netting a flip would leave a 50 short carrying the long's cost basis, so the
    realised gain on the closed leg would silently never be booked.
    """
    position = (
        Position.flat(AAPL)
        .apply(fill(Side.BUY, "100.00", 100))
        .apply(fill(Side.SELL, "102.00", 150))
    )
    assert position.realised_pnl != usd("0")
    assert position.average_price == Price("102.00")


def test_reversal_from_short_to_long() -> None:
    position = (
        Position.flat(AAPL)
        .apply(fill(Side.SELL, "100.00", 100))
        .apply(fill(Side.BUY, "98.00", 250))
    )
    assert position.is_long
    assert position.quantity == Quantity(150)
    assert position.average_price == Price("98.00")
    assert position.realised_pnl == usd("200.00")


# ── Fees kept separate from P&L ──────────────────────────────


def test_fees_accumulate_across_every_fill() -> None:
    position = (
        Position.flat(AAPL)
        .apply(fill(Side.BUY, "100.00", 100, commission="1.00"))
        .apply(fill(Side.SELL, "102.00", 100, commission="1.25"))
    )
    assert position.fees == usd("2.25")


def test_realised_pnl_is_gross_and_net_is_derived() -> None:
    """Netting fees as you go hides whether the strategy or the execution failed."""
    position = (
        Position.flat(AAPL)
        .apply(fill(Side.BUY, "100.00", 100, commission="1.00"))
        .apply(fill(Side.SELL, "102.00", 100, commission="1.00"))
    )
    assert position.realised_pnl == usd("200.00")  # gross
    assert position.fees == usd("2.00")
    assert position.net_realised_pnl == usd("198.00")


def test_fees_can_turn_a_gross_win_into_a_net_loss() -> None:
    """§3.3 — a signal must beat its own cost."""
    position = (
        Position.flat(AAPL)
        .apply(fill(Side.BUY, "100.00", 100, commission="6.00"))
        .apply(fill(Side.SELL, "100.05", 100, commission="6.00"))
    )
    assert position.realised_pnl == usd("5.00")
    assert position.net_realised_pnl == usd("-7.00")


# ── Valuation ────────────────────────────────────────────────


def test_unrealised_pnl_for_a_long() -> None:
    position = Position.flat(AAPL).apply(fill(Side.BUY, "100.00", 100))
    assert position.unrealised_pnl(Price("103.00")) == usd("300.00")
    assert position.unrealised_pnl(Price("98.00")) == usd("-200.00")


def test_unrealised_pnl_for_a_short() -> None:
    position = Position.flat(AAPL).apply(fill(Side.SELL, "100.00", 100))
    assert position.unrealised_pnl(Price("98.00")) == usd("200.00")
    assert position.unrealised_pnl(Price("103.00")) == usd("-300.00")


def test_market_value_is_signed() -> None:
    long = Position.flat(AAPL).apply(fill(Side.BUY, "100.00", 100))
    short = Position.flat(AAPL).apply(fill(Side.SELL, "100.00", 100))
    assert long.market_value(Price("101.00")) == usd("10100.00")
    assert short.market_value(Price("101.00")) == usd("-10100.00")


def test_total_pnl_combines_realised_unrealised_and_fees() -> None:
    position = (
        Position.flat(AAPL)
        .apply(fill(Side.BUY, "100.00", 200, commission="2.00"))
        .apply(fill(Side.SELL, "102.00", 100, commission="1.00"))
    )
    assert position.realised_pnl == usd("200.00")
    assert position.fees == usd("3.00")
    assert position.unrealised_pnl(Price("103.00")) == usd("300.00")
    assert position.total_pnl(Price("103.00")) == usd("497.00")


# ── R-multiples ──────────────────────────────────────────────


def test_realised_r_converts_pnl_into_risk_units() -> None:
    """Risked 1.00/share on 100 shares = 100 risk; +200 net is +2R."""
    position = (
        Position.flat(AAPL)
        .apply(fill(Side.BUY, "100.00", 100))
        .apply(fill(Side.SELL, "102.00", 100))
    )
    assert position.realised_r(Decimal("1.00"), Quantity(100)) == Decimal(2)


def test_realised_r_is_net_of_fees() -> None:
    position = (
        Position.flat(AAPL)
        .apply(fill(Side.BUY, "100.00", 100, commission="5.00"))
        .apply(fill(Side.SELL, "102.00", 100, commission="5.00"))
    )
    assert position.realised_r(Decimal("1.00"), Quantity(100)) == Decimal("1.9")


def test_a_full_stop_out_is_minus_one_r() -> None:
    position = (
        Position.flat(AAPL)
        .apply(fill(Side.BUY, "100.00", 100))
        .apply(fill(Side.SELL, "99.00", 100))
    )
    assert position.realised_r(Decimal("1.00"), Quantity(100)) == Decimal(-1)


def test_r_requires_positive_risk() -> None:
    with pytest.raises(ValueError, match="risk must be positive"):
        Position.flat(AAPL).realised_r(Decimal(0), Quantity(100))


# ── Guards ───────────────────────────────────────────────────


def test_a_fill_for_another_symbol_is_rejected() -> None:
    with pytest.raises(ValueError, match="fill is for"):
        Position.flat(AAPL).apply(fill(Side.BUY, "100", 100, symbol=SHOP))


def test_positions_are_immutable_and_apply_returns_a_new_one() -> None:
    flat = Position.flat(AAPL)
    opened = flat.apply(fill(Side.BUY, "100.00", 100))
    assert flat.is_flat  # unchanged
    assert opened.is_long
    with pytest.raises(AttributeError):
        opened.quantity = Quantity(1)  # type: ignore[misc]


def test_canadian_listings_book_in_cad() -> None:
    position = Position.flat(SHOP).apply(
        fill(Side.BUY, "142.60", 100, symbol=SHOP, commission="1.00")
    )
    assert position.fees.currency is Currency.CAD
    assert position.unrealised_pnl(Price("143.00")).currency is Currency.CAD


def test_fill_count_tracks_the_fold_length() -> None:
    position = (
        Position.flat(AAPL)
        .apply(fill(Side.BUY, "100.00", 50))
        .apply(fill(Side.BUY, "101.00", 50))
        .apply(fill(Side.SELL, "102.00", 100))
    )
    assert position.fill_count == 3
