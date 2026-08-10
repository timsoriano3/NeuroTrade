"""Tests for orders and fills.

The slippage sign convention is the part worth reading closely: it must report a
cost as positive for both directions, or the nightly cost-model recalibration
fits against a number that cancels itself out.
"""

from __future__ import annotations

from dataclasses import fields
from decimal import Decimal

import pytest

from neurotrade.core.events import Event
from neurotrade.core.ids import FillId, IntentId, OrderId
from neurotrade.core.orders import Fill, LiquidityFlag, Order, OrderType, TimeInForce
from neurotrade.core.types import Currency, Money, Price, Quantity, Side, Symbol, Venue

AAPL = Symbol("AAPL", Venue.NASDAQ)
SHOP = Symbol("SHOP", Venue.TSX)
NOW = 1_773_495_000_000_000_000

INTENT = IntentId.derive(strategy="orb", strategy_version="1.0.0", symbol=AAPL, ts_event=NOW, seq=0)
ORDER_ID = OrderId.derive(intent_id=INTENT, ts_event=NOW)


def make_order(**overrides: object) -> Order:
    defaults: dict[str, object] = {
        "id": ORDER_ID,
        "intent_id": INTENT,
        "symbol": AAPL,
        "ts_event": NOW,
        "ts_init": NOW,
        "side": Side.BUY,
        "quantity": Quantity(100),
        "order_type": OrderType.LIMIT,
        "limit_price": Price("100.00"),
        "config_hash": "abc123",
    }
    return Order(**{**defaults, **overrides})  # type: ignore[arg-type]


def make_fill(**overrides: object) -> Fill:
    defaults: dict[str, object] = {
        "id": FillId.derive(order_id=ORDER_ID, broker_exec_id="e.1"),
        "order_id": ORDER_ID,
        "symbol": AAPL,
        "ts_event": NOW,
        "ts_init": NOW,
        "side": Side.BUY,
        "price": Price("100.00"),
        "quantity": Quantity(100),
        "commission": Money("1.00", Currency.USD),
        "broker_exec_id": "e.1",
    }
    return Fill(**{**defaults, **overrides})  # type: ignore[arg-type]


# ── Immutability and the append-only property ────────────────


def test_orders_carry_no_mutable_status() -> None:
    """State is a projection folded from events, not a field that changes (§4.1)."""
    assert "status" not in {f.name for f in fields(Order)}


def test_orders_and_fills_are_immutable() -> None:
    with pytest.raises(AttributeError):
        make_order().quantity = Quantity(1)  # type: ignore[misc]
    with pytest.raises(AttributeError):
        make_fill().price = Price("1")  # type: ignore[misc]


def test_both_are_events_and_order_deterministically() -> None:
    order, fill = make_order(seq=0), make_fill(seq=1)
    assert isinstance(order, Event)
    assert isinstance(fill, Event)
    assert sorted([fill, order], key=lambda e: e.sort_key) == [order, fill]


# ── Audit trail ──────────────────────────────────────────────


def test_order_traces_back_to_its_intent() -> None:
    assert make_order().intent_id == INTENT


def test_order_carries_the_config_hash() -> None:
    """§6.3 — any trade must be reconstructable months later."""
    assert make_order(config_hash="deadbeef").config_hash == "deadbeef"


def test_fill_traces_back_to_its_order() -> None:
    assert make_fill().order_id == ORDER_ID


# ── Order type / price consistency ───────────────────────────


def test_market_order_needs_no_prices() -> None:
    order = make_order(order_type=OrderType.MARKET, limit_price=None)
    assert order.limit_price is None
    assert order.stop_price is None


def test_market_order_rejects_a_limit_price() -> None:
    with pytest.raises(ValueError, match="must not carry a limit_price"):
        make_order(order_type=OrderType.MARKET, limit_price=Price("100"))


def test_limit_order_requires_a_limit_price() -> None:
    with pytest.raises(ValueError, match="requires a limit_price"):
        make_order(order_type=OrderType.LIMIT, limit_price=None)


def test_limit_order_rejects_a_stop_price() -> None:
    with pytest.raises(ValueError, match="must not carry a stop_price"):
        make_order(order_type=OrderType.LIMIT, stop_price=Price("99"))


def test_stop_order_requires_a_stop_price() -> None:
    with pytest.raises(ValueError, match="requires a stop_price"):
        make_order(order_type=OrderType.STOP, limit_price=None, stop_price=None)


def test_stop_limit_requires_both_prices() -> None:
    order = make_order(
        order_type=OrderType.STOP_LIMIT,
        limit_price=Price("100.10"),
        stop_price=Price("100.00"),
    )
    assert order.limit_price == Price("100.10")
    assert order.stop_price == Price("100.00")


@pytest.mark.parametrize(
    ("order_type", "limit", "stop"),
    [
        (OrderType.MARKET, False, False),
        (OrderType.LIMIT, True, False),
        (OrderType.STOP, False, True),
        (OrderType.STOP_LIMIT, True, True),
    ],
)
def test_price_requirements_by_order_type(order_type: OrderType, limit: bool, stop: bool) -> None:
    assert order_type.needs_limit_price is limit
    assert order_type.needs_stop_price is stop


def test_order_rejects_zero_quantity() -> None:
    with pytest.raises(ValueError, match="non-zero quantity"):
        make_order(quantity=Quantity(0))


def test_time_in_force_defaults_to_day() -> None:
    """An order surviving overnight contradicts the day-trading premise."""
    assert make_order().time_in_force is TimeInForce.DAY


# ── Slippage: the sign convention ────────────────────────────


def test_buy_filled_above_reference_is_positive_slippage() -> None:
    """Paying up is a cost."""
    fill = make_fill(side=Side.BUY, price=Price("100.05"), reference_price=Price("100.00"))
    assert fill.slippage_per_share == Decimal("0.05")


def test_sell_filled_below_reference_is_also_positive_slippage() -> None:
    """Giving up edge is the same cost, in the opposite direction."""
    fill = make_fill(side=Side.SELL, price=Price("99.95"), reference_price=Price("100.00"))
    assert fill.slippage_per_share == Decimal("0.05")


def test_price_improvement_is_negative_slippage() -> None:
    """Filling better than the decision price is a genuine gain, not a cost."""
    fill = make_fill(side=Side.BUY, price=Price("99.98"), reference_price=Price("100.00"))
    assert fill.slippage_per_share == Decimal("-0.02")


def test_slippage_costs_do_not_cancel_across_directions() -> None:
    """The bug this convention exists to prevent.

    A naive `price - reference` would make a buy paying up and a sell giving up
    edge sum to zero, so the nightly recalibration would fit against noise and
    conclude execution was free.
    """
    buy = make_fill(side=Side.BUY, price=Price("100.05"), reference_price=Price("100.00"))
    sell = make_fill(side=Side.SELL, price=Price("99.95"), reference_price=Price("100.00"))
    naive = (buy.price - Price("100.00")) + (sell.price - Price("100.00"))
    assert naive == 0  # what the wrong convention would report

    buy_slip, sell_slip = buy.slippage_per_share, sell.slippage_per_share
    assert buy_slip is not None
    assert sell_slip is not None
    assert buy_slip + sell_slip == Decimal("0.10")


def test_slippage_is_none_without_a_reference_price() -> None:
    """An unmeasured cost must not read as zero."""
    fill = make_fill(reference_price=None)
    assert fill.slippage_per_share is None
    assert fill.slippage_cost is None
    assert fill.total_cost is None


def test_slippage_cost_scales_with_size() -> None:
    fill = make_fill(
        side=Side.BUY,
        price=Price("100.05"),
        reference_price=Price("100.00"),
        quantity=Quantity(500),
    )
    assert fill.slippage_cost == Money("25.00", Currency.USD)


def test_total_cost_is_commission_plus_slippage() -> None:
    fill = make_fill(
        side=Side.BUY,
        price=Price("100.05"),
        reference_price=Price("100.00"),
        quantity=Quantity(100),
        commission=Money("1.00", Currency.USD),
    )
    assert fill.slippage_cost == Money("5.00", Currency.USD)
    assert fill.total_cost == Money("6.00", Currency.USD)


# ── Currency correctness ─────────────────────────────────────


def test_notional_is_in_the_instruments_currency() -> None:
    usd = make_fill().notional
    assert usd == Money("10000.00", Currency.USD)


def test_canadian_listing_settles_in_cad() -> None:
    fill = make_fill(
        symbol=SHOP,
        price=Price("142.60"),
        quantity=Quantity(100),
        commission=Money("1.00", Currency.CAD),
    )
    assert fill.notional.currency is Currency.CAD


def test_commission_currency_must_match_the_instrument() -> None:
    """Booking USD commission against a TSX fill would corrupt the equity curve."""
    with pytest.raises(ValueError, match="settles CAD"):
        make_fill(symbol=SHOP, commission=Money("1.00", Currency.USD))


# ── Fills ────────────────────────────────────────────────────


def test_fill_rejects_zero_quantity() -> None:
    with pytest.raises(ValueError, match="non-zero quantity"):
        make_fill(quantity=Quantity(0))


def test_liquidity_defaults_to_unknown() -> None:
    """Not every broker reports it; guessing TAKER would bias the cost model."""
    assert make_fill().liquidity is LiquidityFlag.UNKNOWN


def test_liquidity_is_recorded_when_known() -> None:
    """§5.9 optimises this — 90% TAKER means paying the spread every time."""
    assert make_fill(liquidity=LiquidityFlag.MAKER).liquidity is LiquidityFlag.MAKER


def test_partial_fills_are_separate_records() -> None:
    """One order, several executions, each with its own id and price."""
    first = make_fill(
        id=FillId.derive(order_id=ORDER_ID, broker_exec_id="e.1"),
        broker_exec_id="e.1",
        quantity=Quantity(60),
        price=Price("100.00"),
        seq=0,
    )
    second = make_fill(
        id=FillId.derive(order_id=ORDER_ID, broker_exec_id="e.2"),
        broker_exec_id="e.2",
        quantity=Quantity(40),
        price=Price("100.02"),
        seq=1,
    )
    assert first.id != second.id
    assert first.order_id == second.order_id
    assert first.quantity + second.quantity == Quantity(100)
