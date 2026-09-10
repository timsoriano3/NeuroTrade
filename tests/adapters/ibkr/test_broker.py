"""Tests for the IBKR broker.

The guard tests come first because they are the ones that matter. This is the
only component that can move money, and the property worth proving is not that
it places orders — it is that it refuses to when it should.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from neurotrade.adapters.ibkr.broker import (
    IbkrBroker,
    LiveOrdersNotPermitted,
    OrderRejected,
)
from neurotrade.adapters.ibkr.connection import IbkrConnection
from neurotrade.config import IbkrSettings
from neurotrade.core.clock import SimClock
from neurotrade.core.ids import IntentId, OrderId
from neurotrade.core.orders import Fill, LiquidityFlag, Order, OrderType, TimeInForce
from neurotrade.core.ports import BrokerPort
from neurotrade.core.types import Currency, Money, Price, Quantity, Side, Symbol, Venue
from tests.adapters.ibkr.conftest import FakeIB, an_execution

AAPL = Symbol("AAPL", Venue.NASDAQ)
SHOP = Symbol("SHOP", Venue.TSX)
NOW = 1_773_495_000_000_000_000

INTENT = IntentId.derive(strategy="orb", strategy_version="1.0.0", symbol=AAPL, ts_event=NOW, seq=0)

PAPER_PORTS = (4002, 7497)
LIVE_PORTS = (4001, 7496)


def an_order(**overrides: object) -> Order:
    defaults: dict[str, object] = {
        "id": OrderId.derive(intent_id=INTENT, ts_event=NOW),
        "intent_id": INTENT,
        "symbol": AAPL,
        "ts_event": NOW,
        "ts_init": NOW,
        "side": Side.BUY,
        "quantity": Quantity(100),
        "order_type": OrderType.LIMIT,
        "limit_price": Price("315.00"),
        "config_hash": "cfg_d842e94e473f9869",
    }
    return Order(**{**defaults, **overrides})  # type: ignore[arg-type]


def a_broker(
    *, port: int = 4002, allow_live_orders: bool = False, **fake: object
) -> tuple[IbkrBroker, FakeIB, list[Fill]]:
    ib = FakeIB(**fake)  # type: ignore[arg-type]
    connection = IbkrConnection(IbkrSettings(port=port), ib=ib)
    fills: list[Fill] = []
    broker = IbkrBroker(
        connection,
        SimClock(NOW),
        allow_live_orders=allow_live_orders,
        on_fill=fills.append,
    )
    return broker, ib, fills


# ── The guard ────────────────────────────────────────────────


@pytest.mark.parametrize("port", LIVE_PORTS)
async def test_a_live_order_is_refused_without_permission(port: int) -> None:
    """The structural guard: unable to trade, not merely configured not to."""
    broker, ib, _ = a_broker(port=port, allow_live_orders=False)
    with pytest.raises(LiveOrdersNotPermitted, match="allow_live_orders is false"):
        await broker.submit(an_order())
    assert not ib.placed  # nothing reached IBKR


@pytest.mark.parametrize("port", PAPER_PORTS)
async def test_paper_orders_need_no_permission(port: int) -> None:
    """Paper money is not real money; gate G2 runs here."""
    broker, ib, _ = a_broker(port=port, allow_live_orders=False)
    await broker.submit(an_order())
    assert len(ib.placed) == 1


@pytest.mark.parametrize("port", LIVE_PORTS)
async def test_a_live_order_is_allowed_with_permission(port: int) -> None:
    broker, ib, _ = a_broker(port=port, allow_live_orders=True)
    await broker.submit(an_order())
    assert len(ib.placed) == 1


async def test_the_guard_checks_the_port_not_the_profile_name() -> None:
    """What makes an order real is which Gateway answered.

    A profile labelled `paper` but pointed at 4001 would otherwise trade real
    money, and the label would be the only thing saying it did not.
    """
    broker, ib, _ = a_broker(port=4001, allow_live_orders=False)
    with pytest.raises(LiveOrdersNotPermitted, match="4001"):
        await broker.submit(an_order())
    assert not ib.placed


def test_permission_must_be_stated() -> None:
    """No default: a caller has to say what it is permitting."""
    connection = IbkrConnection(IbkrSettings(), ib=FakeIB())
    with pytest.raises(TypeError):
        IbkrBroker(connection, SimClock(NOW))  # type: ignore[call-arg]


# ── Conformance ──────────────────────────────────────────────


def test_satisfies_the_broker_port() -> None:
    broker, _, _ = a_broker()
    assert isinstance(broker, BrokerPort)


def test_submit_returns_nothing() -> None:
    """A return value would be a lie for every order that does not fill instantly.

    Asserted on the signature: mypy proves the runtime check can never fail, so
    checking the value would be a test that cannot fail.
    """
    from typing import get_type_hints

    assert get_type_hints(IbkrBroker.submit)["return"] is type(None)


# ── Order translation ────────────────────────────────────────


@pytest.mark.parametrize(
    ("order_type", "extra", "expected"),
    [
        (OrderType.MARKET, {"limit_price": None}, "MKT"),
        (OrderType.LIMIT, {}, "LMT"),
        (OrderType.STOP, {"limit_price": None, "stop_price": Price("310")}, "STP"),
        (
            OrderType.STOP_LIMIT,
            {"stop_price": Price("310")},
            "STP LMT",
        ),
    ],
)
async def test_every_order_type_translates(
    order_type: OrderType, extra: dict[str, object], expected: str
) -> None:
    broker, ib, _ = a_broker()
    await broker.submit(an_order(order_type=order_type, **extra))
    assert ib.placed[0].orderType == expected


async def test_side_translates() -> None:
    broker, ib, _ = a_broker()
    await broker.submit(an_order(side=Side.SELL))
    assert ib.placed[0].action == "SELL"


async def test_time_in_force_is_carried() -> None:
    broker, ib, _ = a_broker()
    await broker.submit(an_order(time_in_force=TimeInForce.IOC))
    assert ib.placed[0].tif == "IOC"


async def test_our_order_id_travels_with_the_order() -> None:
    """So an execution report ties back to the intent without a lookup table."""
    order = an_order()
    broker, ib, _ = a_broker()
    await broker.submit(order)
    assert ib.placed[0].orderRef == order.id.value


async def test_prices_reach_ibkr_as_floats() -> None:
    """The one place precision is deliberately given up — IBKR's wire protocol
    has no decimal type. The order record keeps the exact value."""
    order = an_order(limit_price=Price("315.07"))
    broker, ib, _ = a_broker()
    await broker.submit(order)
    assert ib.placed[0].lmtPrice == 315.07
    assert order.limit_price is not None
    assert order.limit_price.value == Decimal("315.07")  # still exact on the record


# ── Fills ────────────────────────────────────────────────────


async def test_an_execution_becomes_a_fill() -> None:
    order = an_order()
    broker, ib, fills = a_broker()
    await broker.submit(order)
    ib.trades[0].fillEvent.emit(an_execution(shares=100.0, price=315.93))

    assert len(fills) == 1
    fill = fills[0]
    assert fill.order_id == order.id
    assert fill.price.value == Decimal("315.93")
    assert fill.quantity == Quantity(100)
    assert fill.commission == Money("1.05", Currency.USD)


async def test_the_same_execution_report_twice_is_one_fill() -> None:
    """IBKR re-sends execution reports on reconnect.

    The ids must match, or a reconnect doubles the position.
    """
    broker, ib, fills = a_broker()
    await broker.submit(an_order())
    ib.trades[0].fillEvent.emit(an_execution("abc.1"))
    ib.trades[0].fillEvent.emit(an_execution("abc.1"))
    assert len({fill.id for fill in fills}) == 1


async def test_separate_executions_are_separate_fills() -> None:
    """A partially filled order produces several."""
    broker, ib, fills = a_broker()
    await broker.submit(an_order())
    ib.trades[0].fillEvent.emit(an_execution("abc.1", shares=60.0))
    ib.trades[0].fillEvent.emit(an_execution("abc.2", shares=40.0))
    assert len({fill.id for fill in fills}) == 2
    assert fills[0].quantity + fills[1].quantity == Quantity(100)


async def test_ibkrs_bot_and_sld_translate_to_our_sides() -> None:
    broker, ib, fills = a_broker()
    await broker.submit(an_order())
    ib.trades[0].fillEvent.emit(an_execution("a.1", side="BOT"))
    ib.trades[0].fillEvent.emit(an_execution("a.2", side="SLD"))
    assert [fill.side for fill in fills] == [Side.BUY, Side.SELL]


@pytest.mark.parametrize(
    ("last_liquidity", "expected"),
    [(1, LiquidityFlag.MAKER), (2, LiquidityFlag.TAKER), (0, LiquidityFlag.UNKNOWN)],
)
async def test_liquidity_is_reported_or_left_unknown(
    last_liquidity: int, expected: LiquidityFlag
) -> None:
    """Guessing biases the cost model toward execution looking cheaper."""
    broker, ib, fills = a_broker()
    await broker.submit(an_order())
    ib.trades[0].fillEvent.emit(an_execution(liquidity=last_liquidity))
    assert fills[0].liquidity is expected


async def test_a_canadian_fill_books_commission_in_cad() -> None:
    """The account's base currency is CAD, so this is the ordinary case."""
    broker, ib, fills = a_broker()
    await broker.submit(an_order(symbol=SHOP, limit_price=Price("180")))
    ib.trades[0].fillEvent.emit(an_execution(currency="CAD", commission=1.25))
    assert fills[0].commission == Money("1.25", Currency.CAD)
    assert fills[0].symbol.currency is Currency.CAD


async def test_ts_init_records_when_we_learned_of_the_fill() -> None:
    """Distinct from ts_event, which is when it happened at the venue."""
    broker, ib, fills = a_broker()
    await broker.submit(an_order())
    ib.trades[0].fillEvent.emit(an_execution())
    assert fills[0].ts_init == NOW
    assert fills[0].ts_event != fills[0].ts_init


async def test_fills_are_dropped_when_nobody_is_listening() -> None:
    """Correct for a probe, wrong for a trading session — hence the explicit
    `on_fill` rather than a default."""
    ib = FakeIB()
    broker = IbkrBroker(
        IbkrConnection(IbkrSettings(), ib=ib), SimClock(NOW), allow_live_orders=False
    )
    await broker.submit(an_order())
    ib.trades[0].fillEvent.emit(an_execution())  # must not raise


# ── Cancellation ─────────────────────────────────────────────


async def test_cancelling_a_working_order_reaches_ibkr() -> None:
    order = an_order()
    broker, ib, _ = a_broker()
    await broker.submit(order)
    await broker.cancel(order.id)
    assert len(ib.cancelled) == 1


async def test_cancelling_an_unknown_order_is_not_an_error() -> None:
    """An order already filled or cancelled is not an error to cancel again."""
    broker, ib, _ = a_broker()
    await broker.cancel(OrderId.derive(intent_id=INTENT, ts_event=NOW, attempt=9))
    assert not ib.cancelled


# ── Failure modes ────────────────────────────────────────────


async def test_a_rejection_is_reported_not_swallowed() -> None:
    """Rejection is normal — margin, a locked symbol, a broker risk limit — and
    must not be treated as a system fault."""
    broker, _, _ = a_broker(place_error=RuntimeError("insufficient margin"))
    with pytest.raises(OrderRejected, match="insufficient margin"):
        await broker.submit(an_order())


def test_every_tradable_venue_is_mapped() -> None:
    """The unmapped-venue branch should be unreachable, and this keeps it so.

    Adding a `Venue` without adding its IBKR name would otherwise fail at
    submission time, for one venue, in production. SMART is excluded because
    `Symbol` refuses it: it is a router, not a listing.
    """
    from neurotrade.adapters.ibkr.market_data import IBKR_EXCHANGE

    tradable = {venue for venue in Venue if venue is not Venue.SMART}
    assert tradable <= set(IBKR_EXCHANGE), f"unmapped: {tradable - set(IBKR_EXCHANGE)}"


# ── Gate G2, against a real Gateway ──────────────────────────


@pytest.mark.ibkr
async def test_a_paper_order_round_trips() -> None:
    """Gate G2: submit, get an acknowledgement, cancel.

    A buy limit far below the market, so it cannot fill: the point is to prove
    the order path works, not to acquire a position. Run with
    `uv run pytest -m ibkr` and Gateway logged in.
    """
    import asyncio

    from neurotrade.config import Profile, load_settings

    settings = load_settings(Profile.PAPER)
    connection = IbkrConnection(IbkrSettings(**{**settings.ibkr.model_dump(), "client_id": 79}))
    fills: list[Fill] = []
    broker = IbkrBroker(
        connection,
        SimClock(NOW),
        allow_live_orders=settings.allow_live_orders,
        on_fill=fills.append,
    )

    order = an_order(
        quantity=Quantity(1),
        limit_price=Price("1.00"),  # far below the market; cannot fill
    )
    try:
        await broker.submit(order)
        trade = broker.working(order.id)
        assert trade is not None

        for _ in range(40):  # up to ten seconds
            await asyncio.sleep(0.25)
            if trade.orderStatus.status in {"Submitted", "PreSubmitted", "Cancelled"}:
                break
        acknowledged = trade.orderStatus.status
        assert acknowledged in {"Submitted", "PreSubmitted"}, acknowledged

        await broker.cancel(order.id)
        for _ in range(40):
            await asyncio.sleep(0.25)
            if trade.orderStatus.status in {"Cancelled", "ApiCancelled"}:
                break
        assert trade.orderStatus.status in {"Cancelled", "ApiCancelled"}
    finally:
        connection.disconnect()
