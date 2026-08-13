"""Shared pytest configuration.

Provides the namespace that docstring examples run in. A doctest only sees its
own module's globals, so an example needing a name the module does not import
would otherwise have to spend two lines importing it — and examples that long
stop being examples.

The names injected here are the ones a reader would naturally have in scope:
the core value types, plus a ready-made `demo_intent` so that methods on
`Intent` can be demonstrated in one line instead of eight.

Anything injected here must be listed in the docstring below, because an example
that depends on invisible state is worse than no example at all.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from neurotrade.core.events import MarketSession
from neurotrade.core.ids import FillId, IntentId, OrderId
from neurotrade.core.intent import EntryTrigger, Intent
from neurotrade.core.orders import Fill
from neurotrade.core.position import Position
from neurotrade.core.types import Currency, Money, Price, Quantity, Side, Symbol, Venue

AAPL = Symbol("AAPL", Venue.NASDAQ)

#: A long AAPL proposal: enter at 100, stop at 99 (so 1R = 1.00), target 2R,
#: one-hour time barrier. Injected as `demo_intent`.
DEMO_INTENT = Intent(
    symbol=AAPL,
    ts_event=1_000,
    ts_init=1_000,
    side=Side.BUY,
    entry=EntryTrigger.LIMIT,
    entry_price=Price("100"),
    invalidation=Price("99"),
    target_r=Decimal(2),
    horizon_ns=3_600_000_000_000,
    strategy="orb_stocks_in_play",
    strategy_version="1.0.0",
    rationale="5-minute opening range break on 3x relative volume",
)

_DEMO_ORDER_ID = OrderId.derive(
    intent_id=IntentId.derive(
        strategy="orb_stocks_in_play",
        strategy_version="1.0.0",
        symbol=AAPL,
        ts_event=1_000,
        seq=0,
    ),
    ts_event=1_000,
)

#: A buy of 100 AAPL filled at 100.05 against a decision price of 100.00 — so it
#: slipped 0.05 per share — with 1.00 commission. Injected as `demo_fill`.
DEMO_FILL = Fill(
    id=FillId.derive(order_id=_DEMO_ORDER_ID, broker_exec_id="demo.1"),
    order_id=_DEMO_ORDER_ID,
    symbol=AAPL,
    ts_event=1_000,
    ts_init=1_000,
    side=Side.BUY,
    price=Price("100.05"),
    quantity=Quantity(100),
    commission=Money("1.00", Currency.USD),
    reference_price=Price("100.00"),
    broker_exec_id="demo.1",
)


def _closing_fill() -> Fill:
    """The other half of DEMO_FILL: sell 100 back at 102.00, 1.00 commission."""
    return Fill(
        id=FillId.derive(order_id=_DEMO_ORDER_ID, broker_exec_id="demo.2"),
        order_id=_DEMO_ORDER_ID,
        symbol=AAPL,
        ts_event=2_000,
        ts_init=2_000,
        side=Side.SELL,
        price=Price("102.05"),
        quantity=Quantity(100),
        commission=Money("1.00", Currency.USD),
        broker_exec_id="demo.2",
    )


#: A completed round trip: bought 100 at 100.05, sold at 102.05, so 200.00 gross
#: and 2.00 of commission, leaving 198.00 net. Injected as `demo_round_trip`.
DEMO_ROUND_TRIP = Position.flat(AAPL).apply(DEMO_FILL).apply(_closing_fill())


@pytest.fixture(autouse=True)
def _doctest_namespace(doctest_namespace: dict[str, Any]) -> None:
    """Inject shared names into every docstring example.

    Available in any `>>>` block:

    - `Decimal`
    - `Price`, `Quantity`, `Side`, `Symbol`, `Venue` — core value types
    - `AAPL` — `Symbol("AAPL", Venue.NASDAQ)`
    - `MarketSession`
    - `demo_intent` — the proposal described on `DEMO_INTENT` above
    - `demo_fill` — the execution described on `DEMO_FILL` above
    - `demo_round_trip` — the closed position described on `DEMO_ROUND_TRIP` above
    """
    doctest_namespace.update(
        Decimal=Decimal,
        Currency=Currency,
        Money=Money,
        Price=Price,
        Quantity=Quantity,
        Side=Side,
        Symbol=Symbol,
        Venue=Venue,
        MarketSession=MarketSession,
        AAPL=AAPL,
        demo_intent=DEMO_INTENT,
        demo_fill=DEMO_FILL,
        demo_round_trip=DEMO_ROUND_TRIP,
    )
