"""Tests for market data events.

The ordering tests matter most: `(ts_event, seq)` being a total order is what
gate G1 rests on.
"""

from __future__ import annotations

import random
from dataclasses import replace
from decimal import Decimal

import pytest

from neurotrade.core.events import Bar, BarInterval, MarketEvent, Quote, TickTrade
from neurotrade.core.types import Price, Quantity, Side, Symbol, Venue

AAPL = Symbol("AAPL", Venue.NASDAQ)
OPEN_NS = 1_773_495_000_000_000_000  # 2026-03-14 13:30:00 UTC


def make_bar(**overrides: object) -> Bar:
    defaults: dict[str, object] = {
        "symbol": AAPL,
        "ts_event": OPEN_NS,
        "ts_init": OPEN_NS,
        "interval": BarInterval.MIN_1,
        "open": Price("100"),
        "high": Price("102"),
        "low": Price("99"),
        "close": Price("101"),
        "volume": Quantity(1_000),
    }
    return Bar(**{**defaults, **overrides})  # type: ignore[arg-type]


def make_quote(**overrides: object) -> Quote:
    defaults: dict[str, object] = {
        "symbol": AAPL,
        "ts_event": OPEN_NS,
        "ts_init": OPEN_NS,
        "bid_price": Price("100.00"),
        "bid_size": Quantity(500),
        "ask_price": Price("100.02"),
        "ask_size": Quantity(300),
    }
    return Quote(**{**defaults, **overrides})  # type: ignore[arg-type]


# ── Ordering — the property gate G1 depends on ───────────────


def test_sort_key_is_ts_event_then_seq() -> None:
    assert make_bar(seq=3).sort_key == (OPEN_NS, 3)


def test_events_sharing_a_timestamp_order_by_seq() -> None:
    """Two prints can land on the same nanosecond; seq breaks the tie."""
    events = [make_bar(seq=2), make_bar(seq=0), make_bar(seq=1)]
    assert [e.seq for e in sorted(events, key=lambda e: e.sort_key)] == [0, 1, 2]


def test_ordering_is_total_and_shuffle_independent() -> None:
    """The same set of events must sort identically regardless of input order."""
    events = [make_bar(ts_event=OPEN_NS + t, seq=s) for t in (0, 1_000, 2_000) for s in (0, 1)]
    canonical = [e.sort_key for e in sorted(events, key=lambda e: e.sort_key)]

    rng = random.Random(1234)
    for _ in range(50):
        shuffled = events[:]
        rng.shuffle(shuffled)
        assert [e.sort_key for e in sorted(shuffled, key=lambda e: e.sort_key)] == canonical


def test_seq_must_not_be_negative() -> None:
    with pytest.raises(ValueError, match="seq"):
        make_bar(seq=-1)


# ── Timestamps ───────────────────────────────────────────────


def test_latency_is_init_minus_event() -> None:
    assert make_bar(ts_event=OPEN_NS, ts_init=OPEN_NS + 5_000_000).latency_ns == 5_000_000


def test_negative_latency_is_allowed() -> None:
    """Clock skew is a thing to measure, not a reason to discard data."""
    assert make_bar(ts_event=OPEN_NS, ts_init=OPEN_NS - 1_000).latency_ns == -1_000


def test_events_are_immutable() -> None:
    with pytest.raises(AttributeError):
        make_bar().close = Price("999")  # type: ignore[misc]


def test_replace_produces_a_new_validated_event() -> None:
    bar = make_bar()
    assert replace(bar, seq=7).seq == 7
    with pytest.raises(ValueError):
        replace(bar, high=Price("1"))  # would sit below low


# ── Bar ──────────────────────────────────────────────────────


def test_bar_ts_event_is_close_time_and_ts_open_derives_from_it() -> None:
    """Stamping a bar with its open time is the classic lookahead bug."""
    bar = make_bar(interval=BarInterval.MIN_1)
    assert bar.ts_open == OPEN_NS - 60_000_000_000


@pytest.mark.parametrize(
    ("interval", "nanos"),
    [
        (BarInterval.SEC_1, 1_000_000_000),
        (BarInterval.MIN_1, 60_000_000_000),
        (BarInterval.MIN_5, 300_000_000_000),
        (BarInterval.HOUR_1, 3_600_000_000_000),
        (BarInterval.DAY_1, 86_400_000_000_000),
    ],
)
def test_bar_interval_durations(interval: BarInterval, nanos: int) -> None:
    assert interval.nanos == nanos


def test_every_interval_has_a_duration() -> None:
    for interval in BarInterval:
        assert interval.nanos > 0


def test_bar_rejects_high_below_low() -> None:
    with pytest.raises(ValueError, match="below low"):
        make_bar(high=Price("98"), low=Price("99"))


@pytest.mark.parametrize("field", ["open", "close"])
def test_bar_rejects_open_or_close_outside_the_range(field: str) -> None:
    with pytest.raises(ValueError, match="outside range"):
        make_bar(**{field: Price("500")})


def test_bar_rejects_vwap_outside_the_range() -> None:
    with pytest.raises(ValueError, match="vwap"):
        make_bar(vwap=Price("500"))


def test_bar_accepts_a_flat_bar() -> None:
    """A symbol that never traded has open == high == low == close."""
    flat = make_bar(open=Price("100"), high=Price("100"), low=Price("100"), close=Price("100"))
    assert flat.range == Decimal(0)
    assert not flat.is_up


def test_bar_range_and_direction() -> None:
    assert make_bar().range == Decimal(3)
    assert make_bar(open=Price("100"), close=Price("101")).is_up
    assert not make_bar(open=Price("101"), close=Price("100")).is_up


def test_bar_rejects_negative_trade_count() -> None:
    with pytest.raises(ValueError, match="trade_count"):
        make_bar(trade_count=-1)


def test_bar_allows_zero_volume() -> None:
    """Illiquid names genuinely produce zero-volume minutes."""
    assert make_bar(volume=Quantity(0)).volume.is_zero


# ── Quote ────────────────────────────────────────────────────


def test_quote_spread_and_mid() -> None:
    quote = make_quote()
    assert quote.spread == Decimal("0.02")
    assert quote.mid == Decimal("100.01")


def test_microprice_leans_toward_the_thin_side() -> None:
    """Heavy bid, thin ask → price more likely to move up, so microprice > mid."""
    quote = make_quote(bid_size=Quantity(900), ask_size=Quantity(100))
    assert quote.microprice > quote.mid


def test_microprice_equals_mid_when_sizes_balance() -> None:
    quote = make_quote(bid_size=Quantity(500), ask_size=Quantity(500))
    assert quote.microprice == quote.mid


def test_microprice_falls_back_to_mid_when_book_is_empty() -> None:
    quote = make_quote(bid_size=Quantity(0), ask_size=Quantity(0))
    assert quote.microprice == quote.mid


def test_locked_book_is_flagged_not_rejected() -> None:
    quote = make_quote(bid_price=Price("100"), ask_price=Price("100"))
    assert quote.is_locked
    assert not quote.is_crossed
    assert quote.spread == Decimal(0)


def test_crossed_book_is_flagged_not_rejected() -> None:
    """Crossed books occur in real consolidated data around fast moves."""
    quote = make_quote(bid_price=Price("100.05"), ask_price=Price("100.00"))
    assert quote.is_crossed
    assert quote.spread < 0


# ── TickTrade ────────────────────────────────────────────────


def test_trade_notional() -> None:
    trade = TickTrade(
        symbol=AAPL,
        ts_event=OPEN_NS,
        ts_init=OPEN_NS,
        price=Price("100.50"),
        size=Quantity(200),
    )
    assert trade.notional == Decimal("20100.00")


def test_trade_aggressor_defaults_to_unknown() -> None:
    """The tape does not carry it; it is inferred later (§5.6)."""
    trade = TickTrade(
        symbol=AAPL, ts_event=OPEN_NS, ts_init=OPEN_NS, price=Price("1"), size=Quantity(1)
    )
    assert trade.aggressor is None


def test_trade_aggressor_can_be_set() -> None:
    trade = TickTrade(
        symbol=AAPL,
        ts_event=OPEN_NS,
        ts_init=OPEN_NS,
        price=Price("1"),
        size=Quantity(1),
        aggressor=Side.BUY,
    )
    assert trade.aggressor is Side.BUY


def test_trade_rejects_zero_size() -> None:
    with pytest.raises(ValueError, match="non-zero size"):
        TickTrade(
            symbol=AAPL,
            ts_event=OPEN_NS,
            ts_init=OPEN_NS,
            price=Price("1"),
            size=Quantity(0),
        )


# ── Construction discipline ──────────────────────────────────


@pytest.mark.parametrize(
    "build",
    [
        lambda: make_bar(seq=-1),
        lambda: TickTrade(
            symbol=AAPL,
            ts_event=OPEN_NS,
            ts_init=OPEN_NS,
            price=Price("1"),
            size=Quantity(1),
            seq=-1,
        ),
        lambda: make_quote(seq=-1),
    ],
)
def test_subclasses_run_base_class_validation(build: object) -> None:
    """Every subclass must inherit the base's checks.

    Not decorative: `super().__post_init__()` raises TypeError under
    `@dataclass(slots=True)`, and the tempting "fix" of dropping the call
    entirely would leave subclasses silently unvalidated.
    """
    with pytest.raises(ValueError, match="seq"):
        build()  # type: ignore[operator]


def test_events_are_keyword_only() -> None:
    """Positional construction of several near-identical integers invites bugs."""
    with pytest.raises(TypeError):
        MarketEvent(AAPL, OPEN_NS, OPEN_NS, 0)  # type: ignore[call-arg]
