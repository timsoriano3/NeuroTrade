"""Tests for Intent.

Two things are load-bearing and both are tested directly: an Intent carries no
size, and its three barriers line up with the triple-barrier labelling in §8.
"""

from __future__ import annotations

from dataclasses import fields
from decimal import Decimal

import pytest

from neurotrade.core.events import Event, MarketEvent
from neurotrade.core.intent import EntryTrigger, Intent
from neurotrade.core.types import Price, Side, Symbol, Venue

AAPL = Symbol("AAPL", Venue.NASDAQ)
NOW = 1_773_495_000_000_000_000  # 2026-03-14 13:30:00 UTC
ONE_HOUR = 3_600_000_000_000


def make_intent(**overrides: object) -> Intent:
    defaults: dict[str, object] = {
        "symbol": AAPL,
        "ts_event": NOW,
        "ts_init": NOW,
        "side": Side.BUY,
        "entry": EntryTrigger.LIMIT,
        "entry_price": Price("100.00"),
        "invalidation": Price("99.00"),
        "target_r": Decimal(2),
        "horizon_ns": ONE_HOUR,
        "strategy": "orb_stocks_in_play",
        "strategy_version": "1.0.0",
        "rationale": "5-min opening range break on 3x relative volume",
    }
    return Intent(**{**defaults, **overrides})  # type: ignore[arg-type]


# ── The absence that matters ─────────────────────────────────


def test_intent_carries_no_size() -> None:
    """Sizing belongs to the risk engine, not to a rule-based strategy (§5.1)."""
    names = {f.name for f in fields(Intent)}
    assert not names & {"quantity", "size", "shares", "notional", "risk_fraction"}


def test_intent_is_not_a_market_event() -> None:
    """It is an output of the system, not an observation of the market.

    Asserted on the classes rather than an instance: mypy proves the instance
    check can never be true, which makes it a test that cannot fail.
    """
    assert issubclass(Intent, Event)
    assert not issubclass(Intent, MarketEvent)


# ── Triple-barrier correspondence (§8) ───────────────────────


def test_the_three_barriers_are_all_present() -> None:
    """Stop, profit and time — the same three §8 labels training examples by."""
    intent = make_intent()
    assert intent.invalidation == Price("99.00")  # stop barrier
    assert intent.target_r == Decimal(2)  # profit barrier
    assert intent.expires_at() == NOW + ONE_HOUR  # time barrier


def test_r_is_the_distance_from_entry_to_invalidation() -> None:
    assert make_intent().risk_per_share(Price("100.00")) == Decimal("1.00")


def test_r_is_computed_from_the_realised_entry_not_the_proposed_one() -> None:
    """A MARKET intent's real R depends on what was actually paid."""
    intent = make_intent(entry=EntryTrigger.MARKET, entry_price=None)
    assert intent.risk_per_share(Price("100.50")) == Decimal("1.50")
    assert intent.risk_per_share(Price("99.50")) == Decimal("0.50")


def test_target_price_for_a_long() -> None:
    """Entry 100, stop 99 → 1R = 1.00 → 2R target = 102."""
    assert make_intent().target_price(Price("100.00")) == Price("102.00")


def test_target_price_for_a_short() -> None:
    """Entry 100, stop 101 → 1R = 1.00 → 2R target = 98."""
    short = make_intent(side=Side.SELL, entry_price=Price("100.00"), invalidation=Price("101.00"))
    assert short.target_price(Price("100.00")) == Price("98.00")


def test_target_price_scales_with_slippage_on_entry() -> None:
    """Worse fill, wider stop distance, further target — R stays coherent."""
    intent = make_intent(entry=EntryTrigger.MARKET, entry_price=None)
    assert intent.target_price(Price("100.00")) == Price("102.00")
    assert intent.target_price(Price("100.50")) == Price("103.50")


def test_zero_risk_is_rejected() -> None:
    """Entry at the stop means a trade with no defined risk — R is undefined."""
    intent = make_intent(entry=EntryTrigger.MARKET, entry_price=None)
    with pytest.raises(ValueError, match="no defined risk"):
        intent.risk_per_share(Price("99.00"))


# ── Entry trigger consistency ────────────────────────────────


def test_market_entry_must_not_carry_a_price() -> None:
    with pytest.raises(ValueError, match="entry_price"):
        make_intent(entry=EntryTrigger.MARKET, entry_price=Price("100"))


@pytest.mark.parametrize("trigger", [EntryTrigger.LIMIT, EntryTrigger.STOP])
def test_non_market_entry_requires_a_price(trigger: EntryTrigger) -> None:
    with pytest.raises(ValueError, match="entry_price"):
        make_intent(entry=trigger, entry_price=None)


def test_market_entry_without_a_price_is_valid() -> None:
    assert make_intent(entry=EntryTrigger.MARKET, entry_price=None).entry_price is None


# ── Direction sanity ─────────────────────────────────────────


def test_long_invalidation_must_sit_below_entry() -> None:
    with pytest.raises(ValueError, match="wrong side"):
        make_intent(side=Side.BUY, entry_price=Price("100"), invalidation=Price("101"))


def test_short_invalidation_must_sit_above_entry() -> None:
    with pytest.raises(ValueError, match="wrong side"):
        make_intent(side=Side.SELL, entry_price=Price("100"), invalidation=Price("99"))


def test_invalidation_equal_to_entry_is_rejected() -> None:
    with pytest.raises(ValueError, match="wrong side"):
        make_intent(side=Side.BUY, entry_price=Price("100"), invalidation=Price("100"))


def test_direction_is_unchecked_for_market_entries() -> None:
    """There is no reference price to check against until it fills."""
    intent = make_intent(
        side=Side.BUY,
        entry=EntryTrigger.MARKET,
        entry_price=None,
        invalidation=Price("500"),
    )
    assert intent.invalidation == Price("500")


# ── Audit fields ─────────────────────────────────────────────


@pytest.mark.parametrize("blank", ["", "   ", "\n"])
def test_rationale_must_not_be_empty(blank: str) -> None:
    """It is an audit record (§6.3) and a dashboard field (§14)."""
    with pytest.raises(ValueError, match="rationale"):
        make_intent(rationale=blank)


@pytest.mark.parametrize("field", ["strategy", "strategy_version"])
def test_authorship_is_required(field: str) -> None:
    with pytest.raises(ValueError, match="identify the author"):
        make_intent(**{field: ""})


def test_intent_records_which_plugin_version_produced_it() -> None:
    """Reconstructing a trade months later needs the exact strategy version."""
    intent = make_intent(strategy="vwap_reclaim", strategy_version="2.3.1")
    assert (intent.strategy, intent.strategy_version) == ("vwap_reclaim", "2.3.1")


# ── Validation of the remaining fields ───────────────────────


@pytest.mark.parametrize("bad", [Decimal(0), Decimal("-1")])
def test_target_r_must_be_positive(bad: Decimal) -> None:
    with pytest.raises(ValueError, match="target_r"):
        make_intent(target_r=bad)


@pytest.mark.parametrize("bad", [0, -1])
def test_horizon_must_be_positive(bad: int) -> None:
    """Every intent expires. A trade with no time barrier is not a day trade."""
    with pytest.raises(ValueError, match="horizon_ns"):
        make_intent(horizon_ns=bad)


def test_intent_is_immutable() -> None:
    with pytest.raises(AttributeError):
        make_intent().target_r = Decimal(5)  # type: ignore[misc]


def test_intents_are_ordered_like_every_other_event() -> None:
    a = make_intent(ts_event=NOW, seq=0)
    b = make_intent(ts_event=NOW, seq=1)
    assert sorted([b, a], key=lambda e: e.sort_key) == [a, b]


def test_reward_to_risk_is_gross_of_costs() -> None:
    """§3.3: a signal must beat spread, fees and slippage, none of which are here."""
    assert make_intent(target_r=Decimal(2)).reward_to_risk == Decimal(2)
