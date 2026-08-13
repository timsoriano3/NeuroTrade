"""Tests for the strategy contract.

Two guarantees carry the weight: a strategy cannot read a feature it did not
declare, and a strategy cannot reach anything that would let it look ahead.
Both are asserted directly rather than inferred from the design.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import fields
from decimal import Decimal

import pytest

from neurotrade.core.events import Bar, BarInterval, MarketSession
from neurotrade.core.intent import EntryTrigger, Intent
from neurotrade.core.registry import DuplicateRegistration, RegistryFrozen
from neurotrade.core.types import Price, Quantity, Side, Symbol, Venue
from neurotrade.strategies.base import (
    FeatureRef,
    Regime,
    Strategy,
    StrategyContext,
    StrategyRegistry,
    UndeclaredFeature,
)

AAPL = Symbol("AAPL", Venue.NASDAQ)
NOW = 1_773_495_000_000_000_000


def context(**overrides: object) -> StrategyContext:
    defaults: dict[str, object] = {
        "symbol": AAPL,
        "as_of": NOW,
        "session": MarketSession.REGULAR,
        "regime": Regime.TREND_UP,
        "values": {"atr": 1.25, "rvol": 3.0},
    }
    return StrategyContext(**{**defaults, **overrides})  # type: ignore[arg-type]


def a_bar() -> Bar:
    return Bar(
        symbol=AAPL,
        ts_event=NOW,
        ts_init=NOW,
        interval=BarInterval.MIN_1,
        open=Price("100"),
        high=Price("101"),
        low=Price("99"),
        close=Price("100.5"),
        volume=Quantity(1_000),
    )


class Breakout(Strategy):
    name = "breakout"
    version = "1.0.0"
    regimes = (Regime.TREND_UP, Regime.TREND_DOWN)
    features = (FeatureRef("atr"), FeatureRef("rvol"))


class Fade(Strategy):
    name = "fade"
    version = "1.0.0"
    regimes = (Regime.CHOP, Regime.REVERSAL)


# ── The context is deliberately narrow ───────────────────────


def test_context_exposes_no_history_clock_or_broker() -> None:
    """Each of these would break a guarantee the rest of the system relies on.

    History permits lookahead, a clock breaks replay determinism, and a broker
    lets a strategy bypass the risk engine entirely.
    """
    exposed = {f.name for f in fields(StrategyContext)}
    assert exposed == {"symbol", "as_of", "session", "regime", "values"}
    assert not exposed & {"history", "bars", "clock", "broker", "portfolio"}


def test_undeclared_features_are_refused() -> None:
    """The declaration is load-bearing, not documentation."""
    with pytest.raises(UndeclaredFeature, match="was not declared"):
        context().feature("vwap")


def test_the_error_lists_what_is_available() -> None:
    with pytest.raises(UndeclaredFeature, match=r"\['atr', 'rvol'\]"):
        context().feature("vwap")


def test_a_warming_feature_reads_as_none_not_zero() -> None:
    """Zero is a real number a strategy would act on; None is 'not yet'."""
    assert context(values={"atr": None}).feature("atr") is None


def test_requires_is_false_while_anything_is_warming() -> None:
    ctx = context(values={"a": 1.0, "b": None})
    assert ctx.requires("a")
    assert not ctx.requires("a", "b")
    assert not ctx.requires("b")


def test_requires_still_refuses_undeclared_names() -> None:
    with pytest.raises(UndeclaredFeature):
        context().requires("atr", "nonexistent")


def test_context_is_immutable() -> None:
    with pytest.raises(AttributeError):
        context().as_of = NOW + 1  # type: ignore[misc]


# ── Regime gating ────────────────────────────────────────────


def test_eligibility_follows_the_declaration() -> None:
    assert Breakout().is_eligible(Regime.TREND_UP)
    assert not Breakout().is_eligible(Regime.CHOP)


def test_opposing_strategies_are_never_eligible_together() -> None:
    """ORB and ORB-fade are opposites; §5.7's gating exists to keep them apart."""
    for regime in Regime:
        assert not (Breakout().is_eligible(regime) and Fade().is_eligible(regime))


def test_undeclared_regimes_grant_nothing() -> None:
    """The safe default for a subclass that forgets to declare."""

    class Forgetful(Strategy):
        name, version = "forgetful", "1.0.0"

    assert not any(Forgetful().is_eligible(regime) for regime in Regime)


def test_unknown_regime_grants_nothing() -> None:
    """Before the Phase 5 classifier exists, every regime is UNKNOWN."""
    assert not Breakout().is_eligible(Regime.UNKNOWN)
    assert not Fade().is_eligible(Regime.UNKNOWN)


# ── Handlers ─────────────────────────────────────────────────


def test_handlers_default_to_producing_nothing() -> None:
    """A bar-driven strategy should not have to stub quote and trade handlers."""
    strategy = Breakout()
    assert strategy.on_bar(a_bar(), context()) == ()
    assert strategy.on_trade.__func__ is Strategy.on_trade  # type: ignore[attr-defined]


def test_a_strategy_can_produce_an_intent() -> None:
    class Simple(Strategy):
        name, version = "simple", "1.0.0"
        regimes = (Regime.TREND_UP,)

        def on_bar(self, bar: Bar, context: StrategyContext) -> Sequence[Intent]:
            return (
                Intent(
                    symbol=context.symbol,
                    ts_event=bar.ts_event,
                    ts_init=bar.ts_event,
                    side=Side.BUY,
                    entry=EntryTrigger.STOP,
                    entry_price=bar.high,
                    invalidation=bar.low,
                    target_r=Decimal(2),
                    horizon_ns=3_600_000_000_000,
                    strategy=self.name,
                    strategy_version=self.version,
                    rationale="bar broke its own high",
                ),
            )

    intents = Simple().on_bar(a_bar(), context())
    assert len(intents) == 1
    assert (intents[0].strategy, intents[0].strategy_version) == ("simple", "1.0.0")


def test_qualified_name_matches_what_intents_record() -> None:
    """Reconstructing a trade months later needs the exact plugin version."""
    assert Breakout().qualified_name == "breakout@1.0.0"


def test_intents_carry_no_size() -> None:
    """§5.1 — a strategy proposes; the risk engine sizes."""
    assert "quantity" not in {f.name for f in fields(Intent)}


# ── Registry ─────────────────────────────────────────────────


def test_registration_takes_identity_from_the_class() -> None:
    """A strategy cannot be registered under a name it does not stamp on Intents."""
    registry = StrategyRegistry()
    registry.strategy(Breakout)
    assert registry.get("breakout") is Breakout
    assert registry.snapshot() == ("breakout@1.0.0",)


def test_a_class_without_a_name_is_rejected() -> None:
    class Nameless(Strategy):
        version = "1.0.0"

    with pytest.raises(ValueError, match="must declare a name"):
        StrategyRegistry().strategy(Nameless)


def test_a_class_without_a_version_is_rejected() -> None:
    class Unversioned(Strategy):
        name = "unversioned"

    with pytest.raises(ValueError, match="must declare a version"):
        StrategyRegistry().strategy(Unversioned)


def test_duplicate_registration_is_rejected() -> None:
    registry = StrategyRegistry()
    registry.strategy(Breakout)
    with pytest.raises(DuplicateRegistration):
        registry.strategy(Breakout)


def test_champion_and_challenger_coexist() -> None:
    """§10.2 runs both on identical input."""

    class BreakoutV2(Strategy):
        name, version = "breakout", "1.1.0"
        regimes = (Regime.TREND_UP,)

    registry = StrategyRegistry()
    registry.strategy(Breakout)
    registry.strategy(BreakoutV2)
    assert registry.get("breakout", "1.0.0") is Breakout
    assert registry.get("breakout") is BreakoutV2  # latest
    assert len(registry) == 2


def test_freeze_closes_registration() -> None:
    registry = StrategyRegistry()
    registry.freeze()
    with pytest.raises(RegistryFrozen):
        registry.strategy(Breakout)


def test_eligible_filters_by_regime() -> None:
    registry = StrategyRegistry()
    registry.strategy(Breakout)
    registry.strategy(Fade)
    assert [cls.name for cls in registry.eligible(Regime.CHOP)] == ["fade"]
    assert [cls.name for cls in registry.eligible(Regime.TREND_UP)] == ["breakout"]
    assert registry.eligible(Regime.UNKNOWN) == ()


def test_required_features_is_the_deduplicated_union() -> None:
    """The engine computes these and nothing else."""

    class Other(Strategy):
        name, version = "other", "1.0.0"
        features = (FeatureRef("atr"), FeatureRef("vwap"))

    registry = StrategyRegistry()
    registry.strategy(Breakout)  # atr, rvol
    registry.strategy(Other)  # atr, vwap
    assert [str(ref) for ref in registry.required_features()] == ["atr", "rvol", "vwap"]


def test_required_features_distinguishes_pinned_versions() -> None:
    """A strategy pinned to atr@1.0.0 needs a different resolution than 'latest'."""

    class Pinned(Strategy):
        name, version = "pinned", "1.0.0"
        features = (FeatureRef("atr", "1.0.0"),)

    registry = StrategyRegistry()
    registry.strategy(Breakout)  # FeatureRef("atr") — latest
    registry.strategy(Pinned)  # FeatureRef("atr", "1.0.0")
    assert [str(ref) for ref in registry.required_features()] == ["atr", "atr@1.0.0", "rvol"]


def test_registry_holds_classes_not_instances() -> None:
    """The host builds one strategy per symbol; a shared instance would share state."""
    registry = StrategyRegistry()
    registry.strategy(Breakout)
    found = registry.get("breakout")
    assert isinstance(found, type)
    assert found() is not found()


def test_membership_and_names() -> None:
    registry = StrategyRegistry()
    registry.strategy(Breakout)
    assert "breakout" in registry
    assert "nope" not in registry
    assert registry.names() == ("breakout",)
