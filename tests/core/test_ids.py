"""Tests for derived identifiers.

The determinism tests are the point. Everything else in the replay harness can
be correct and a single `uuid4` in here would still make gate G1 unachievable.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from neurotrade.core.ids import FillId, IntentId, OrderId
from neurotrade.core.types import Symbol, Venue

AAPL = Symbol("AAPL", Venue.NASDAQ)
NOW = 1_773_495_000_000_000_000


def an_intent_id(**overrides: object) -> IntentId:
    kwargs: dict[str, object] = {
        "strategy": "orb_stocks_in_play",
        "strategy_version": "1.0.0",
        "symbol": AAPL,
        "ts_event": NOW,
        "seq": 0,
    }
    return IntentId.derive(**{**kwargs, **overrides})  # type: ignore[arg-type]


# ── Determinism — what gate G1 rests on ──────────────────────


def test_same_inputs_derive_the_same_id() -> None:
    assert an_intent_id() == an_intent_id()


def test_ids_are_stable_across_processes_and_hash_seeds() -> None:
    """Python salts hash() per process; BLAKE2b must not be affected.

    Runs the same derivation in two subprocesses with different PYTHONHASHSEED
    values. If this ever fails, replay determinism is gone and the cause will be
    invisible from inside a single process.
    """
    script = (
        "from neurotrade.core.ids import IntentId;"
        "from neurotrade.core.types import Symbol, Venue;"
        "print(IntentId.derive(strategy='s', strategy_version='1.0.0',"
        f" symbol=Symbol('AAPL', Venue.NASDAQ), ts_event={NOW}, seq=0))"
    )
    outputs = {
        subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
            env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
        ).stdout.strip()
        for seed in ("0", "1", "random")
    }
    assert len(outputs) == 1, f"id varied across hash seeds: {outputs}"


def test_derivation_is_not_time_dependent() -> None:
    """A UUID1 would embed the wall clock; this must not."""
    first = an_intent_id()
    second = an_intent_id()
    assert first == second
    assert first.value == second.value


# ── Distinctness ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "change",
    [
        {"strategy": "vwap_reclaim"},
        {"strategy_version": "1.0.1"},
        {"symbol": Symbol("MSFT", Venue.NASDAQ)},
        {"ts_event": NOW + 1},
        {"seq": 1},
    ],
)
def test_every_input_changes_the_id(change: dict[str, object]) -> None:
    assert an_intent_id(**change) != an_intent_id()


def test_same_ticker_on_different_venues_derives_different_ids() -> None:
    """Symbol carries its venue, so TD.TSX and TD.NASDAQ cannot collide."""
    assert an_intent_id(symbol=Symbol("TD", Venue.TSX)) != an_intent_id(
        symbol=Symbol("TD", Venue.NASDAQ)
    )


def test_field_boundaries_cannot_be_confused() -> None:
    """("AB","C") and ("A","BC") must not collide — hence the \\x1f separator."""
    assert an_intent_id(strategy="ab", strategy_version="c") != an_intent_id(
        strategy="a", strategy_version="bc"
    )


def test_repeating_the_same_proposal_is_one_id_not_two() -> None:
    """A strategy firing twice on the same bar proposed one thing, not two."""
    assert an_intent_id(seq=0) == an_intent_id(seq=0)


# ── Type distinctness ────────────────────────────────────────


def test_id_kinds_have_distinct_prefixes() -> None:
    intent = an_intent_id()
    order = OrderId.derive(intent_id=intent, ts_event=NOW)
    fill = FillId.derive(order_id=order, broker_exec_id="0000e0d5.68a1b2c3.01.01")
    assert str(intent).startswith("int_")
    assert str(order).startswith("ord_")
    assert str(fill).startswith("fil_")


def test_an_id_rejects_the_wrong_prefix() -> None:
    """Type distinctness is enforced at runtime as well as by mypy."""
    order = OrderId.derive(intent_id=an_intent_id(), ts_event=NOW)
    with pytest.raises(ValueError, match="must start with"):
        IntentId(order.value)


@pytest.mark.parametrize("bad", ["int_", "int_tooshort", "int_" + "a" * 32, "nope"])
def test_malformed_ids_are_rejected(bad: str) -> None:
    with pytest.raises(ValueError):
        IntentId(bad)


# ── Order ids ────────────────────────────────────────────────


def test_orders_from_the_same_intent_differ_by_attempt() -> None:
    """Cancel-and-replace up the escalation ladder is a new order (§5.9)."""
    intent = an_intent_id()
    first = OrderId.derive(intent_id=intent, ts_event=NOW, attempt=0)
    second = OrderId.derive(intent_id=intent, ts_event=NOW, attempt=1)
    assert first != second


def test_orders_from_different_intents_differ() -> None:
    a = OrderId.derive(intent_id=an_intent_id(seq=0), ts_event=NOW)
    b = OrderId.derive(intent_id=an_intent_id(seq=1), ts_event=NOW)
    assert a != b


# ── Fill ids ─────────────────────────────────────────────────


def test_fill_id_is_idempotent_on_broker_execution_id() -> None:
    """Brokers re-send execution reports on reconnect; the same one is one fill."""
    order = OrderId.derive(intent_id=an_intent_id(), ts_event=NOW)
    exec_id = "0000e0d5.68a1b2c3.01.01"
    assert FillId.derive(order_id=order, broker_exec_id=exec_id) == FillId.derive(
        order_id=order, broker_exec_id=exec_id
    )


def test_different_executions_are_different_fills() -> None:
    """A partially filled order produces several executions."""
    order = OrderId.derive(intent_id=an_intent_id(), ts_event=NOW)
    assert FillId.derive(order_id=order, broker_exec_id="a.1") != FillId.derive(
        order_id=order, broker_exec_id="a.2"
    )


def test_fill_requires_a_broker_execution_id() -> None:
    order = OrderId.derive(intent_id=an_intent_id(), ts_event=NOW)
    with pytest.raises(ValueError, match="idempotent"):
        FillId.derive(order_id=order, broker_exec_id="")


# ── Shape ────────────────────────────────────────────────────


def test_ids_are_short_enough_to_read_in_a_log_line() -> None:
    assert len(str(an_intent_id())) == 20  # "int_" + 16 hex


def test_ids_are_hashable_and_immutable() -> None:
    intent = an_intent_id()
    assert {intent: "ok"}[an_intent_id()] == "ok"
    with pytest.raises(AttributeError):
        intent.value = "int_0000000000000000"  # type: ignore[misc]
