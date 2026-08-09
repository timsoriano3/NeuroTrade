"""Tests for the clock abstraction.

Includes an architectural test asserting that no module outside `clock.py`
reads the wall clock. That test is the actual enforcement of the determinism
invariant — the rest of this file only checks that the clocks themselves behave.
"""

from __future__ import annotations

import ast
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from neurotrade.core.clock import (
    Clock,
    LiveClock,
    SimClock,
    to_datetime,
    to_nanos,
)

# ── Conversion ───────────────────────────────────────────────


def test_epoch_round_trips() -> None:
    assert to_nanos(datetime(1970, 1, 1, tzinfo=UTC)) == 0
    assert to_datetime(0) == datetime(1970, 1, 1, tzinfo=UTC)


def test_to_nanos_rejects_naive_datetime() -> None:
    """A naive timestamp would silently shift every bar by the UTC offset."""
    with pytest.raises(ValueError, match="timezone-aware"):
        to_nanos(datetime(2026, 3, 14, 9, 30))  # noqa: DTZ001 - the point of the test


def test_to_datetime_is_always_utc_aware() -> None:
    assert to_datetime(1_700_000_000_000_000_000).tzinfo is UTC


def test_datetime_round_trip_at_microsecond_precision() -> None:
    moment = datetime(2026, 3, 14, 13, 30, 15, 123_456, tzinfo=UTC)
    assert to_datetime(to_nanos(moment)) == moment


def test_to_nanos_is_exact_across_a_session() -> None:
    """Guards against routing the conversion through a float.

    `datetime.timestamp() * 1e9` looks correct and is wrong: epoch nanoseconds
    need 19 significant digits, float64 offers ~16, and the result drifts by
    around +/-100 ns. Small enough to miss, large enough to reorder events.
    """
    base = datetime(2026, 3, 14, 13, 30, tzinfo=UTC)
    for microseconds in range(0, 23_400_000, 7_919):  # ~2,900 points in a session
        moment = base + timedelta(microseconds=microseconds)
        delta = moment - datetime(1970, 1, 1, tzinfo=UTC)
        expected = (delta.days * 86_400 + delta.seconds) * 10**9 + delta.microseconds * 1_000
        assert to_nanos(moment) == expected, f"drift at {moment}"


def test_to_datetime_truncates_rather_than_rounds() -> None:
    """Rounding up could push an event past a boundary it never reached."""
    base_ns = to_nanos(datetime(2026, 3, 14, 13, 30, tzinfo=UTC))
    assert to_datetime(base_ns + 999).microsecond == 0
    assert to_datetime(base_ns + 1_999).microsecond == 1
    assert to_datetime(base_ns + 123_456_789).microsecond == 123_456


def test_conversion_round_trips_exactly_at_microsecond_boundaries() -> None:
    base = datetime(2026, 3, 14, 13, 30, tzinfo=UTC)
    for microseconds in range(0, 1_000_000, 7_919):
        moment = base + timedelta(microseconds=microseconds)
        assert to_datetime(to_nanos(moment)) == moment


def test_pre_epoch_timestamps_convert_correctly() -> None:
    """Negative nanoseconds must floor, not truncate toward zero."""
    moment = datetime(1969, 7, 20, 20, 17, 40, tzinfo=UTC)
    assert to_datetime(to_nanos(moment)) == moment
    assert to_nanos(moment) < 0


def test_nanosecond_precision_survives_in_the_integer_domain() -> None:
    """Two events one nanosecond apart must remain distinguishable.

    They are not distinguishable as datetimes, which is why ns is canonical.
    """
    a, b = 1_700_000_000_000_000_001, 1_700_000_000_000_000_002
    assert a != b
    assert to_datetime(a) == to_datetime(b)  # datetime cannot tell them apart


# ── SimClock ─────────────────────────────────────────────────


def test_simclock_starts_where_told() -> None:
    assert SimClock(1_234).now_ns() == 1_234
    assert SimClock(datetime(2026, 1, 1, tzinfo=UTC)).now() == datetime(2026, 1, 1, tzinfo=UTC)


def test_simclock_does_not_move_on_its_own() -> None:
    """The property that makes replay deterministic."""
    clock = SimClock(1_000)
    readings = [clock.now_ns() for _ in range(1_000)]
    assert set(readings) == {1_000}


def test_simclock_advances_explicitly() -> None:
    clock = SimClock(1_000)
    clock.advance_ns(500)
    assert clock.now_ns() == 1_500
    clock.set_time_ns(2_000)
    assert clock.now_ns() == 2_000


def test_simclock_refuses_to_move_backwards() -> None:
    """Rewinding would let a component observe a future it should not know."""
    clock = SimClock(1_000)
    with pytest.raises(ValueError, match="backwards"):
        clock.set_time_ns(999)
    assert clock.now_ns() == 1_000


def test_simclock_refuses_negative_advance() -> None:
    clock = SimClock(1_000)
    with pytest.raises(ValueError, match="negative duration"):
        clock.advance_ns(-1)


def test_simclock_set_time_accepts_datetime() -> None:
    clock = SimClock(0)
    clock.set_time(datetime(2026, 3, 14, 13, 30, tzinfo=UTC))
    assert clock.now() == datetime(2026, 3, 14, 13, 30, tzinfo=UTC)


def test_simclock_advancing_to_the_same_instant_is_allowed() -> None:
    """Several events can share a timestamp; ordering comes from seq, not time."""
    clock = SimClock(1_000)
    clock.set_time_ns(1_000)
    assert clock.now_ns() == 1_000


# ── LiveClock ────────────────────────────────────────────────


def test_liveclock_is_utc_aware() -> None:
    assert LiveClock().now().tzinfo is UTC


def test_liveclock_moves_forward() -> None:
    clock = LiveClock()
    first = clock.now_ns()
    assert clock.now_ns() >= first


def test_liveclock_is_roughly_correct() -> None:
    """Guards against a unit error — ns vs µs would be off by 1000x."""
    assert abs(LiveClock().now() - datetime.now(UTC)) < timedelta(seconds=5)


# ── Protocol conformance ─────────────────────────────────────


@pytest.mark.parametrize("clock", [LiveClock(), SimClock(0)])
def test_both_clocks_satisfy_the_protocol(clock: Clock) -> None:
    assert isinstance(clock, Clock)


# ── Architectural guard ──────────────────────────────────────

_WALL_CLOCK_CALLS = {
    ("datetime", "now"),
    ("datetime", "utcnow"),
    ("datetime", "today"),
    ("time", "time"),
    ("time", "time_ns"),
    ("time", "monotonic"),
    ("time", "monotonic_ns"),
}


def _wall_clock_reads(tree: ast.AST) -> list[str]:
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        owner = node.func.value
        if isinstance(owner, ast.Name) and (owner.id, node.func.attr) in _WALL_CLOCK_CALLS:
            found.append(f"{owner.id}.{node.func.attr}() at line {node.lineno}")
    return found


def test_no_module_outside_clock_reads_the_wall_clock() -> None:
    """The determinism invariant, enforced rather than documented.

    ruff's DTZ rules ban *naive* datetimes; this bans reading the OS clock at
    all outside clock.py, which is the stricter property replay actually needs.
    """
    package = Path(__file__).resolve().parents[2] / "src" / "neurotrade"
    offenders: dict[str, list[str]] = {}

    for module in package.rglob("*.py"):
        if module.name == "clock.py":
            continue
        reads = _wall_clock_reads(ast.parse(module.read_text()))
        if reads:
            offenders[str(module.relative_to(package))] = reads

    assert not offenders, (
        f"wall-clock reads outside clock.py: {offenders}. "
        f"Take time from a Clock instead — see CLAUDE.md."
    )


def test_the_architectural_guard_actually_detects_violations() -> None:
    """A guard that has never fired is not known to work."""
    violating = ast.parse("import time\ndef f():\n    return time.time_ns()\n")
    assert _wall_clock_reads(violating) == ["time.time_ns() at line 3"]
