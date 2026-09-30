"""Volatility-regime coverage: the difference between sampling a regime and earning in it.

The defect this file is heaviest on produces a *better* number rather than an
error: counting buckets a strategy merely traded in, rather than buckets it made
money in, turns one contiguous macro episode into "full regime coverage". §15 asks
for positive expectancy in >=2 regimes, and `coverage` is the only property that
answers it.
"""

from __future__ import annotations

from datetime import date

import pytest

from neurotrade.lab.regimes import (
    DEFAULT_MIN_TRADES,
    BucketOutcome,
    RegimeCoverage,
    VolatilityBucket,
    bucket_sessions,
    coverage_of,
    trailing_volatility,
)

LOW = VolatilityBucket.LOW
MID = VolatilityBucket.MID
HIGH = VolatilityBucket.HIGH


def days(*values: float) -> dict[date, float]:
    """Consecutive January 2024 sessions carrying the values given."""
    return {date(2024, 1, index + 1): value for index, value in enumerate(values)}


# ── Trailing volatility ─────────────────────────────────────────────────


def test_a_reading_uses_only_sessions_before_the_one_it_labels() -> None:
    """Including the current session would let a regime label see its own day."""
    readings = trailing_volatility(days(1.0, 2.0, 99.0, 4.0), memory=2)
    # The 3rd session is labelled 1.5 — the mean of sessions 1 and 2 — so the
    # spike on that day cannot influence its own label.
    assert readings[date(2024, 1, 3)] == 1.5
    # The 4th sees the spike, because by then it is history.
    assert readings[date(2024, 1, 4)] == pytest.approx(50.5)


def test_the_first_sessions_have_no_reading_rather_than_a_short_one() -> None:
    """A band from three days is a different statistic, not a rough version."""
    readings = trailing_volatility(days(1.0, 2.0, 3.0, 4.0, 5.0), memory=3)
    assert set(readings) == {date(2024, 1, 4), date(2024, 1, 5)}


def test_readings_do_not_depend_on_the_order_the_sessions_arrived() -> None:
    """Determinism: the mapping is sorted before anything is averaged."""
    forward = trailing_volatility(days(1.0, 2.0, 3.0, 4.0), memory=2)
    shuffled = dict(reversed(list(days(1.0, 2.0, 3.0, 4.0).items())))
    assert trailing_volatility(shuffled, memory=2) == forward


@pytest.mark.parametrize("memory", [0, -1])
def test_a_memory_below_one_session_is_refused(memory: int) -> None:
    """Averaging zero sessions is not a volatility reading."""
    with pytest.raises(ValueError, match=f"memory {memory} must be at least 1"):
        trailing_volatility(days(1.0, 2.0), memory=memory)


def test_a_memory_longer_than_the_sample_yields_nothing() -> None:
    """No session has enough history, and that is an empty result not an error."""
    assert trailing_volatility(days(1.0, 2.0), memory=5) == {}


# ── Bucketing ───────────────────────────────────────────────────────────


def test_sessions_split_into_three_populated_thirds() -> None:
    """The point of terciles: every bucket gets sessions."""
    buckets = bucket_sessions(days(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
    counts = {
        bucket: sum(1 for got in buckets.values() if got is bucket) for bucket in (LOW, MID, HIGH)
    }
    assert counts == {LOW: 2, MID: 2, HIGH: 2}


def test_one_reading_is_all_one_bucket_not_an_error() -> None:
    """A single session cannot be split, and refusing would be worse than saying so."""
    assert set(bucket_sessions(days(1.0)).values()) == {HIGH}


def test_identical_readings_do_not_fabricate_a_spread() -> None:
    """Every session the same volatility is one regime, however many sessions."""
    assert len(set(bucket_sessions(days(*[2.0] * 9)).values())) == 1


def test_no_sessions_at_all_is_refused() -> None:
    """There is nothing to split, and an empty mapping would read as no coverage."""
    with pytest.raises(ValueError, match="at least one session"):
        bucket_sessions({})


# ── Coverage: sampled is not covered ────────────────────────────────────


def test_a_losing_bucket_is_sampled_but_not_covered() -> None:
    """The distinction the whole module exists for."""
    got = coverage_of([LOW] * 40 + [HIGH] * 40, [0.01] * 40 + [-0.01] * 40, min_trades=30)
    assert got.sampled == 2
    assert got.coverage == 1
    assert not got.meets_spec


def test_a_thin_bucket_counts_neither_way() -> None:
    """Eight profitable trades in one regime is not a regime that was covered."""
    got = coverage_of([LOW] * 40 + [HIGH] * 8, [0.01] * 48, min_trades=30)
    assert (got.sampled, got.coverage) == (1, 1)
    assert [outcome.bucket for outcome in got.outcomes] == [LOW, HIGH]


def test_two_profitable_buckets_meet_the_spec() -> None:
    """§15's wording, and the only way `meets_spec` becomes True."""
    got = coverage_of([LOW] * 40 + [MID] * 40, [0.01] * 80, min_trades=30)
    assert (got.coverage, got.meets_spec) == (2, True)


def test_observations_with_no_reading_are_dropped_not_reassigned() -> None:
    """Folding unclassifiable sessions into MID invents a regime result."""
    got = coverage_of([None] * 50 + [LOW] * 30, [0.01] * 80, min_trades=30)
    assert [outcome.bucket for outcome in got.outcomes] == [LOW]
    assert got.outcomes[0].n_trades == 30


def test_observations_the_variant_sat_out_are_not_trades() -> None:
    """A None return is no view, not a flat trade that would dilute expectancy."""
    got = coverage_of([LOW] * 4, [0.02, None, 0.04, None], min_trades=1)
    assert got.outcomes[0].n_trades == 2
    assert got.outcomes[0].expectancy == pytest.approx(0.03)


def test_buckets_come_back_in_volatility_order() -> None:
    """Reading order is low to high, whatever order the observations arrived in."""
    got = coverage_of([HIGH, LOW, MID], [0.01, 0.01, 0.01], min_trades=1)
    assert [outcome.bucket for outcome in got.outcomes] == [LOW, MID, HIGH]


def test_misaligned_buckets_and_returns_are_refused() -> None:
    """A silent zip would score returns against the wrong sessions."""
    with pytest.raises(ValueError, match="buckets 3 and returns 2 must align"):
        coverage_of([LOW, LOW, LOW], [0.01, 0.01])


@pytest.mark.parametrize("floor", [0, -1])
def test_a_trade_floor_below_one_is_refused(floor: int) -> None:
    """Zero would make every bucket a result, including empty ones."""
    with pytest.raises(ValueError, match=f"min_trades {floor} must be at least 1"):
        coverage_of([LOW], [0.01], min_trades=floor)


def test_no_trades_anywhere_is_zero_coverage_not_a_crash() -> None:
    """A variant that never traded has no regime evidence; it is not an error here."""
    got = coverage_of([LOW, MID], [None, None])
    assert got.outcomes == ()
    assert (got.coverage, got.sampled, got.meets_spec) == (0, 0, False)


# ── Reporting ───────────────────────────────────────────────────────────


def test_the_printed_form_shows_every_bucket_and_the_floor() -> None:
    """Coverage alone hides which regime failed; the reader needs both."""
    printed = str(
        RegimeCoverage(
            outcomes=(
                BucketOutcome(LOW, 50, 0.0002),
                BucketOutcome(HIGH, 8, -0.0009),
            ),
            min_trades=30,
        )
    )
    assert "regimes 1/1 covered (min 30t)" in printed
    assert "low=+0.00020/50t" in printed
    assert "high=-0.00090/8t" in printed


def test_the_default_floor_is_half_the_money_gate() -> None:
    """§13.1 wants 60+ labelled trades; a per-regime slice gets half of that."""
    assert DEFAULT_MIN_TRADES == 30
