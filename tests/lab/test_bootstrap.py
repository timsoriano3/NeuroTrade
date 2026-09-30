"""Tests for the session-block bootstrap.

The module exists to answer one question — how many *independent* bets a
dependent sample is worth — so the tests that matter are the ones that would
catch it answering in the wrong direction. Making observations more dependent
must lower the effective count; it must never raise it above the number of
observations actually held, and it must never be pinned to the session count by
construction, because pinning it there is the assumption the bootstrap replaces.
"""

from __future__ import annotations

import random

import pytest

from neurotrade.lab.bootstrap import (
    DEFAULT_BOOTSTRAP_SEED,
    SharpeInterval,
    bootstrap_sharpe,
    group_by_cluster,
)

# ── fixtures ────────────────────────────────────────────────────────────────


def independent(
    n_sessions: int, per_session: int, *, seed: int, drift: float = 0.0
) -> tuple[list[float], list[int]]:
    """Returns with no within-session structure: every observation its own draw."""
    rng = random.Random(seed)
    returns = [rng.gauss(drift, 0.01) for _ in range(n_sessions * per_session)]
    sessions = [index // per_session for index in range(len(returns))]
    return returns, sessions


def clustered(
    n_sessions: int, per_session: int, *, seed: int, drift: float = 0.0
) -> tuple[list[float], list[int]]:
    """Returns that are identical inside a session: total within-session dependence.

    The extreme case. The sample holds `n_sessions * per_session` observations
    and exactly `n_sessions` independent draws, so an honest effective count has
    to come back near the session count rather than near the observation count.
    """
    rng = random.Random(seed)
    returns: list[float] = []
    sessions: list[int] = []
    for session in range(n_sessions):
        value = rng.gauss(drift, 0.01)
        returns.extend([value] * per_session)
        sessions.extend([session] * per_session)
    return returns, sessions


# ── group_by_cluster ────────────────────────────────────────────────────────


def test_grouping_keeps_first_appearance_order_not_sorted_order() -> None:
    """Cluster keys need not be orderable, so the order comes from the data."""
    blocks = group_by_cluster([1.0, 2.0, 3.0, 4.0], ["tue", "mon", "tue", "mon"])
    assert blocks == ((1.0, 3.0), (2.0, 4.0))


def test_grouping_refuses_mismatched_lengths() -> None:
    """A silent zip would drop the tail, which is the most recent sessions."""
    with pytest.raises(ValueError, match=r"3 returns against 2 cluster keys"):
        group_by_cluster([1.0, 2.0, 3.0], ["mon", "tue"])


# ── the interval itself ─────────────────────────────────────────────────────


def test_a_sample_with_one_session_carries_no_interval() -> None:
    """Nothing to resample: every draw is the same series."""
    assert bootstrap_sharpe([0.01, -0.01, 0.02], ["mon"] * 3) is None


def test_a_flat_sample_carries_no_interval() -> None:
    """`moments` raises on zero variance; the values are tested, not the result."""
    assert bootstrap_sharpe([0.01] * 20, [index // 5 for index in range(20)]) is None


def test_the_interval_is_reproducible_at_one_seed() -> None:
    """A confidence interval is subject to the determinism invariant like a digest."""
    returns, sessions = independent(40, 6, seed=11, drift=0.002)
    first = bootstrap_sharpe(returns, sessions, n_resamples=300)
    again = bootstrap_sharpe(returns, sessions, n_resamples=300)
    assert first == again


def test_a_different_seed_moves_the_bounds_but_not_the_point_estimate() -> None:
    """The observed Sharpe is the data's; only the bounds are resampled."""
    returns, sessions = independent(40, 6, seed=12, drift=0.002)
    first = bootstrap_sharpe(returns, sessions, n_resamples=300)
    other = bootstrap_sharpe(returns, sessions, n_resamples=300, seed=DEFAULT_BOOTSTRAP_SEED + 1)
    assert first is not None and other is not None
    assert first.observed == other.observed
    assert (first.lower, first.upper) != (other.lower, other.upper)


# ── the finding the module was built for ────────────────────────────────────


def test_within_session_dependence_collapses_the_effective_count() -> None:
    """Identical observations inside a session are worth one bet, not `per_session`.

    The whole reason the module exists. `deflated` charges the raw count and
    `deflated_clustered` charges the session count; this asserts the measurement
    lands near the session count when the dependence really is total, rather than
    being pinned there by a floor.
    """
    returns, sessions = clustered(60, 8, seed=5, drift=0.003)
    bounds = bootstrap_sharpe(returns, sessions, n_resamples=400)
    assert bounds is not None
    assert bounds.n_observations == 480
    assert bounds.n_clusters == 60
    # Within a factor of two of the session count, and nowhere near the 480 the
    # raw count claims. The bound is loose on purpose: the point is the order of
    # magnitude, and pinning a tolerance tighter than the estimator's own noise
    # would make this test a fixture of its seed.
    assert 30 <= bounds.effective_n <= 120


def test_independent_observations_are_worth_far_more_than_their_sessions() -> None:
    """The other direction: no within-session structure, so the raw count is close to right.

    Without this the module could satisfy the test above by always returning the
    session count, which is exactly the bound it was built to improve on.
    """
    returns, sessions = independent(60, 8, seed=5, drift=0.003)
    bounds = bootstrap_sharpe(returns, sessions, n_resamples=400)
    assert bounds is not None
    assert bounds.effective_n > 4 * bounds.n_clusters


def test_the_effective_count_never_exceeds_the_observations_held() -> None:
    """A cap, not a guess: a sample cannot be worth more bets than it has rows."""
    returns, sessions = independent(4, 4, seed=7, drift=0.05)
    bounds = bootstrap_sharpe(returns, sessions, n_resamples=200)
    assert bounds is not None
    assert bounds.effective_n <= bounds.n_observations


def test_inflation_is_the_ratio_the_psr_z_score_is_overstated_by() -> None:
    """`sqrt(inflation)` is what the reported clustering correction shrinks z by."""
    returns, sessions = clustered(50, 7, seed=9, drift=0.004)
    bounds = bootstrap_sharpe(returns, sessions, n_resamples=300)
    assert bounds is not None
    assert bounds.inflation == pytest.approx(bounds.n_observations / bounds.effective_n)
    assert bounds.inflation > 1.0


# ── bounds and validation ───────────────────────────────────────────────────


def test_a_strong_edge_produces_an_interval_that_excludes_zero() -> None:
    returns, sessions = independent(80, 5, seed=3, drift=0.02)
    bounds = bootstrap_sharpe(returns, sessions, n_resamples=400)
    assert bounds is not None
    assert bounds.excludes_zero


def test_a_marginal_edge_produces_an_interval_that_does_not() -> None:
    returns, sessions = independent(80, 5, seed=3, drift=0.0002)
    bounds = bootstrap_sharpe(returns, sessions, n_resamples=400)
    assert bounds is not None
    assert not bounds.excludes_zero


def test_a_wider_confidence_gives_wider_bounds() -> None:
    returns, sessions = independent(50, 5, seed=4, drift=0.003)
    narrow = bootstrap_sharpe(returns, sessions, n_resamples=400, confidence=0.80)
    wide = bootstrap_sharpe(returns, sessions, n_resamples=400, confidence=0.99)
    assert narrow is not None and wide is not None
    assert wide.lower < narrow.lower
    assert wide.upper > narrow.upper


def test_a_longer_block_narrows_the_interval_on_independent_sessions() -> None:
    """A longer block is NOT automatically more conservative — measured, not assumed.

    The intuition is that longer blocks preserve more dependence and therefore
    widen the interval. On sessions that are genuinely independent the opposite
    happens: each draw reproduces more of the original ordering, so the
    resampled Sharpe ratios cluster tighter around the observed one and the
    standard error *falls*. That makes `mean_block` a claim about the data rather
    than a safety dial, which is why `DEFAULT_MEAN_BLOCK` is one and raising it
    is opt-in. This test pins the falsifying direction so the docstring cannot
    drift back to the intuition.
    """
    returns, sessions = independent(60, 5, seed=6, drift=0.003)
    single = bootstrap_sharpe(returns, sessions, n_resamples=400, mean_block=1.0)
    long_blocks = bootstrap_sharpe(returns, sessions, n_resamples=400, mean_block=6.0)
    assert single is not None and long_blocks is not None
    assert long_blocks.mean_block == 6.0
    assert long_blocks.standard_error < single.standard_error


@pytest.mark.parametrize(
    ("n_resamples", "mean_block", "confidence", "message"),
    [
        (0, 1.0, 0.95, "n_resamples 0 must be at least 1"),
        (100, 0.5, 0.95, "mean_block 0.5 must be at least one cluster"),
        (100, 1.0, 1.0, r"confidence 1\.0 must be strictly inside \(0, 1\)"),
        (100, 1.0, 0.0, r"confidence 0\.0 must be strictly inside \(0, 1\)"),
    ],
)
def test_the_bootstrap_refuses_impossible_settings(
    n_resamples: int, mean_block: float, confidence: float, message: str
) -> None:
    returns, sessions = independent(10, 4, seed=1)
    with pytest.raises(ValueError, match=message):
        bootstrap_sharpe(
            returns,
            sessions,
            n_resamples=n_resamples,
            mean_block=mean_block,
            confidence=confidence,
        )


def test_a_zero_spread_falls_back_to_the_session_count() -> None:
    """Rather than dividing by zero. Only reachable on a degenerate sample."""
    interval = SharpeInterval(
        observed=0.01,
        lower=0.01,
        upper=0.01,
        standard_error=0.0,
        n_observations=100,
        n_clusters=20,
        n_resamples=10,
        mean_block=1.0,
        confidence=0.95,
    )
    assert interval.effective_n == 20.0
