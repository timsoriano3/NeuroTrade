"""Tests for the deflated Sharpe ratio and PBO.

Both quantities exist to make a good-looking backtest look worse, so the tests
that matter are the monotonicity ones: more trials must lower the DSR, a wider
search must lower it further, and a shorter sample must lower the PSR. A bug
that got any of those signs backwards would be invisible in a single number and
would approve exactly the strategies this module exists to reject.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import pytest

from neurotrade.lab.significance import (
    OverfittingReport,
    annualized,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    moments,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
    sharpe_ratio,
)


def alternating(count: int, *, high: float, low: float) -> list[float]:
    """A deterministic return series with non-zero variance and no RNG."""
    return [high if index % 2 else low for index in range(count)]


def sawtooth(count: int, *, mean: float, amplitude: float) -> list[float]:
    """Deterministic returns cycling through four levels around `mean`."""
    offsets = (amplitude, -amplitude, amplitude / 2, -amplitude / 2)
    return [mean + offsets[index % 4] for index in range(count)]


# ── moments ──────────────────────────────────────────────────


def test_a_normal_looking_series_has_kurtosis_near_three() -> None:
    # Not a normality test — a check that the fourth moment is reported
    # non-excess, since the PSR formula subtracts one from it.
    values = [-2.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0]
    assert 2.0 < moments(values).kurtosis < 3.0


def test_skew_signs_follow_the_tail() -> None:
    left = [-5.0, 1.0, 1.0, 1.0, 1.0]
    right = [5.0, -1.0, -1.0, -1.0, -1.0]
    assert moments(left).skew < 0 < moments(right).skew


def test_stdev_is_the_sample_estimate() -> None:
    assert moments([1.0, 3.0]).stdev == pytest.approx(math.sqrt(2.0))


@pytest.mark.parametrize("values", [[], [0.01]])
def test_too_few_returns_is_rejected(values: Sequence[float]) -> None:
    with pytest.raises(ValueError, match="need at least 2 returns"):
        moments(values)


def test_a_constant_series_has_no_sharpe_ratio() -> None:
    # Returning inf here would put an unbeatable strategy on the dashboard.
    with pytest.raises(ValueError, match="zero variance"):
        sharpe_ratio([0.01] * 50)


# ── sharpe_ratio and annualized ──────────────────────────────


def test_sharpe_is_mean_over_stdev() -> None:
    values = sawtooth(40, mean=0.001, amplitude=0.01)
    stats = moments(values)
    assert sharpe_ratio(values) == pytest.approx(stats.mean / stats.stdev)


def test_risk_free_rate_is_subtracted_from_the_mean() -> None:
    values = sawtooth(40, mean=0.002, amplitude=0.01)
    assert sharpe_ratio(values, risk_free=0.002) == pytest.approx(0.0, abs=1e-12)


def test_annualization_scales_by_the_square_root_of_frequency() -> None:
    assert annualized(0.05, periods_per_year=256) == pytest.approx(0.8)


@pytest.mark.parametrize("periods", [0, -1, -252.0])
def test_annualization_rejects_a_non_positive_frequency(periods: float) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        annualized(0.05, periods_per_year=periods)


# ── probabilistic Sharpe ratio ───────────────────────────────


def test_a_longer_sample_raises_confidence() -> None:
    short = probabilistic_sharpe_ratio(0.08, n_observations=30)
    long = probabilistic_sharpe_ratio(0.08, n_observations=1000)
    assert short < long


def test_negative_skew_is_charged_for() -> None:
    # Same Sharpe, same sample: the series that wins small and loses big is
    # less believable, and the PSR has to say so.
    symmetric = probabilistic_sharpe_ratio(0.1, n_observations=250, skew=0.0)
    left_tailed = probabilistic_sharpe_ratio(0.1, n_observations=250, skew=-1.5)
    assert left_tailed < symmetric


def test_fat_tails_are_charged_for() -> None:
    normal = probabilistic_sharpe_ratio(0.1, n_observations=250, kurtosis=3.0)
    fat = probabilistic_sharpe_ratio(0.1, n_observations=250, kurtosis=12.0)
    assert fat < normal


def test_a_sharpe_equal_to_the_benchmark_is_a_coin_flip() -> None:
    assert probabilistic_sharpe_ratio(0.1, benchmark=0.1, n_observations=500) == pytest.approx(0.5)


def test_a_sharpe_below_the_benchmark_falls_under_a_half() -> None:
    assert probabilistic_sharpe_ratio(0.05, benchmark=0.1, n_observations=500) < 0.5


@pytest.mark.parametrize("n_observations", [0, 1, -10])
def test_psr_rejects_a_sample_too_short_to_estimate(n_observations: int) -> None:
    with pytest.raises(ValueError, match="must be at least 2"):
        probabilistic_sharpe_ratio(0.1, n_observations=n_observations)


def test_psr_rejects_moments_no_return_series_could_have() -> None:
    # Large positive skew with a large Sharpe drives the variance term
    # negative; silently taking the square root of it would return nan.
    with pytest.raises(ValueError, match="degenerate Sharpe variance"):
        probabilistic_sharpe_ratio(2.0, n_observations=250, skew=3.0, kurtosis=1.0)


# ── expected maximum Sharpe ──────────────────────────────────


def test_one_trial_is_no_search() -> None:
    assert expected_max_sharpe(n_trials=1, trial_variance=1.0) == 0.0


def test_identical_trials_produce_no_hurdle() -> None:
    assert expected_max_sharpe(n_trials=500, trial_variance=0.0) == 0.0


def test_the_hurdle_rises_with_the_number_of_trials() -> None:
    hurdles = [
        expected_max_sharpe(n_trials=n, trial_variance=0.04) for n in (10, 100, 1_000, 10_000)
    ]
    assert hurdles == sorted(hurdles)
    assert hurdles[0] < hurdles[-1]


def test_the_hurdle_rises_with_the_spread_of_the_search() -> None:
    narrow = expected_max_sharpe(n_trials=200, trial_variance=0.001)
    wide = expected_max_sharpe(n_trials=200, trial_variance=0.1)
    assert narrow < wide


@pytest.mark.parametrize(
    ("n_trials", "variance", "message"),
    [(0, 0.04, "must be at least 1"), (-3, 0.04, "must be at least 1"), (10, -0.1, "negative")],
)
def test_expected_max_rejects_impossible_inputs(
    n_trials: int, variance: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        expected_max_sharpe(n_trials=n_trials, trial_variance=variance)


# ── deflated Sharpe ratio ────────────────────────────────────


def test_deflation_never_exceeds_the_undeflated_confidence() -> None:
    trials = [0.002 * index - 0.1 for index in range(100)]
    undeflated = probabilistic_sharpe_ratio(0.12, n_observations=500)
    assert deflated_sharpe_ratio(0.12, trial_sharpes=trials, n_observations=500) <= undeflated


def test_more_trials_lower_the_deflated_sharpe() -> None:
    few = [0.002 * index - 0.02 for index in range(20)]
    many = [0.002 * index - 0.02 for index in range(20)] + [
        0.0001 * index for index in range(2_000)
    ]
    assert deflated_sharpe_ratio(
        0.1, trial_sharpes=many, n_observations=500
    ) < deflated_sharpe_ratio(0.1, trial_sharpes=few, n_observations=500)


def test_a_single_trial_is_not_deflated_at_all() -> None:
    # One look is no search: DSR and PSR must agree exactly.
    assert deflated_sharpe_ratio(0.1, trial_sharpes=[0.1], n_observations=400) == pytest.approx(
        probabilistic_sharpe_ratio(0.1, n_observations=400)
    )


def test_deflation_needs_a_trial_history() -> None:
    with pytest.raises(ValueError, match="at least one trial"):
        deflated_sharpe_ratio(0.1, trial_sharpes=[], n_observations=400)


# ── probability of backtest overfitting ──────────────────────


def test_a_consistently_better_strategy_is_not_overfit() -> None:
    rows = [[0.02, 0.01] if index % 2 else [-0.01, -0.02] for index in range(80)]
    report = probability_of_backtest_overfitting(rows, n_blocks=6)
    assert report.pbo == 0.0
    assert not report.is_overfit


def test_a_strategy_that_only_wins_in_the_first_half_is_overfit() -> None:
    # Strategy 0 wins every early block and loses every late one; strategy 1
    # is the mirror. Whichever wins in sample loses out of sample, which is
    # exactly what PBO is built to detect.
    rows: list[list[float]] = []
    for index in range(80):
        early = index < 40
        rows.append([0.02, -0.02] if early else [-0.02, 0.02])
        rows.append([-0.01, 0.01] if early else [0.01, -0.01])
    report = probability_of_backtest_overfitting(rows, n_blocks=4)
    assert report.is_overfit


def test_the_report_carries_one_logit_per_combination() -> None:
    rows = [[0.02, 0.01] if index % 2 else [-0.01, -0.02] for index in range(60)]
    report = probability_of_backtest_overfitting(rows, n_blocks=6)
    assert report.n_combinations == math.comb(6, 3) == len(report.logits)


def test_pbo_is_the_share_of_non_positive_logits() -> None:
    rows = [[0.02, 0.01] if index % 2 else [-0.01, -0.02] for index in range(60)]
    report = probability_of_backtest_overfitting(rows, n_blocks=6)
    misses = sum(1 for value in report.logits if value <= 0.0)
    assert report.pbo == pytest.approx(misses / report.n_combinations)


def test_the_overfit_threshold_is_a_coin_flip() -> None:
    assert OverfittingReport(pbo=0.5, n_combinations=1, logits=()).is_overfit
    assert not OverfittingReport(pbo=0.49, n_combinations=1, logits=()).is_overfit


def test_a_column_that_never_traded_does_not_crash_the_ranking() -> None:
    # A strategy flat over a subsample has no Sharpe. Ranking it zero keeps
    # the split informative about the others rather than discarding it.
    rows = [[0.02, 0.0] if index % 2 else [-0.01, 0.0] for index in range(60)]
    assert probability_of_backtest_overfitting(rows, n_blocks=4).n_combinations == math.comb(4, 2)


@pytest.mark.parametrize("n_blocks", [3, 5, 2, 0, -4])
def test_pbo_rejects_block_counts_it_cannot_halve(n_blocks: int) -> None:
    rows = [[0.01, 0.02] for _ in range(40)]
    with pytest.raises(ValueError, match="must be even and at least 4"):
        probability_of_backtest_overfitting(rows, n_blocks=n_blocks)


def test_pbo_rejects_a_sample_shorter_than_its_blocks() -> None:
    rows = [[0.01, 0.02] for _ in range(5)]
    with pytest.raises(ValueError, match="cannot fill"):
        probability_of_backtest_overfitting(rows, n_blocks=6)


def test_pbo_needs_something_to_rank_against() -> None:
    rows = [[0.01] for _ in range(40)]
    with pytest.raises(ValueError, match="at least 2 strategies"):
        probability_of_backtest_overfitting(rows, n_blocks=4)


def test_pbo_rejects_ragged_input() -> None:
    rows: list[list[float]] = [[0.01, 0.02] for _ in range(39)]
    rows.append([0.01])
    with pytest.raises(ValueError, match="one entry per strategy"):
        probability_of_backtest_overfitting(rows, n_blocks=4)


def test_pbo_is_deterministic_across_repeated_calls() -> None:
    # No RNG, and ties broken by column index: two runs must agree exactly.
    rows = [alternating(2, high=0.02, low=-0.01) for _ in range(60)]
    first = probability_of_backtest_overfitting(rows, n_blocks=6)
    second = probability_of_backtest_overfitting(rows, n_blocks=6)
    assert first == second
