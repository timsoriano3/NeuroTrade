"""Is this result real? — the deflated Sharpe ratio and PBO (§8).

A backtest produces a number. The number is not evidence, for a reason that has
nothing to do with the code being wrong: **we looked many times**. Try enough
parameter sets, enough symbols, enough entry rules, and the best of them will
show a fine Sharpe ratio drawn from pure noise. §17 names this as the project's
primary risk — not a crash, a clean equity curve that does not survive live.

This module holds the two answers §8 requires.

**Deflated Sharpe Ratio.** Ask what the *best of N tries* would have scored if
every try were worthless, then ask how confident we are that the observed
Sharpe beats that. The first step is `expected_max_sharpe`; the second is the
probabilistic Sharpe ratio, which also charges the strategy for short samples,
negative skew and fat tails — the three ways a Sharpe ratio lies. `N` comes
from `lab/trials.py`, which counts every hypothesis anyone tested, including
the ones that were abandoned and the ones a discovery process generated
automatically. Deflating against the trials we chose to remember is the same
error one level up.

**Probability of Backtest Overfitting.** Different question: not "is this one
result significant" but "does in-sample rank predict out-of-sample rank at
all?" CSCV splits the sample into blocks, tries every way of halving them into
in-sample and out-of-sample, and each time asks where the in-sample winner
lands out of sample. If the winner lands in the bottom half as often as not,
selection is doing nothing and PBO approaches 0.5 — the backtest is a lottery
and the configuration chosen from it is a lottery ticket.

**Both are per-period, not annualized.** Every function here takes and returns
Sharpe ratios at the observation frequency, because the sample-size term in the
PSR is in the same units. Annualize for reporting with `annualized`, never
before passing a value into `deflated_sharpe_ratio` — a Sharpe multiplied by
`sqrt(252)` and a sample size of 252 describe two different experiments.

No RNG anywhere in this module: same inputs, same numbers, on every machine.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from statistics import NormalDist, StatisticsError, correlation
from typing import Final

__all__ = [
    "HaircutReport",
    "Moments",
    "OverfittingReport",
    "annualized",
    "deflated_sharpe_ratio",
    "effective_n_trials",
    "expected_max_sharpe",
    "haircut_sharpe",
    "minimum_backtest_length",
    "moments",
    "probabilistic_sharpe_ratio",
    "probability_of_backtest_overfitting",
    "sharpe_ratio",
]

_NORMAL: Final = NormalDist()

_EULER_MASCHERONI: Final = 0.5772156649015329
"""Appears in the expected maximum of N normal draws (Bailey & López de Prado).

Not a tuning constant — it is the constant in the asymptotic expansion of the
maximum order statistic, and changing it would mean computing a different
quantity.
"""


@dataclass(frozen=True, slots=True)
class Moments:
    """The shape of a return series, beyond its mean and spread.

    Skew and kurtosis are here because the Sharpe ratio ignores them and they
    are exactly how it misleads. A strategy that wins pennies daily and loses
    a year occasionally — selling options, mean reversion without a stop — has
    a superb Sharpe and negative skew, and the PSR charges it for that.

    Example:
        >>> m = moments([0.01, -0.005, 0.02, 0.0, 0.015])
        >>> round(m.mean, 4)
        0.008
    """

    n: int  # observations the moments were computed from
    mean: float  # arithmetic mean return per observation
    stdev: float  # sample standard deviation, ddof=1
    skew: float  # third standardized moment; 0 for a normal distribution
    kurtosis: float  # fourth standardized moment, NOT excess; 3 for a normal distribution


def moments(returns: Sequence[float]) -> Moments:
    """Summarize a return series.

    Args:
        returns: Per-period returns, in order. Floats, not `Decimal` — these
            are derived statistics, not money (see the precision invariant in
            `CLAUDE.md`). Convert realized returns at the boundary.

    Returns:
        The count, mean, sample standard deviation and the third and fourth
        standardized moments.

    Raises:
        ValueError: If fewer than two returns are given, or if every return is
            identical. A zero-variance series has no Sharpe ratio, and silently
            returning `inf` would put an unbeatable strategy on the dashboard.

    Example:
        >>> moments([0.01, -0.01, 0.01, -0.01]).kurtosis
        1.0
    """
    n = len(returns)
    if n < 2:
        raise ValueError(f"need at least 2 returns to estimate moments, got {n}")
    mean = math.fsum(returns) / n
    deviations = [value - mean for value in returns]
    # Sample variance (ddof=1) for the Sharpe ratio, population moments for the
    # shape terms: that is the pairing the PSR was derived with.
    variance = math.fsum(d * d for d in deviations) / (n - 1)
    if variance <= 0.0:
        raise ValueError("returns have zero variance; a Sharpe ratio is undefined")
    stdev = math.sqrt(variance)
    m2 = math.fsum(d * d for d in deviations) / n
    m3 = math.fsum(d**3 for d in deviations) / n
    m4 = math.fsum(d**4 for d in deviations) / n
    return Moments(
        n=n,
        mean=mean,
        stdev=stdev,
        skew=m3 / m2**1.5,
        kurtosis=m4 / m2**2,
    )


def sharpe_ratio(returns: Sequence[float], *, risk_free: float = 0.0) -> float:
    """Mean excess return over its standard deviation, per observation.

    Args:
        returns: Per-period returns, in order.
        risk_free: Per-period risk-free rate, in the same units as `returns`.
            Defaults to zero, which is the right choice intraday: capital is
            not held overnight, so there is no financing leg to net out.

    Returns:
        The per-period Sharpe ratio. Not annualized — see `annualized`.

    Raises:
        ValueError: If fewer than two returns are given or the series has zero
            variance.

    Example:
        >>> round(sharpe_ratio([0.01, -0.005, 0.02, 0.0, 0.015]), 4)
        0.7716
    """
    stats = moments(returns)
    return (stats.mean - risk_free) / stats.stdev


def annualized(sharpe: float, *, periods_per_year: float) -> float:
    """Scale a per-period Sharpe ratio to annual units, for reporting only.

    Args:
        sharpe: Per-period Sharpe ratio.
        periods_per_year: Observations in a year — 252 for daily bars, 252*390
            for US regular-hours minute bars.

    Returns:
        `sharpe * sqrt(periods_per_year)`.

    Raises:
        ValueError: If `periods_per_year` is not positive.

    Example:
        >>> round(annualized(0.05, periods_per_year=252), 4)
        0.7937
    """
    if periods_per_year <= 0:
        raise ValueError(f"periods_per_year {periods_per_year} must be positive")
    return sharpe * math.sqrt(periods_per_year)


def probabilistic_sharpe_ratio(
    observed: float,
    *,
    benchmark: float = 0.0,
    n_observations: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Confidence that the true Sharpe ratio exceeds `benchmark`.

    The estimation error on a Sharpe ratio is not `1/sqrt(n)`. It widens with
    negative skew and with fat tails, which is why a strategy can post a high
    Sharpe over a short sample and mean nothing: the denominator below is what
    charges it for that.

    Args:
        observed: Per-period Sharpe ratio measured on the sample.
        benchmark: Per-period Sharpe ratio to beat. Zero asks only "is this
            better than nothing"; `deflated_sharpe_ratio` passes the expected
            maximum across trials instead, which is the question worth asking.
        n_observations: Returns the Sharpe was computed from.
        skew: Third standardized moment of the returns.
        kurtosis: Fourth standardized moment, not excess. 3 is normal.

    Returns:
        A probability in `(0, 1)`.

    Raises:
        ValueError: If `n_observations` is below 2, or if the variance term is
            not positive — which means the supplied moments are not consistent
            with any real return series.

    Example:
        >>> round(probabilistic_sharpe_ratio(0.1, n_observations=250), 4)
        0.9423
        >>> round(probabilistic_sharpe_ratio(0.1, n_observations=20), 4)
        0.6681
    """
    if n_observations < 2:
        raise ValueError(f"n_observations {n_observations} must be at least 2")
    variance = 1.0 - skew * observed + (kurtosis - 1.0) / 4.0 * observed**2
    if variance <= 0.0:
        raise ValueError(f"degenerate Sharpe variance {variance}; check skew and kurtosis")
    statistic = (observed - benchmark) * math.sqrt(n_observations - 1) / math.sqrt(variance)
    return _NORMAL.cdf(statistic)


def expected_max_sharpe(*, n_trials: int, trial_variance: float) -> float:
    """The Sharpe ratio the best of `n_trials` worthless strategies would post.

    This is the number a result has to beat to mean anything. It grows with the
    number of things tried and with how much they differed from each other:
    searching a wide space is what produces a big winner from nothing.

    Args:
        n_trials: Independent hypotheses tested, from the trial ledger. **Not**
            the number kept — the number tried.
        trial_variance: Variance of the Sharpe ratios across those trials.

    Returns:
        The expected maximum Sharpe ratio under the null that every trial has a
        true Sharpe of zero. Zero when `n_trials` is 1: one look is no search.

    Raises:
        ValueError: If `n_trials` is below 1 or `trial_variance` is negative.

    Example:
        >>> round(expected_max_sharpe(n_trials=1, trial_variance=0.04), 4)
        0.0
        >>> round(expected_max_sharpe(n_trials=100, trial_variance=0.04), 4)
        0.5061
    """
    if n_trials < 1:
        raise ValueError(f"n_trials {n_trials} must be at least 1")
    if trial_variance < 0.0:
        raise ValueError(f"trial_variance {trial_variance} must not be negative")
    if n_trials == 1 or trial_variance == 0.0:
        return 0.0
    # The expected maximum of N standard normals, to the order Bailey & López
    # de Prado use, scaled by the spread of the trials.
    gamma = _EULER_MASCHERONI
    first = _NORMAL.inv_cdf(1.0 - 1.0 / n_trials)
    second = _NORMAL.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(trial_variance) * ((1.0 - gamma) * first + gamma * second)


def effective_n_trials(series: Sequence[Sequence[float]]) -> float:
    """How many *independent* looks a set of correlated trials is worth.

    `expected_max_sharpe` assumes every trial is an independent draw. A sweep
    rarely is: nested bands on one rule produce return series that move together,
    so counting them as separate looks sets the hurdle higher than the search
    earned. This is the first-order correction — `N / (1 + (N-1) * mean_corr)`,
    the standard effective-number-of-tests form — driven by the mean pairwise
    Pearson correlation of the trials' return series.

    A negative mean correlation is clamped to zero, so the answer never exceeds
    the trial count: anti-correlated trials are not *more* than independent looks
    for this purpose, and letting them inflate the count would deflate less than
    a genuinely independent search would.

    **A mean cannot see structure, and this is its blind spot.** Two duplicates
    plus one anti-correlated trial average to roughly zero, the clamp returns the
    full count, and three trials worth two looks are charged as three. That errs
    toward deflating more than necessary, so it is safe — but it is exactly what
    clustering the matrix would catch. López de Prado & Lewis (*Quantitative
    Finance* 19(9), 2019) do that (ONC); it needs a stored return series per
    trial, and the ledger keeps only Sharpe ratios, so ONC is the refinement this
    leaves room for rather than the thing implemented.

    Args:
        series: One return series per trial, all the same length, aligned on the
            same observations.

    Returns:
        A count in `[1, len(series)]`. Exactly `len(series)` when the trials are
        uncorrelated, and near 1 when they are near-identical.

    Raises:
        ValueError: If fewer than one series is given, or the series differ in
            length — correlations across misaligned samples are meaningless.

    Example:
        Two variants that agree on every observation are one look:

        >>> a = [0.01, -0.02, 0.03, -0.01, 0.02, 0.00, -0.03, 0.01]
        >>> b = [0.011, -0.019, 0.031, -0.009, 0.021, 0.001, -0.029, 0.011]
        >>> round(effective_n_trials([a, b]), 4)
        1.0

        Partial agreement lands in between:

        >>> m = [0.01, -0.02, 0.03, 0.01, -0.02, 0.00, -0.03, 0.02]
        >>> round(effective_n_trials([a, m]), 4)
        1.2063

        Anti-correlation is clamped, never credited as extra looks:

        >>> u = [0.02, 0.01, -0.01, 0.03, -0.02, -0.03, 0.01, 0.00]
        >>> round(effective_n_trials([a, u]), 4)
        2.0
    """
    n_trials = len(series)
    if n_trials < 1:
        raise ValueError("effective_n_trials needs at least one series")
    lengths = {len(one) for one in series}
    if len(lengths) > 1:
        raise ValueError(f"series lengths differ: {sorted(lengths)}")
    if n_trials == 1:
        return 1.0
    pairs = []
    for left, right in combinations(series, 2):
        try:
            pairs.append(correlation(left, right))
        except StatisticsError:
            # A constant series has no correlation with anything. Scoring the
            # pair as uncorrelated keeps the effective count high, which is the
            # conservative side: it deflates more, not less.
            pairs.append(0.0)
    mean_corr = max(math.fsum(pairs) / len(pairs), 0.0)
    return n_trials / (1.0 + (n_trials - 1) * mean_corr)


def deflated_sharpe_ratio(
    observed: float,
    *,
    trial_sharpes: Sequence[float],
    n_observations: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    n_trials_effective: int | None = None,
) -> float:
    """Confidence that a result survives the search that produced it.

    The probabilistic Sharpe ratio with the benchmark set to what the search
    itself would have produced from noise. A DSR below the promotion threshold
    means the result is indistinguishable from the best of however many things
    were tried — regardless of how good the raw Sharpe looked.

    Args:
        observed: Per-period Sharpe ratio of the candidate.
        trial_sharpes: Per-period Sharpe ratio of **every** trial in the search,
            from the ledger. Their count and variance are both used; passing
            only the survivors understates both and inflates the answer, which
            is the failure this whole module exists to prevent.
        n_observations: Returns the candidate's Sharpe was computed from.
        skew: Third standardized moment of the candidate's returns.
        kurtosis: Fourth standardized moment, not excess.
        n_trials_effective: Independent looks the search was really worth, from
            `effective_n_trials`, when the trials are correlated. Replaces the
            count in the expected maximum **only** — the spread still comes from
            every trial, because correlated trials add no look but do describe
            how wide the space was. Omit it to count every trial as its own look.

    Returns:
        A probability in `(0, 1)`.

    Raises:
        ValueError: If `trial_sharpes` is empty, `n_observations` is below 2, or
            `n_trials_effective` falls outside `[1, len(trial_sharpes)]`.

    Example:
        A Sharpe of 0.1 per day over a year looks strong on its own:

        >>> trials = [0.1] + [0.001 * i - 0.1 for i in range(200)]
        >>> round(probabilistic_sharpe_ratio(0.1, n_observations=250), 3)
        0.942

        Against the 201 trials that produced it, the same number is a coin
        flip away from meaningless:

        >>> round(deflated_sharpe_ratio(0.1, trial_sharpes=trials, n_observations=250), 3)
        0.169
    """
    if not trial_sharpes:
        raise ValueError("deflation needs at least one trial; see lab/trials.py")
    if n_trials_effective is not None and not 1 <= n_trials_effective <= len(trial_sharpes):
        raise ValueError(
            f"n_trials_effective {n_trials_effective} must be between 1 and "
            f"{len(trial_sharpes)}, the trials actually run"
        )
    n_trials = len(trial_sharpes)
    if n_trials == 1:
        trial_variance = 0.0
    else:
        mean = math.fsum(trial_sharpes) / n_trials
        trial_variance = math.fsum((value - mean) ** 2 for value in trial_sharpes) / (n_trials - 1)
    # The *spread* of the search is measured from every trial even when the count
    # is reduced: correlated trials add no independent look, but they do say how
    # wide the space searched was, which is the other half of the hurdle.
    benchmark = expected_max_sharpe(
        n_trials=n_trials if n_trials_effective is None else n_trials_effective,
        trial_variance=trial_variance,
    )
    return probabilistic_sharpe_ratio(
        observed,
        benchmark=benchmark,
        n_observations=n_observations,
        skew=skew,
        kurtosis=kurtosis,
    )


@dataclass(frozen=True, slots=True)
class OverfittingReport:
    """What CSCV found when it re-picked the winner on every split.

    Example:
        >>> report = OverfittingReport(pbo=0.5, n_combinations=252, logits=())
        >>> report.is_overfit
        True
    """

    pbo: float  # fraction of splits whose in-sample winner ranked below median out of sample
    n_combinations: int  # splits examined, C(n_blocks, n_blocks/2)
    logits: tuple[float, ...]  # per-split logit of the winner's OOS rank; <= 0 is a miss

    @property
    def is_overfit(self) -> bool:
        """Whether selection failed to beat a coin flip.

        At `pbo >= 0.5` the in-sample winner is out-of-sample median or worse
        as often as not: the ranking carries no information, so nothing was
        learned by picking the best backtest.
        """
        return self.pbo >= 0.5


def probability_of_backtest_overfitting(
    performance: Sequence[Sequence[float]],
    *,
    n_blocks: int = 10,
) -> OverfittingReport:
    """CSCV: does in-sample rank predict out-of-sample rank?

    The sample is cut into `n_blocks` contiguous blocks. For every way of
    choosing half of them as in-sample, the remaining half is out-of-sample;
    the best in-sample strategy is found and its out-of-sample rank recorded.
    PBO is the fraction of those splits where the winner ranked in the bottom
    half out of sample.

    Blocks are recombined without regard to time order, which is deliberate and
    is the one place in this codebase where that is allowed: CSCV is asking
    about the *stability of a ranking across subsamples*, not about forecasting
    forward. Use `lab/cv.py` for anything that trains a model — there, order is
    the whole point.

    Args:
        performance: Per-period performance, one row per observation, one
            column per strategy configuration tried. Every row must be the same
            length. Returns net of costs, since a ranking on gross returns
            ranks the wrong thing.
        n_blocks: How many blocks to cut the sample into. Must be even and at
            least 4. Larger is more thorough and costs `C(n, n/2)` splits —
            16 blocks is 12,870.

    Returns:
        The report, including the per-split logits so the distribution can be
        inspected rather than only its summary.

    Raises:
        ValueError: If `n_blocks` is odd, below 4, or exceeds the number of
            observations; if fewer than two strategies are supplied; or if the
            rows are ragged.

    Example:
        Two strategies with the same volatility, one earning 1% more per
        period in every block — selection works, so PBO is zero:

        >>> rows = [[0.02, 0.01] if i % 2 else [-0.01, -0.02] for i in range(40)]
        >>> probability_of_backtest_overfitting(rows, n_blocks=4).pbo
        0.0
    """
    if n_blocks < 4 or n_blocks % 2:
        raise ValueError(f"n_blocks {n_blocks} must be even and at least 4")
    n_observations = len(performance)
    if n_observations < n_blocks:
        raise ValueError(f"{n_observations} observations cannot fill {n_blocks} blocks")
    n_strategies = len(performance[0])
    if n_strategies < 2:
        raise ValueError(f"PBO needs at least 2 strategies to rank, got {n_strategies}")
    if any(len(row) != n_strategies for row in performance):
        raise ValueError("every row of `performance` must have one entry per strategy")

    base, extra = divmod(n_observations, n_blocks)
    blocks: list[list[Sequence[float]]] = []
    start = 0
    for index in range(n_blocks):
        size = base + (1 if index < extra else 0)
        blocks.append(list(performance[start : start + size]))
        start += size

    half = n_blocks // 2
    logits: list[float] = []
    for chosen in combinations(range(n_blocks), half):
        in_sample = set(chosen)
        train = [row for index in chosen for row in blocks[index]]
        test = [row for index in range(n_blocks) if index not in in_sample for row in blocks[index]]
        train_scores = [_column_sharpe(train, column) for column in range(n_strategies)]
        test_scores = [_column_sharpe(test, column) for column in range(n_strategies)]
        # Ties broken by the lower column index, so the result does not depend
        # on iteration order — the determinism invariant applies to research
        # code too.
        winner = max(range(n_strategies), key=lambda column: (train_scores[column], -column))
        # Relative rank of the winner out of sample, in (0, 1). Strictly worse
        # competitors counted, so identical scores do not push it to an extreme.
        worse = sum(1 for score in test_scores if score < test_scores[winner])
        omega = (worse + 1) / (n_strategies + 1)
        logits.append(math.log(omega / (1.0 - omega)))

    overfit = sum(1 for value in logits if value <= 0.0)
    return OverfittingReport(
        pbo=overfit / len(logits),
        n_combinations=len(logits),
        logits=tuple(logits),
    )


def _column_sharpe(rows: Sequence[Sequence[float]], column: int) -> float:
    """Per-period Sharpe of one strategy over the given rows.

    A degenerate column — constant returns, which a strategy that never traded
    in this subsample produces — scores zero rather than raising: it is a real
    outcome to rank, and the split it appears in is still informative about the
    others.
    """
    series = [row[column] for row in rows]
    try:
        return sharpe_ratio(series)
    except ValueError:
        return 0.0


def minimum_backtest_length(*, n_trials: int, target_sharpe: float) -> float:
    """Years of out-of-sample a search of this size needs before it means anything.

    Bailey, Borwein, Lopez de Prado and Zhu, *Pseudo-Mathematics and Financial
    Charlatanism* (Notices of the AMS 61(5), 2014), observe that the expected
    maximum annualised Sharpe of `N` independent worthless trials over `T` years
    is about `sqrt(2 * ln(N) / T)` — the same asymptotic `expected_max_sharpe`
    is built on, in annualised units. Setting that equal to the in-sample Sharpe
    actually reported and solving for `T` gives the shortest sample in which
    that Sharpe is not simply what looking `N` times produces from noise. The
    rearrangement is arithmetic, not a second result from the paper.

    **Why print it.** Plan decision 1 keeps each strategy's trials in their own
    family, so `N` stays small and the requirement stays modest. Pooling families
    would change that sharply and silently — at an in-sample Sharpe of 1.0 a
    two-variant search needs 1.4 years and an eight-variant one needs 4.2, which
    is more than the crawl holds. Reporting the number makes the choice visible
    rather than implicit.

    Args:
        n_trials: Independent hypotheses the search burned. At least 2, since a
            search of one has no maximum to be flattered by.
        target_sharpe: The **annualised** in-sample Sharpe being claimed. Every
            other function in this module is per-period; this one is not, because
            the result is a span in years and the two have to agree.

    Returns:
        The required sample length, in years.

    Raises:
        ValueError: If `n_trials` is below 2, or `target_sharpe` is not strictly
            positive — a non-positive Sharpe needs no defending against selection
            bias, and dividing by it would hand back a negative or infinite span.

    Example:
        Two variants, claiming an annualised Sharpe of 1.0:

        >>> round(minimum_backtest_length(n_trials=2, target_sharpe=1.0), 2)
        1.39
        >>> round(minimum_backtest_length(n_trials=8, target_sharpe=1.0), 2)
        4.16
    """
    if n_trials < 2:
        raise ValueError(f"n_trials {n_trials} must be at least 2")
    if target_sharpe <= 0:
        raise ValueError(f"target_sharpe {target_sharpe} must be positive")
    return 2.0 * math.log(n_trials) / target_sharpe**2


@dataclass(frozen=True, slots=True)
class HaircutReport:
    """What three multiple-testing corrections do to one Sharpe ratio.

    Harvey and Liu, *Evaluating Trading Strategies* (Journal of Portfolio
    Management 40(5), 2014; SSRN 2474755), run the correction in **p-value
    space**: convert the Sharpe to a t-statistic, adjust its p-value for the
    number of tests, then convert back. That is a second opinion on `deflated`
    from entirely different machinery — DSR asks how the best of N worthless
    tries would have scored, these ask how surprising this p-value is among N.

    The three disagree on purpose. Bonferroni is the crudest and assumes the
    tests are independent. Holm is a step-down refinement of it and, **for the
    smallest p-value in the set, gives exactly the same answer** — which is worth
    knowing, because the smallest p-value is always the one we are deflating.
    BHY controls the false-discovery rate instead of the family-wise error rate,
    and is the only one of the three that reads the shape of the p-value
    distribution rather than only its size.

    **BHY is not the lenient one here, and that surprised us.** Yekutieli's
    `c(N)` factor — the harmonic sum that buys validity under arbitrary
    dependence — multiplies its whole scale, so at the *winner's* rank it starts
    out `c(N)` times harsher than Bonferroni and only comes back under when some
    lower-ranked test is itself strong enough to pull the step-up minimum down.
    Measured on a search of three: one strong trial beside two weak ones gives
    Bonferroni 0.329 and BHY 0.603, while four jointly strong trials give
    Bonferroni 0.438 and BHY 0.403. A sweep of nested variants around one weak
    edge is the first case, which is ours.

    The haircut is a ratio of Sharpe ratios, so it is unit-free: it is the same
    number whether the inputs are per-period or annualised.

    Example:
        >>> report = HaircutReport(observed=0.05, n_observations=400, n_trials=2,
        ...                        p_value=0.32, bonferroni_p=0.63, holm_p=0.63, bhy_p=0.47)
        >>> report.haircut(report.bonferroni) is not None
        True
    """

    observed: float  # the Sharpe under test, per period, net of costs
    n_observations: int  # returns it was computed from
    n_trials: int  # tests it is being corrected for
    p_value: float  # its own two-sided p-value, uncorrected
    bonferroni_p: float  # p-value after the Bonferroni correction
    holm_p: float  # after Holm's step-down correction
    bhy_p: float  # after Benjamini-Hochberg-Yekutieli

    @property
    def bonferroni(self) -> float:
        """Sharpe ratio implied by `bonferroni_p` at the same sample size."""
        return self._signed(self.bonferroni_p)

    @property
    def holm(self) -> float:
        """Sharpe ratio implied by `holm_p`."""
        return self._signed(self.holm_p)

    @property
    def bhy(self) -> float:
        """Sharpe ratio implied by `bhy_p`."""
        return self._signed(self.bhy_p)

    def _signed(self, p_value: float) -> float:
        """The adjusted Sharpe, carrying `observed`'s sign.

        The p-value is two-sided and therefore blind to direction, so the
        magnitude has to be signed back. Without this a correction applied to a
        **negative** Sharpe reports a positive adjusted one — measured on
        `momentum_ignition`, where an observed -0.0914 printed as bonferroni
        +0.0801 and read as the correction having improved the result.
        """
        magnitude = _sharpe_for_p(p_value, self.n_observations)
        return -magnitude if self.observed < 0 else magnitude

    def haircut(self, adjusted: float) -> float | None:
        """Fraction of the observed Sharpe one correction takes away.

        Args:
            adjusted: One of `bonferroni`, `holm`, `bhy`.

        Returns:
            `1 - adjusted / observed`, or `None` when `observed` is not positive.
            A haircut off a Sharpe that was already negative is not a defined
            quantity — there is no performance to discount — and returning zero
            would read as "the correction cost nothing".

        Example:
            >>> report = HaircutReport(observed=-0.05, n_observations=400, n_trials=2,
            ...                        p_value=0.32, bonferroni_p=0.63, holm_p=0.63,
            ...                        bhy_p=0.47)
            >>> report.haircut(report.bhy) is None
            True
        """
        if self.observed <= 0:
            return None
        return 1.0 - adjusted / self.observed

    def __str__(self) -> str:
        def shown(adjusted: float) -> str:
            cut = self.haircut(adjusted)
            return f"{adjusted:+.4f}" if cut is None else f"{adjusted:+.4f} (-{cut:.0%})"

        return (
            f"p={self.p_value:.3f} over {self.n_trials} tests -> "
            f"bonferroni {shown(self.bonferroni)} "
            f"holm {shown(self.holm)} bhy {shown(self.bhy)}"
        )


def haircut_sharpe(
    observed: float,
    *,
    trial_sharpes: Sequence[float],
    n_observations: int,
) -> HaircutReport:
    """Discount a Sharpe ratio for the number of tests that produced it.

    Args:
        observed: The Sharpe under test, per period. Normally the maximum of
            `trial_sharpes`, which is the one a search would have reported.
        trial_sharpes: Every Sharpe in the search, including `observed` itself.
            The whole set is needed because BHY reads the shape of the p-value
            distribution, not only its size — that is what makes it a different
            opinion rather than a rescaled Bonferroni.
        n_observations: Returns each Sharpe was computed from. Shared across the
            trials, which holds here because the variants are scored over one
            aligned candidate set.

    Returns:
        The report. Every method is reported; none is the answer.

    Raises:
        ValueError: If `trial_sharpes` is empty or `n_observations` is below 2 —
            a t-statistic needs a sample.

    Example:
        A modest Sharpe drawn from a search of three. Bonferroni and Holm agree,
        because `observed` carries the smallest p-value in the set.

        >>> report = haircut_sharpe(0.08, trial_sharpes=[0.08, 0.02, -0.01],
        ...                         n_observations=400)
        >>> report.bonferroni_p == report.holm_p
        True

        With the rest of the search weak, BHY is the harshest of the three; with
        four jointly strong trials it is the mildest. Both directions matter, so
        both are reported and neither is the answer.

        >>> report.bhy_p > report.bonferroni_p
        True
        >>> together = haircut_sharpe(0.08, trial_sharpes=[0.08, 0.075, 0.07, 0.065],
        ...                           n_observations=400)
        >>> together.bhy_p < together.bonferroni_p
        True
    """
    if not trial_sharpes:
        raise ValueError("a haircut needs at least one trial")
    if n_observations < 2:
        raise ValueError(f"n_observations {n_observations} cannot carry a t-statistic")

    n_trials = len(trial_sharpes)
    p_values = sorted(_p_for_sharpe(value, n_observations) for value in trial_sharpes)
    own = _p_for_sharpe(observed, n_observations)
    # The rank the observed p-value occupies among the trials', 1-based. Found by
    # value rather than by position because `observed` need not be an element of
    # `trial_sharpes` — a caller may deflate a pooled winner against a family
    # whose stored Sharpes were computed elsewhere.
    rank = 1 + sum(1 for value in p_values if value < own)

    return HaircutReport(
        observed=observed,
        n_observations=n_observations,
        n_trials=n_trials,
        p_value=own,
        bonferroni_p=min(1.0, n_trials * own),
        holm_p=_holm(p_values, own, rank),
        bhy_p=_bhy(p_values, rank),
    )


def _holm(p_values: Sequence[float], own: float, rank: int) -> float:
    """Holm's step-down adjusted p-value at one rank.

    `max` over the prefix, which is what enforces monotonicity: an adjusted
    p-value may not fall below one assigned to a more significant test. For the
    smallest p-value the prefix is a single term and the result is `N * p`, i.e.
    Bonferroni — stated in `HaircutReport` because it means Holm never rescues a
    search's winner, only its runners-up.
    """
    n_trials = len(p_values)
    prefix = [*p_values[: rank - 1], own]
    return min(1.0, max((n_trials - index) * value for index, value in enumerate(prefix)))


def _bhy(p_values: Sequence[float], rank: int) -> float:
    """Benjamini-Hochberg-Yekutieli adjusted p-value at one rank.

    Step-*up*: the minimum over the suffix, so a large p-value further down the
    ordering can pull an adjusted p-value below its own Bonferroni level. `c(N)`
    is the harmonic sum Yekutieli's correction adds so the guarantee survives
    arbitrary dependence between the tests, which is the case that matters here —
    nested parameter variants of one rule are about as dependent as tests get.
    """
    n_trials = len(p_values)
    harmonic = math.fsum(1.0 / index for index in range(1, n_trials + 1))
    scaled = [
        min(1.0, n_trials * harmonic * value / index)
        for index, value in enumerate(p_values, start=1)
    ]
    return min(scaled[rank - 1 :])


def _p_for_sharpe(sharpe: float, n_observations: int) -> float:
    """Two-sided p-value of a per-period Sharpe ratio.

    `t = SR * sqrt(n)` under the null of a zero Sharpe, read against the normal
    rather than Student's t: at the sample sizes a pooled measurement produces
    (thousands of observations) the two agree to more decimals than anything
    downstream prints, and the rest of this module is normal-based already.

    Two-sided, matching Harvey and Liu's own worked examples, where a
    t-statistic of 3.0 is quoted as p = 0.0027. A one-sided test would halve
    every p-value here and make every haircut look smaller.
    """
    statistic = abs(sharpe) * math.sqrt(n_observations)
    return 2.0 * (1.0 - _NORMAL.cdf(statistic))


def _sharpe_for_p(p_value: float, n_observations: int) -> float:
    """The per-period Sharpe ratio a two-sided p-value implies. Inverse of `_p_for_sharpe`.

    Returns 0.0 at `p >= 1`: the correction has taken the whole result, and
    `inv_cdf(0.5)` is exactly zero anyway — spelled out so the boundary does not
    depend on floating-point luck.
    """
    if p_value >= 1.0:
        return 0.0
    return _NORMAL.inv_cdf(1.0 - p_value / 2.0) / math.sqrt(n_observations)
