"""How much of a Sharpe ratio survives the sample not being independent (§8).

Every Sharpe ratio this lab reports is computed over *observations* — one per
decision a strategy took. The deflation machinery in `lab/significance.py` then
charges the result for the size of that sample. Both are right only if the
observations are independent bets, and at one-minute bars over a pooled universe
they are not: a strategy that decides seven times in a session has read that
session once, and ten instruments breaking out on one morning is one market
event.

`lab/evaluation.py` already reports the two ends of that interval. `deflated`
takes the raw observation count, which is optimistic; `deflated_clustered` takes
the session count, which is a **bound** — it holds the return distribution fixed
and removes all the sample-size credit above one observation per session, so it
cannot flatter a result and cannot be the truth either. On the crawled window the
two ends are 0.537 and 0.514 for `intraday_momentum`, which is the difference
between an edge and a coin flip, and nothing said where between them the answer
sat.

This module measures it. Resample whole **sessions**, with replacement, and look
at how much the Sharpe ratio actually moves. The spread of the resampled Sharpe
ratios is an estimate of its standard error under the dependence the sample
really has, and that standard error converts straight back into an *effective*
observation count — the number of independent bets this sample is worth. That
count is what `deflated_sharpe_ratio` should be charged at, and it lands between
the two ends by construction rather than by assumption.

**Why the session is the block.** It is the coarsest grouping that is honest
without being wasteful. Within a session everything shares the same news, the
same opening auction and the same regime; across sessions the overnight break
resets the order book and most of the participants. Sessions are also what
`lab/measure.py` already counts, so the two numbers describe the same grouping.

**Stationary, not fixed-length.** Blocks are drawn with a geometric length, the
Politis-Romano construction, so a resample is stationary rather than carrying
the artefacts of a fixed block boundary. The block unit is a session and the
default mean length is one session, which makes the default exactly "resample
whole sessions, independently" — the fix `06-out-of-sample-and-inference.doc.md`
ranked first. Raising `mean_block` above 1 models dependence *across* sessions
(a volatility cluster spanning a week), which is a stronger claim about the data
and is therefore opt-in rather than the default.

**A longer block is not a safety dial, which is not the intuition.** It reads as
"more conservative", and on sessions that really are independent it is the
opposite: each draw reproduces more of the original ordering, the resampled
Sharpe ratios cluster tighter, and the interval *narrows*. Measured, and pinned
in `test_a_longer_block_narrows_the_interval_on_independent_sessions`. So
`mean_block` states a belief about the data and has to be justified from the
data; the default states the weakest belief this project can defend.

**`random.Random`, not numpy.** `lab/controls.py` already generates every
synthetic series this way, the stack is numpy + pandas + duckdb and `arch` (whose
`StationaryBootstrap` this reimplements) would drag scipy and statsmodels in
behind it. The seed is an explicit argument with a fixed default, so two runs of
the same measurement produce the same interval — the determinism invariant
applies to a confidence interval exactly as it does to a digest.
"""

from __future__ import annotations

import math
import random
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Final

from neurotrade.lab.significance import sharpe_ratio

__all__ = [
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_MEAN_BLOCK",
    "DEFAULT_RESAMPLES",
    "SharpeInterval",
    "bootstrap_sharpe",
    "group_by_cluster",
]

DEFAULT_RESAMPLES: Final = 2_000
"""Resamples per interval.

Two thousand puts the Monte Carlo error on a 95% percentile bound at roughly a
fortieth of the interval's own width, which is far below the resolution anything
downstream reads it at. Ten thousand costs five times as much and moves the
third decimal of a number whose second decimal is the finding."""

DEFAULT_MEAN_BLOCK: Final = 1.0
"""Mean block length, in sessions.

One session means blocks of exactly one session: the sample is treated as
independent *across* sessions and arbitrarily dependent *within* them, which is
the claim this project can defend. Above one prices dependence across sessions
too, and is a stronger claim about the data than the corpus has been asked to
support."""

DEFAULT_BOOTSTRAP_SEED: Final = 20260929
"""Fixed so an interval is reproducible.

Same reasoning as `lab/gate.py`'s `DEFAULT_SEED`: a result that holds on one seed
only is fitted to its own resampling, so the tests sweep seeds while the reported
number comes from one."""

_MIN_CLUSTERS: Final = 2
"""Below two clusters there is nothing to resample — every draw returns the same
series and the interval collapses to a point, which would read as certainty."""


@dataclass(frozen=True, slots=True)
class SharpeInterval:
    """A Sharpe ratio, and what resampling its sessions did to it.

    Example:
        >>> interval = SharpeInterval(
        ...     observed=0.01, lower=-0.002, upper=0.022, standard_error=0.006,
        ...     n_observations=4503, n_clusters=644, n_resamples=2000,
        ...     mean_block=1.0, confidence=0.95,
        ... )
        >>> interval.excludes_zero
        False
    """

    observed: float  # the point estimate, per observation, net of costs
    lower: float  # lower percentile bound of the resampled distribution
    upper: float  # upper percentile bound
    standard_error: float  # standard deviation of the resampled Sharpe ratios
    n_observations: int  # observations the point estimate was computed from
    n_clusters: int  # sessions those observations fell in; the resampling unit
    n_resamples: int  # resamples drawn
    mean_block: float  # mean block length, in clusters
    confidence: float  # two-sided coverage the bounds were cut at

    @property
    def excludes_zero(self) -> bool:
        """Whether the interval is entirely on one side of zero.

        The weakest claim worth making about a strategy: that resampling its own
        sessions does not routinely turn its edge into a loss. It is not
        significance — the search that found it is not priced here, which is what
        `deflated` is for.
        """
        return self.lower > 0.0 or self.upper < 0.0

    @property
    def effective_n(self) -> float:
        """Independent observations this sample is worth.

        Under independence the Sharpe estimator's variance is about
        `(1 + SR^2 / 2) / T` (Lo 2002, *The Statistics of Sharpe Ratios*,
        Financial Analysts Journal 58(4), equation 9, at zero autocorrelation).
        The bootstrap measures the left-hand side without assuming independence,
        so rearranging for `T` gives the sample size a dependent sample behaves
        like. That is the number `deflated_sharpe_ratio` should be charged at,
        and it falls between `n_clusters` and `n_observations` whenever the
        dependence is real but not total.

        **Capped at `n_observations`, and not floored at `n_clusters`.** The cap
        is a fact rather than a guess: a sample cannot behave like more
        independent bets than it holds observations, and on a short sample the
        resampled spread is itself noisy enough to imply that it does. There is
        deliberately no floor, because dependence *across* sessions is possible
        and a floor at the session count would assert it away — which is the
        assumption this whole module exists to replace with a measurement.

        Returns:
            The implied count, in `[1.0, n_observations]`. Falls back to
            `n_clusters` when the resampled spread is zero, which only happens on
            a degenerate sample and would otherwise divide by zero.

        Example:
            >>> SharpeInterval(observed=0.0072, lower=0.0, upper=0.0, standard_error=0.0,
            ...                n_observations=4503, n_clusters=644, n_resamples=2000,
            ...                mean_block=1.0, confidence=0.95).effective_n
            644.0
        """
        if self.standard_error <= 0.0:
            return float(self.n_clusters)
        implied = (1.0 + self.observed**2 / 2.0) / self.standard_error**2
        return min(float(self.n_observations), max(1.0, implied))

    @property
    def inflation(self) -> float:
        """How many times more independent the raw count claims to be.

        `n_observations / effective_n`. The square root of this is what the
        PSR z-score is overstated by, since it carries `sqrt(n - 1)`.
        """
        return self.n_observations / self.effective_n

    def __str__(self) -> str:
        return (
            f"sharpe={self.observed:+.4f} "
            f"[{self.lower:+.4f}, {self.upper:+.4f}] at {self.confidence:.0%} "
            f"n_eff={self.effective_n:.0f} of {self.n_observations} "
            f"({self.n_clusters} sessions, {self.inflation:.1f}x claimed)"
        )


def group_by_cluster(
    returns: Sequence[float], clusters: Sequence[Hashable]
) -> tuple[tuple[float, ...], ...]:
    """Split returns into blocks, one per cluster, in first-appearance order.

    Args:
        returns: One per observation.
        clusters: The cluster each observation belongs to, aligned with
            `returns`. Session dates, for this project. Anything hashable;
            equality is all that is used, so a `date` and a string both work.

    Returns:
        One tuple of returns per distinct cluster, ordered by where the cluster
        first appears. Order is from the data rather than from `sorted()`,
        because a cluster key need not be orderable and the determinism
        invariant forbids leaning on set or dict iteration for it.

    Raises:
        ValueError: If the two sequences are different lengths. A silent `zip`
            would drop the tail, and the tail of a pooled sample is its most
            recent sessions.

    Example:
        >>> group_by_cluster([0.1, 0.2, 0.3], ["mon", "mon", "tue"])
        ((0.1, 0.2), (0.3,))
    """
    if len(returns) != len(clusters):
        raise ValueError(f"{len(returns)} returns against {len(clusters)} cluster keys")
    blocks: dict[Hashable, list[float]] = {}
    for value, key in zip(returns, clusters, strict=True):
        blocks.setdefault(key, []).append(value)
    # Python dicts preserve insertion order, so this is first-appearance order
    # and is reproducible without requiring the keys to be sortable.
    return tuple(tuple(block) for block in blocks.values())


def bootstrap_sharpe(
    returns: Sequence[float],
    clusters: Sequence[Hashable],
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    mean_block: float = DEFAULT_MEAN_BLOCK,
    confidence: float = 0.95,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> SharpeInterval | None:
    """Resample whole sessions to price the Sharpe ratio's dependence.

    Args:
        returns: Per-observation returns, net of costs — the same series
            `Evaluation.best_sharpe` was computed from.
        clusters: The session each observation opened in, aligned with
            `returns`.
        n_resamples: How many resamples to draw.
        mean_block: Mean block length in *clusters*, for the geometric draw. One
            resamples sessions independently; above one prices dependence across
            them as well.
        confidence: Two-sided coverage for the percentile bounds.
        seed: Fixed so the interval is reproducible.

    Returns:
        The interval, or `None` when the sample cannot carry one: fewer than two
        clusters, or a point estimate that does not exist because the returns
        have no variance. `None` rather than a raise, because a thin sample is a
        finding `measure_strategy` reports beside the rest.

    Raises:
        ValueError: If `n_resamples` is not positive, `mean_block` is below one,
            or `confidence` is not strictly inside `(0, 1)`.

    Example:
        Sixteen observations over four sessions. Four sessions is far too few to
        resample meaningfully, which is visible in the result rather than hidden:
        the implied effective count runs into its cap at the observation count.

        >>> returns = [0.01, -0.005, 0.02, 0.0, 0.015, -0.01, 0.005, 0.01,
        ...            -0.002, 0.012, 0.004, -0.008, 0.02, 0.001, -0.003, 0.009]
        >>> sessions = ["mon"] * 4 + ["tue"] * 4 + ["wed"] * 4 + ["thu"] * 4
        >>> interval = bootstrap_sharpe(returns, sessions, n_resamples=200)
        >>> interval.n_clusters, interval.n_observations
        (4, 16)
        >>> interval.effective_n == interval.n_observations    # the cap, binding
        True
    """
    if n_resamples < 1:
        raise ValueError(f"n_resamples {n_resamples} must be at least 1")
    if mean_block < 1.0:
        raise ValueError(f"mean_block {mean_block} must be at least one cluster")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence {confidence} must be strictly inside (0, 1)")

    blocks = group_by_cluster(returns, clusters)
    if len(blocks) < _MIN_CLUSTERS:
        return None
    observed = _sharpe_or_none(returns)
    if observed is None:
        return None

    # Politis-Romano: restart probability 1/L, so block lengths are geometric
    # with mean L and the resampled series is stationary. At L = 1 the
    # probability is 1 and every block is one session, which is the default.
    restart = 1.0 / mean_block
    rng = random.Random(seed)
    n_blocks = len(blocks)
    target = len(returns)
    drawn: list[float] = []
    for _ in range(n_resamples):
        drawn_sharpe = _one_resample(blocks, rng, restart=restart, target=target)
        if drawn_sharpe is not None:
            drawn.append(drawn_sharpe)
    if len(drawn) < 2:
        # Every resample was flat. Possible only on a degenerate sample; report
        # no interval rather than a zero-width one, which would read as certainty.
        return None

    drawn.sort()
    tail = (1.0 - confidence) / 2.0
    mean = math.fsum(drawn) / len(drawn)
    variance = math.fsum((value - mean) ** 2 for value in drawn) / (len(drawn) - 1)
    return SharpeInterval(
        observed=observed,
        lower=_percentile(drawn, tail),
        upper=_percentile(drawn, 1.0 - tail),
        standard_error=math.sqrt(variance),
        n_observations=len(returns),
        n_clusters=n_blocks,
        n_resamples=len(drawn),
        mean_block=mean_block,
        confidence=confidence,
    )


def _one_resample(
    blocks: Sequence[Sequence[float]],
    rng: random.Random,
    *,
    restart: float,
    target: int,
) -> float | None:
    """One stationary-bootstrap draw, as a Sharpe ratio.

    Blocks are appended until at least `target` observations are held, so a
    resample is never shorter than the original sample. It may be longer by up to
    one session, which is why the Sharpe is computed over what was drawn rather
    than over a truncation — cutting mid-session would break exactly the grouping
    the block structure exists to preserve.
    """
    n_blocks = len(blocks)
    index = rng.randrange(n_blocks)
    drawn: list[float] = []
    while len(drawn) < target:
        drawn.extend(blocks[index])
        # Wrap rather than stop at the end: circular resampling is what keeps
        # every observation equally likely to be drawn, so the first and last
        # sessions are not under-weighted.
        index = rng.randrange(n_blocks) if rng.random() < restart else (index + 1) % n_blocks
    return _sharpe_or_none(drawn)


def _sharpe_or_none(returns: Sequence[float]) -> float | None:
    """`sharpe_ratio`, or `None` where it is undefined.

    `moments` raises on zero variance rather than returning a stdev of zero, and
    a resample that happens to draw one flat session repeatedly is a real
    outcome, not an error. Tested on the values because `moments(...).stdev`
    cannot be read to find out — it raises first.
    """
    if len(returns) < 2 or len(set(returns)) < 2:
        return None
    return sharpe_ratio(returns)


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    """Linear-interpolated percentile of an already-sorted sequence.

    Written out rather than taken from `statistics.quantiles`, which cuts at
    `n + 1` boundaries and therefore cannot return the extremes — on 200
    resamples at 2.5% that difference is visible in the reported bound.
    """
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = fraction * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight
