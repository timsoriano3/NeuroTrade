"""Which volatility regimes a result actually earned its money in (§13.1, §15).

§15 asks for "positive expectancy in >=2 volatility regimes" and §13.1 makes two
regimes a money gate. Neither defines the term, so this module does, in the one
way available before Phase 5's HMM exists: **bucket sessions by trailing realised
volatility, then report expectancy per bucket over the trades actually taken.**

**Sessions present is the wrong measure, and it is the tempting one.** A single
contiguous year of minute bars spans low, mid and high volatility sessions, so
counting buckets *sampled* would score one macro episode as full coverage. What
§15 asks is whether the edge survived in each bucket, which is a statement about
trades and their returns. `RegimeCoverage.coverage` counts only buckets that both
carry enough trades to mean anything and made money; `sampled` reports presence
separately, so the gap between the two numbers is visible.

**No look-ahead in the volatility, some in the thresholds.** `trailing_volatility`
reads only sessions *before* the one it labels, so the measure itself could drive
a live gate. The tercile boundaries in `bucket_sessions` come from the whole
sample, which is fine for describing a finished measurement and wrong for a live
decision — a live classifier needs a trailing quantile. Standard practice is the
33rd/67th percentile of a trailing 252-session distribution.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from enum import IntEnum
from statistics import fmean

__all__ = [
    "DEFAULT_MIN_TRADES",
    "DEFAULT_VOLATILITY_MEMORY",
    "BucketOutcome",
    "RegimeCoverage",
    "VolatilityBucket",
    "bucket_sessions",
    "coverage_of",
    "trailing_volatility",
]

DEFAULT_VOLATILITY_MEMORY: int = 14
"""Sessions of history behind each session's volatility reading.

Matches `SessionLevels`' `RANGE_MEMORY`, so a regime label and the band a strategy
trades against are built from the same amount of past — a mismatch there would
mean a strategy called "high volatility" by one window and not the other.
"""

DEFAULT_MIN_TRADES: int = 30
"""Trades a bucket needs before its expectancy is treated as a result.

Half of §13.1's 60-trade money gate. Below this a bucket's mean is noise, and
reporting it as a covered regime is how a two-regime claim gets made from eight
trades in one of them.
"""


class VolatilityBucket(IntEnum):
    """Which third of the volatility distribution a session sits in."""

    LOW = 0
    MID = 1
    HIGH = 2


def trailing_volatility(
    ranges: Mapping[date, float], *, memory: int = DEFAULT_VOLATILITY_MEMORY
) -> dict[date, float]:
    """Each session's volatility, read off the sessions before it.

    Args:
        ranges: Per session, a volatility proxy — the market-wide mean of
            `(high - low) / close` is what `measure_strategy` supplies.
        memory: How many prior sessions to average. Must be at least 1.

    Returns:
        One reading per session that has a full `memory` of history behind it.
        The first `memory` sessions are absent rather than averaged over a short
        window, because a band built from three days is not the same statistic.

    Raises:
        ValueError: If `memory` is below 1.

    Example:
        >>> from datetime import date
        >>> days = {date(2024, 1, day): float(day) for day in range(1, 6)}
        >>> readings = trailing_volatility(days, memory=2)
        >>> for day, value in readings.items():
        ...     print(day, round(value, 2))
        2024-01-03 1.5
        2024-01-04 2.5
        2024-01-05 3.5
    """
    if memory < 1:
        raise ValueError(f"memory {memory} must be at least 1")
    sessions = sorted(ranges)
    return {
        session: fmean(ranges[prior] for prior in sessions[index - memory : index])
        for index, session in enumerate(sessions)
        if index >= memory
    }


def bucket_sessions(volatility: Mapping[date, float]) -> dict[date, VolatilityBucket]:
    """Split sessions into volatility terciles of their own distribution.

    Args:
        volatility: Per session, its reading from `trailing_volatility`.

    Returns:
        One bucket per session given. Ties land in the lower bucket, so the
        boundaries are reproducible when many sessions share a reading.

    Raises:
        ValueError: If no sessions are given — there is nothing to split.

    Example:
        >>> from datetime import date
        >>> readings = {date(2024, 1, day): float(day) for day in range(1, 7)}
        >>> buckets = bucket_sessions(readings)
        >>> for day in sorted(buckets):
        ...     print(day, buckets[day].name)
        2024-01-01 LOW
        2024-01-02 LOW
        2024-01-03 MID
        2024-01-04 MID
        2024-01-05 HIGH
        2024-01-06 HIGH
    """
    if not volatility:
        raise ValueError("bucket_sessions needs at least one session")
    ordered = sorted(volatility.values())
    low = ordered[len(ordered) // 3]
    high = ordered[2 * len(ordered) // 3]
    return {
        session: VolatilityBucket.LOW
        if value < low
        else (VolatilityBucket.HIGH if value >= high else VolatilityBucket.MID)
        for session, value in volatility.items()
    }


@dataclass(frozen=True, slots=True)
class BucketOutcome:
    """What one variant earned inside one volatility bucket.

    Example:
        >>> BucketOutcome(bucket=VolatilityBucket.HIGH, n_trades=40,
        ...               expectancy=0.0004).is_result(min_trades=30)
        True
    """

    bucket: VolatilityBucket  # the third of the distribution these trades sat in
    n_trades: int  # trades the variant took in sessions of this bucket
    expectancy: float  # mean return per trade there, net of modelled costs

    def is_result(self, *, min_trades: int) -> bool:
        """Whether this bucket carries enough trades *and* made money."""
        return self.n_trades >= min_trades and self.expectancy > 0


@dataclass(frozen=True, slots=True)
class RegimeCoverage:
    """Per-bucket expectancy, and how many regimes it amounts to.

    Example:
        >>> coverage = RegimeCoverage(
        ...     outcomes=(
        ...         BucketOutcome(VolatilityBucket.LOW, 50, 0.0002),
        ...         BucketOutcome(VolatilityBucket.MID, 40, 0.0001),
        ...         BucketOutcome(VolatilityBucket.HIGH, 8, 0.0009),
        ...     ),
        ...     min_trades=30,
        ... )
        >>> (coverage.coverage, coverage.sampled, coverage.meets_spec)
        (2, 2, True)
    """

    outcomes: tuple[BucketOutcome, ...]  # one per bucket that saw any trade, ascending
    min_trades: int  # trades a bucket needs before it counts either way

    @property
    def coverage(self) -> int:
        """Buckets with enough trades and positive expectancy. §15's number."""
        return sum(outcome.is_result(min_trades=self.min_trades) for outcome in self.outcomes)

    @property
    def sampled(self) -> int:
        """Buckets with enough trades, whether or not they made money.

        Reported beside `coverage` because the difference is the finding: three
        sampled and one covered says the edge is regime-specific, which a single
        coverage count would hide.
        """
        return sum(outcome.n_trades >= self.min_trades for outcome in self.outcomes)

    @property
    def meets_spec(self) -> bool:
        """Whether §15's ">=2 volatility regimes" is satisfied."""
        return self.coverage >= 2

    def __str__(self) -> str:
        parts = " ".join(
            f"{outcome.bucket.name.lower()}={outcome.expectancy:+.5f}/{outcome.n_trades}t"
            for outcome in self.outcomes
        )
        return f"regimes {self.coverage}/{self.sampled} covered (min {self.min_trades}t)  {parts}"


def coverage_of(
    buckets: Sequence[VolatilityBucket | None],
    returns: Sequence[float | None],
    *,
    min_trades: int = DEFAULT_MIN_TRADES,
) -> RegimeCoverage:
    """Group a variant's realised returns by the volatility of the session.

    Args:
        buckets: Per observation, the bucket of the session it opened in, or
            None when that session has no volatility reading — the sample's first
            sessions, which have no history behind them.
        returns: Per observation, the realised return of the trade taken there,
            or None where the variant did not trade. Must align with `buckets`.
        min_trades: Trades a bucket needs before it counts.

    Returns:
        One `BucketOutcome` per bucket that saw at least one trade, ascending.

    Raises:
        ValueError: If the two sequences differ in length, or `min_trades` is
            below 1.

    Example:
        >>> low, high = VolatilityBucket.LOW, VolatilityBucket.HIGH
        >>> got = coverage_of([low, low, high, None], [0.01, -0.02, 0.03, 0.04],
        ...                   min_trades=2)
        >>> for outcome in got.outcomes:
        ...     print(outcome.bucket.name, outcome.n_trades, round(outcome.expectancy, 4))
        LOW 2 -0.005
        HIGH 1 0.03
        >>> (got.coverage, got.sampled)
        (0, 1)
    """
    if len(buckets) != len(returns):
        raise ValueError(f"buckets {len(buckets)} and returns {len(returns)} must align")
    if min_trades < 1:
        raise ValueError(f"min_trades {min_trades} must be at least 1")
    grouped: dict[VolatilityBucket, list[float]] = {}
    for bucket, value in zip(buckets, returns, strict=True):
        # An observation with no reading is dropped rather than assigned a
        # bucket. Folding the unclassifiable sessions into MID would put trades
        # in a regime nothing measured.
        if bucket is None or value is None:
            continue
        grouped.setdefault(bucket, []).append(value)
    return RegimeCoverage(
        outcomes=tuple(
            BucketOutcome(bucket=bucket, n_trades=len(values), expectancy=fmean(values))
            for bucket, values in sorted(grouped.items())
        ),
        min_trades=min_trades,
    )
