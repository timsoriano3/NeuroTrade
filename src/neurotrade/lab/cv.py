"""Combinatorial Purged Cross-Validation — §8's primary validation gate.

Ordinary k-fold cross-validation is invalid on financial data, for two reasons
that compound.

**Labels overlap in time.** A triple-barrier label opened at bar 100 and
running thirty bars shares twenty-nine bars with one opened at bar 101. Put one
in the training set and the other in the test set and the model has been
trained on the answer it is about to be tested on. The leak is invisible: no
column is shared, no index is duplicated, and the score simply comes out too
high. **Purging** removes from the training set every observation whose label
span overlaps any test label's span.

**Serial correlation leaks forward past the purge.** Features are autocorrelated
— a volatility estimate at bar 130 is nearly the one at bar 131 — so a training
observation that starts just after the test set ends still carries information
about it. **Embargo** drops a further window of training observations
immediately after each test block.

**And a single train/test split gives a single number.** One backtest path
cannot distinguish a strategy that is good from one that was lucky once. CPCV
splits the sample into `n_groups` blocks, takes every combination of
`n_test_groups` of them as a test set, and so produces `C(N, k)` splits and
`C(N, k) * k / N` distinct backtest *paths*. A distribution of outcomes instead
of a point, which is what the deflated Sharpe and PBO in `lab/significance.py`
need as input.

**This is combinatorial, not a renamed k-fold.** With `n_test_groups=1` it
degenerates to purged k-fold: `C(N, 1) = N` splits and exactly one path. That
is a legitimate configuration for a quick check and it is *not* the gate — §8
asks for CPCV as the primary gate and walk-forward as the secondary sanity
check, and a k-fold wearing the CPCV name would provide neither.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import combinations
from math import comb

__all__ = [
    "CombinatorialPurgedCV",
    "Split",
    "WalkForward",
    "purge_and_embargo",
]


@dataclass(frozen=True, slots=True)
class Split:
    """One train/test partition.

    Example:
        >>> split = Split(train=(0, 1), test=(3, 4), test_groups=(1,), purged=2, embargoed=0)
        >>> (len(split.train), split.dropped)
        (2, 2)
    """

    train: tuple[int, ...]  # observation indices to fit on, after purge and embargo
    test: tuple[int, ...]  # observation indices to evaluate on
    test_groups: tuple[int, ...]  # which group indices formed the test set
    purged: int  # training observations dropped for overlapping a test label
    embargoed: int  # training observations dropped by the embargo window

    @property
    def dropped(self) -> int:
        """Training observations removed by purging and embargo together.

        Worth watching: if this approaches the size of the training set, the
        labels are long relative to the sample and the split is not really
        training on anything. A silent near-empty training set produces a model
        that is noise, and noise can still score well on one path.
        """
        return self.purged + self.embargoed


def purge_and_embargo(
    candidates: Sequence[int],
    test_blocks: Sequence[Sequence[int]],
    spans: Sequence[tuple[int, int]],
    *,
    embargo: int,
) -> tuple[tuple[int, ...], int, int]:
    """Drop training observations that leak information from the test set.

    **`test_blocks`, not one flat test set.** A CPCV test set is made of `k`
    groups that are usually *not* adjacent — groups 0 and 5 of 6, say. Treating
    them as one span from the first group's start to the last group's end would
    purge the entire middle of the sample, which is the training data. Each
    contiguous run of test groups is purged against separately, so a split
    testing the two ends of the sample still trains on the middle.

    Args:
        candidates: Training observation indices, before removal.
        test_blocks: One sequence of test observation indices per *contiguous*
            run of test groups. Non-contiguous runs must be passed as separate
            blocks — that is the whole point of the argument.
        spans: `(start, end)` bar indices per observation, both inclusive,
            indexed the same way as the observations themselves. These are the
            *label* spans — how long each position stayed open — not the
            feature windows.
        embargo: Bars after a test block during which training observations are
            also dropped. Zero disables it, which is only safe if the features
            carry no serial correlation, which they always do.

    Returns:
        The surviving training indices, how many were purged, and how many were
        embargoed. An observation removed for both reasons counts as purged.

    Raises:
        ValueError: If `embargo` is negative.

    Example:
        >>> spans = [(0, 5), (10, 15), (20, 25)]
        >>> purge_and_embargo([0, 2], [[1]], spans, embargo=0)
        ((0, 2), 0, 0)
        >>> purge_and_embargo([0, 2], [[1]], spans, embargo=10)
        ((0,), 0, 1)

        Two separate test blocks leave the observation between them alone:

        >>> spans = [(0, 5), (10, 15), (20, 25), (30, 35), (40, 45)]
        >>> purge_and_embargo([1, 2, 3], [[0], [4]], spans, embargo=0)
        ((1, 2, 3), 0, 0)
    """
    if embargo < 0:
        raise ValueError(f"embargo {embargo} must not be negative")
    intervals = [
        (min(spans[index][0] for index in block), max(spans[index][1] for index in block))
        for block in test_blocks
        if block
    ]
    if not intervals:
        return tuple(candidates), 0, 0

    kept: list[int] = []
    purged = embargoed = 0
    for index in candidates:
        start, end = spans[index]
        # Purge: any temporal overlap at all between this label and a test
        # block's span. Not "starts inside" — a label that opened before the
        # block and closed inside it saw the same bars.
        if any(start <= block_end and end >= block_start for block_start, block_end in intervals):
            purged += 1
            continue
        # Embargo: begins after a test block but too soon after it. Only
        # forward — a label that closed before a block began cannot carry
        # information from it.
        if any(block_end < start <= block_end + embargo for _, block_end in intervals):
            embargoed += 1
            continue
        kept.append(index)
    return tuple(kept), purged, embargoed


@dataclass(frozen=True, slots=True)
class CombinatorialPurgedCV:
    """CPCV: every combination of `n_test_groups` blocks, purged and embargoed.

    Example:
        >>> cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, embargo=0)
        >>> (cv.n_splits, cv.n_paths)
        (15, 5)
    """

    n_groups: int = 6  # blocks the sample is cut into
    n_test_groups: int = 2  # blocks held out per split; >1 is what makes it combinatorial
    embargo: int = 0  # bars of embargo after each test block

    def __post_init__(self) -> None:
        """Validate the configuration.

        Raises:
            ValueError: If the group counts are not sensible, or the embargo is
                negative.
        """
        if self.n_groups < 2:
            raise ValueError(f"n_groups {self.n_groups} must be at least 2")
        if not 1 <= self.n_test_groups < self.n_groups:
            raise ValueError(
                f"n_test_groups {self.n_test_groups} must be at least 1 and "
                f"below n_groups {self.n_groups}"
            )
        if self.embargo < 0:
            raise ValueError(f"embargo {self.embargo} must not be negative")

    @property
    def n_splits(self) -> int:
        """How many train/test partitions this produces: `C(N, k)`."""
        return comb(self.n_groups, self.n_test_groups)

    @property
    def n_paths(self) -> int:
        """How many distinct backtest paths the splits assemble into.

        `C(N, k) * k / N`. Each group appears in the test set of that many
        splits, so that many complete walk-throughs of the sample can be
        reconstructed — and it is the *spread* across those paths, not any one
        of them, that says whether a result is real.
        """
        return self.n_splits * self.n_test_groups // self.n_groups

    @property
    def is_combinatorial(self) -> bool:
        """Whether this is genuinely CPCV rather than purged k-fold.

        `n_test_groups == 1` gives one path and no distribution. Legitimate as
        a quick check; not the gate §8 asks for.
        """
        return self.n_test_groups > 1

    def groups(self, n_observations: int) -> tuple[tuple[int, ...], ...]:
        """Cut the observations into contiguous, near-equal blocks.

        Contiguous and in time order, never shuffled: the whole point is that
        training and test are separated *in time*, and a shuffle would scatter
        each block across the sample so that no purge could separate them.

        Args:
            n_observations: How many observations there are.

        Returns:
            One tuple of indices per group.

        Raises:
            ValueError: If there are fewer observations than groups.
        """
        if n_observations < self.n_groups:
            raise ValueError(f"{n_observations} observations cannot fill {self.n_groups} groups")
        base, extra = divmod(n_observations, self.n_groups)
        blocks: list[tuple[int, ...]] = []
        start = 0
        for index in range(self.n_groups):
            size = base + (1 if index < extra else 0)
            blocks.append(tuple(range(start, start + size)))
            start += size
        return tuple(blocks)

    def split(self, spans: Sequence[tuple[int, int]]) -> Iterator[Split]:
        """Yield every purged, embargoed train/test partition.

        Args:
            spans: `(start, end)` label spans per observation, both inclusive,
                in observation order. Purging needs these — an index alone
                cannot say when a label closed.

        Yields:
            `C(n_groups, n_test_groups)` splits, in combination order.

        Raises:
            ValueError: If there are fewer observations than groups.

        Example:
            >>> cv = CombinatorialPurgedCV(n_groups=4, n_test_groups=2, embargo=0)
            >>> spans = [(i * 10, i * 10 + 5) for i in range(8)]
            >>> len(list(cv.split(spans)))
            6
        """
        blocks = self.groups(len(spans))
        for test_groups in combinations(range(self.n_groups), self.n_test_groups):
            test_blocks = [
                [index for group in run for index in blocks[group]]
                for run in _contiguous_runs(test_groups)
            ]
            test = tuple(index for block in test_blocks for index in block)
            candidates = [
                index
                for group in range(self.n_groups)
                if group not in test_groups
                for index in blocks[group]
            ]
            train, purged, embargoed = purge_and_embargo(
                candidates, test_blocks, spans, embargo=self.embargo
            )
            yield Split(
                train=train,
                test=tuple(sorted(test)),
                test_groups=test_groups,
                purged=purged,
                embargoed=embargoed,
            )

    def paths(self) -> tuple[tuple[tuple[int, int], ...], ...]:
        """Assemble the splits into complete walk-throughs of the sample.

        Each split predicts on `n_test_groups` groups, so no single split
        covers the sample. But every group is tested in exactly `n_paths`
        different splits, so the predictions can be re-dealt into `n_paths`
        complete series, each covering every group exactly once, each a
        different combination of the models that produced it.

        That is the output the rest of §8 consumes: `lab/significance.py` needs
        a *distribution* of backtest outcomes — the variance across paths is
        what deflates the Sharpe and what PBO is computed over. One path is a
        backtest; the spread across paths is evidence.

        Returns:
            One tuple per path, holding `(split index, group)` pairs in group
            order — i.e. in time order. Pair `(s, g)` means "take this path's
            predictions for group `g` from split `s`".

        Example:
            >>> cv = CombinatorialPurgedCV(n_groups=4, n_test_groups=2)
            >>> paths = cv.paths()
            >>> (len(paths), len(paths[0]))
            (3, 4)
            >>> paths[0]
            ((0, 0), (0, 1), (1, 2), (2, 3))
        """
        used = [0] * self.n_groups
        assembled: list[list[tuple[int, int]]] = [[] for _ in range(self.n_paths)]
        for split_index, test_groups in enumerate(
            combinations(range(self.n_groups), self.n_test_groups)
        ):
            for group in test_groups:
                assembled[used[group]].append((split_index, group))
                used[group] += 1
        return tuple(tuple(path) for path in assembled)


def _contiguous_runs(groups: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    """Split ascending group indices into maximal runs of consecutive values.

    Example:
        >>> _contiguous_runs((0, 1, 3, 5, 6))
        ((0, 1), (3,), (5, 6))
    """
    runs: list[list[int]] = []
    for group in groups:
        if runs and group == runs[-1][-1] + 1:
            runs[-1].append(group)
        else:
            runs.append([group])
    return tuple(tuple(run) for run in runs)


@dataclass(frozen=True, slots=True)
class WalkForward:
    """The secondary sanity check §8 asks for: train on the past, test forward.

    CPCV is the gate because it gives a distribution. Walk-forward gives one
    path, which is its weakness as evidence and its strength as a check: it is
    the only scheme in which every test observation is strictly *after* every
    observation the model saw, so it cannot be wrong in the direction that
    flatters a strategy. If CPCV says a strategy works and walk-forward says it
    never worked going forward, the disagreement is the finding.

    Training is expanding by default — each fold trains on everything before
    it. `expanding=False` gives a rolling window of the previous fold only,
    which is the right choice when the relationship being learned is not
    expected to be stable across the whole sample.

    Example:
        >>> wf = WalkForward(n_splits=3)
        >>> spans = [(i * 10, i * 10 + 5) for i in range(12)]
        >>> [split.test_groups for split in wf.split(spans)]
        [(1,), (2,), (3,)]
    """

    n_splits: int = 5  # forward folds; the first block is training-only
    embargo: int = 0  # bars of embargo between train and test
    expanding: bool = True  # train on everything before the fold, or only the previous fold

    def __post_init__(self) -> None:
        """Validate the configuration.

        Raises:
            ValueError: If `n_splits` is below 1 or `embargo` is negative.
        """
        if self.n_splits < 1:
            raise ValueError(f"n_splits {self.n_splits} must be at least 1")
        if self.embargo < 0:
            raise ValueError(f"embargo {self.embargo} must not be negative")

    def split(self, spans: Sequence[tuple[int, int]]) -> Iterator[Split]:
        """Yield the forward folds, purged and embargoed.

        The sample is cut into `n_splits + 1` blocks: block 0 is training-only,
        and each later block is tested in turn. Purging still applies — a
        training label that closed *after* the test block opened overlaps it,
        and "the training data came first" is about when positions opened, not
        when they closed.

        Args:
            spans: `(start, end)` label spans per observation, in order.

        Yields:
            `n_splits` splits, earliest test block first.

        Raises:
            ValueError: If there are fewer observations than blocks.

        Example:
            >>> wf = WalkForward(n_splits=2, expanding=False)
            >>> spans = [(i * 10, i * 10 + 5) for i in range(9)]
            >>> [(split.train, split.test) for split in wf.split(spans)]
            [((0, 1, 2), (3, 4, 5)), ((3, 4, 5), (6, 7, 8))]
        """
        blocks = CombinatorialPurgedCV(
            n_groups=self.n_splits + 1, n_test_groups=1, embargo=self.embargo
        ).groups(len(spans))
        for fold in range(1, self.n_splits + 1):
            test = list(blocks[fold])
            earlier = range(fold) if self.expanding else range(fold - 1, fold)
            candidates = [index for group in earlier for index in blocks[group]]
            train, purged, embargoed = purge_and_embargo(
                candidates, [test], spans, embargo=self.embargo
            )
            yield Split(
                train=train,
                test=tuple(test),
                test_groups=(fold,),
                purged=purged,
                embargoed=embargoed,
            )
