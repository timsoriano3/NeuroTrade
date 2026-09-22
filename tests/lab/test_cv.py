"""Tests for CPCV, purging, embargo and walk-forward.

The defect this file exists to catch is not a crash. It is a split that leaks:
a training observation whose label overlapped the test block, left in the
training set, producing a score that is too high and no error at all. So most
of what follows asserts about *what was removed*, and the leak tests check the
property directly — no surviving training label may overlap any test label.
"""

from __future__ import annotations

from math import comb

import pytest

from neurotrade.lab.cv import CombinatorialPurgedCV, Split, WalkForward, purge_and_embargo


def spans(count: int, *, length: int = 5, step: int = 10) -> list[tuple[int, int]]:
    """`count` label spans of `length` bars, opening every `step` bars."""
    return [(index * step, index * step + length) for index in range(count)]


def overlapping_spans(count: int, *, length: int) -> list[tuple[int, int]]:
    """Spans that open one bar apart and run `length` bars — heavily overlapping."""
    return [(index, index + length) for index in range(count)]


# ── purge_and_embargo ────────────────────────────────────────


def test_purges_a_label_that_closes_inside_the_test_block() -> None:
    # Opens before the block, closes inside it: the classic leak that a
    # "starts inside the block" rule would miss.
    label_spans = [(0, 25), (20, 30)]
    kept, purged, embargoed = purge_and_embargo([0], [[1]], label_spans, embargo=0)
    assert (kept, purged, embargoed) == ((), 1, 0)


def test_purges_a_label_that_opens_inside_the_test_block() -> None:
    label_spans = [(22, 40), (20, 30)]
    kept, purged, _ = purge_and_embargo([0], [[1]], label_spans, embargo=0)
    assert (kept, purged) == ((), 1)


def test_purges_a_label_that_straddles_the_test_block() -> None:
    label_spans = [(0, 100), (20, 30)]
    kept, purged, _ = purge_and_embargo([0], [[1]], label_spans, embargo=0)
    assert (kept, purged) == ((), 1)


def test_keeps_a_label_that_closes_before_the_test_block_opens() -> None:
    label_spans = [(0, 19), (20, 30)]
    kept, purged, embargoed = purge_and_embargo([0], [[1]], label_spans, embargo=0)
    assert (kept, purged, embargoed) == ((0,), 0, 0)


def test_embargo_only_looks_forward() -> None:
    # Before the block: kept, however close. After it: dropped.
    label_spans = [(0, 19), (20, 30), (31, 40)]
    kept, _, embargoed = purge_and_embargo([0, 2], [[1]], label_spans, embargo=5)
    assert (kept, embargoed) == ((0,), 1)


def test_embargo_window_is_inclusive_at_its_edge_and_open_beyond_it() -> None:
    label_spans = [(20, 30), (35, 45), (36, 46)]
    kept, _, embargoed = purge_and_embargo([1, 2], [[0]], label_spans, embargo=5)
    assert (kept, embargoed) == ((2,), 1)


def test_non_adjacent_test_blocks_do_not_purge_the_middle() -> None:
    # The defect a single (min start, max end) hull would introduce: testing
    # the two ends of the sample would purge everything between them, which is
    # the training data.
    label_spans = spans(10)
    candidates = list(range(1, 9))
    kept, purged, _ = purge_and_embargo(candidates, [[0], [9]], label_spans, embargo=0)
    assert kept == tuple(candidates)
    assert purged == 0


def test_an_observation_removed_for_both_reasons_counts_once_as_purged() -> None:
    label_spans = [(20, 30), (25, 35)]
    _, purged, embargoed = purge_and_embargo([1], [[0]], label_spans, embargo=100)
    assert (purged, embargoed) == (1, 0)


def test_no_test_blocks_removes_nothing() -> None:
    assert purge_and_embargo([0, 1], [], spans(2), embargo=5) == ((0, 1), 0, 0)


def test_empty_test_blocks_are_ignored() -> None:
    assert purge_and_embargo([0, 1], [[], []], spans(2), embargo=5) == ((0, 1), 0, 0)


def test_negative_embargo_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        purge_and_embargo([0], [[1]], spans(2), embargo=-1)


# ── CombinatorialPurgedCV: configuration ─────────────────────


@pytest.mark.parametrize(
    ("n_groups", "n_test_groups", "message"),
    [
        (1, 1, "n_groups 1 must be at least 2"),
        (0, 1, "n_groups 0 must be at least 2"),
        (6, 0, "n_test_groups 0 must be at least 1"),
        (6, 6, "n_test_groups 6 must be at least 1"),
        (6, 7, "n_test_groups 7 must be at least 1"),
    ],
)
def test_rejects_group_counts_that_cannot_split(
    n_groups: int, n_test_groups: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        CombinatorialPurgedCV(n_groups=n_groups, n_test_groups=n_test_groups)


def test_rejects_negative_embargo() -> None:
    with pytest.raises(ValueError, match="embargo -1 must not be negative"):
        CombinatorialPurgedCV(embargo=-1)


@pytest.mark.parametrize(
    ("n_groups", "n_test_groups", "n_paths"),
    [(6, 2, 5), (10, 2, 9), (6, 1, 1), (8, 3, 21)],
)
def test_split_and_path_counts_match_the_combinatorics(
    n_groups: int, n_test_groups: int, n_paths: int
) -> None:
    cv = CombinatorialPurgedCV(n_groups=n_groups, n_test_groups=n_test_groups)
    assert cv.n_splits == comb(n_groups, n_test_groups)
    assert cv.n_paths == n_paths


def test_one_test_group_is_purged_k_fold_and_says_so() -> None:
    # A legitimate configuration, and not the gate §8 asks for. The flag is
    # what keeps a k-fold from being reported as CPCV.
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=1)
    assert (cv.is_combinatorial, cv.n_paths) == (False, 1)
    assert CombinatorialPurgedCV(n_groups=6, n_test_groups=2).is_combinatorial


# ── CombinatorialPurgedCV: grouping ──────────────────────────


def test_groups_are_contiguous_and_cover_every_observation() -> None:
    blocks = CombinatorialPurgedCV(n_groups=4).groups(10)
    assert [list(block) for block in blocks] == [[0, 1, 2], [3, 4, 5], [6, 7], [8, 9]]


def test_groups_are_never_shuffled() -> None:
    # Time order is the entire mechanism: a shuffle would scatter each block
    # across the sample and no purge could separate training from test.
    blocks = CombinatorialPurgedCV(n_groups=3).groups(9)
    flattened = [index for block in blocks for index in block]
    assert flattened == sorted(flattened)


def test_too_few_observations_to_fill_the_groups_is_rejected() -> None:
    with pytest.raises(ValueError, match="5 observations cannot fill 6 groups"):
        CombinatorialPurgedCV(n_groups=6).groups(5)


# ── CombinatorialPurgedCV: splitting ─────────────────────────


def test_yields_one_split_per_combination_in_combination_order() -> None:
    cv = CombinatorialPurgedCV(n_groups=4, n_test_groups=2)
    splits = list(cv.split(spans(8)))
    assert len(splits) == cv.n_splits
    assert [split.test_groups for split in splits] == [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
    ]


def test_train_and_test_never_intersect() -> None:
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, embargo=3)
    for split in cv.split(spans(60)):
        assert not set(split.train) & set(split.test)


def test_no_surviving_training_label_overlaps_any_test_label() -> None:
    # The property the whole module exists for, asserted directly rather than
    # through a count.
    label_spans = overlapping_spans(120, length=9)
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, embargo=4)
    for split in cv.split(label_spans):
        test_ranges = [label_spans[index] for index in split.test]
        for index in split.train:
            start, end = label_spans[index]
            assert not any(
                start <= test_end and end >= test_start for test_start, test_end in test_ranges
            )


def test_longer_labels_purge_more() -> None:
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2)
    short = sum(split.purged for split in cv.split(overlapping_spans(120, length=2)))
    long = sum(split.purged for split in cv.split(overlapping_spans(120, length=20)))
    assert long > short


def test_embargo_drops_strictly_more_than_no_embargo() -> None:
    label_spans = overlapping_spans(120, length=3)
    without = sum(split.dropped for split in CombinatorialPurgedCV().split(label_spans))
    with_embargo = sum(
        split.dropped for split in CombinatorialPurgedCV(embargo=10).split(label_spans)
    )
    assert with_embargo > without


def test_splitting_a_sample_smaller_than_the_groups_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot fill"):
        list(CombinatorialPurgedCV(n_groups=6).split(spans(3)))


def test_dropped_is_the_sum_of_both_reasons() -> None:
    split = Split(train=(), test=(), test_groups=(), purged=7, embargoed=3)
    assert split.dropped == 10


# ── Path assembly ────────────────────────────────────────────


@pytest.mark.parametrize(("n_groups", "n_test_groups"), [(4, 2), (6, 2), (6, 3), (8, 3)])
def test_every_path_covers_every_group_exactly_once(n_groups: int, n_test_groups: int) -> None:
    cv = CombinatorialPurgedCV(n_groups=n_groups, n_test_groups=n_test_groups)
    paths = cv.paths()
    assert len(paths) == cv.n_paths
    for path in paths:
        assert [group for _, group in path] == list(range(n_groups))


def test_every_split_group_slot_is_used_exactly_once_across_paths() -> None:
    # Each (split, group) pair is one block of predictions; none may be
    # dropped and none reused, or the paths are not real backtests.
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2)
    slots = [pair for path in cv.paths() for pair in path]
    assert len(slots) == len(set(slots)) == cv.n_splits * cv.n_test_groups


def test_paths_reference_splits_that_actually_test_that_group() -> None:
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2)
    test_groups = [split.test_groups for split in cv.split(spans(60))]
    for path in cv.paths():
        for split_index, group in path:
            assert group in test_groups[split_index]


# ── WalkForward ──────────────────────────────────────────────


def test_walk_forward_tests_each_block_after_the_first() -> None:
    wf = WalkForward(n_splits=4)
    assert [split.test_groups for split in wf.split(spans(50))] == [(1,), (2,), (3,), (4,)]


def test_walk_forward_training_is_always_earlier_than_its_test() -> None:
    label_spans = spans(60)
    for split in WalkForward(n_splits=5).split(label_spans):
        latest_train = max(label_spans[index][1] for index in split.train)
        earliest_test = min(label_spans[index][0] for index in split.test)
        assert latest_train < earliest_test


def test_expanding_training_grows_and_rolling_does_not() -> None:
    label_spans = spans(60)
    expanding = [len(split.train) for split in WalkForward(n_splits=5).split(label_spans)]
    rolling = [
        len(split.train) for split in WalkForward(n_splits=5, expanding=False).split(label_spans)
    ]
    assert expanding == sorted(expanding) and expanding[-1] > expanding[0]
    assert len(set(rolling)) == 1


def test_walk_forward_purges_an_overlapping_training_label() -> None:
    # "The training data came first" is about when positions opened. A label
    # that opened earlier but closed inside the test block still saw it.
    wf = WalkForward(n_splits=1)
    label_spans = [(0, 5), (6, 40), (20, 25), (26, 31)]
    split = next(iter(wf.split(label_spans)))
    assert split.test == (2, 3)
    assert (split.train, split.purged) == ((0,), 1)


@pytest.mark.parametrize(("n_splits", "embargo"), [(0, 0), (-1, 0), (3, -1)])
def test_walk_forward_rejects_impossible_configuration(n_splits: int, embargo: int) -> None:
    with pytest.raises(ValueError, match=r"must (be at least 1|not be negative)"):
        WalkForward(n_splits=n_splits, embargo=embargo)
