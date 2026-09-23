"""Gate G3 itself: the lab rejects a snooped result and accepts a real one.

The gate is a fixture with a known answer, so the risk is that it becomes fitted
to the one seed it was developed on. These tests sweep several, and pin the
mechanism as well as the verdict — a run where the search found nothing worth
rejecting would satisfy "the overfit control was rejected" while proving nothing.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from neurotrade.lab.controls import CrossoverParams
from neurotrade.lab.gate import (
    DEFAULT_SEED,
    DSR_THRESHOLD,
    ControlOutcome,
    GateReport,
    run_exit_gate,
)

# Long enough for a 100-bar average to warm up and still leave ~300 labelled
# decisions; roughly a second per seed, against three for the 3,000-bar default.
_TEST_BARS = 1_500
_SEEDS = (DEFAULT_SEED, 1, 2, 3)


_REJECTED_SNOOP = ControlOutcome(
    control="overfit",
    family="control-snooped",
    n_variants=70,
    n_observations=294,
    best=CrossoverParams(fast=5, slow=40),
    best_sharpe=0.18,
    hurdle=0.35,
    deflated=0.002,
    pbo=0.3,
    path_sharpes=(0.05, 0.02),
)
"""Shaped like a real overfit-control result, so the verdict logic can be tested
without paying three seconds to generate one."""

_ACCEPTED_EDGE = replace(
    _REJECTED_SNOOP, control="honest", deflated=1.0, hurdle=0.05, best_sharpe=0.27
)
"""Shaped like a real honest-control result. A `GateReport` needs both halves,
and a test about one of them should not have to spell out the other."""


# ── The gate ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("seed", _SEEDS)
def test_the_gate_passes_on_every_seed_swept(seed: int) -> None:
    """A gate that only holds on its development seed is fitted to its fixture."""
    report = run_exit_gate(seed=seed, n_bars=_TEST_BARS)
    assert report.passed, report.failures


@pytest.mark.parametrize("seed", _SEEDS)
def test_the_snooped_search_found_something_tempting(seed: int) -> None:
    """The rejection means nothing unless there was something to reject.

    Seventy variants over a driftless walk have to produce a *positive* best
    Sharpe — that is the number a naive researcher would have reported — and
    the hurdle for a search that size has to stand above it.
    """
    overfit = run_exit_gate(seed=seed, n_bars=_TEST_BARS).overfit
    assert overfit.best_sharpe > 0
    assert overfit.hurdle > overfit.best_sharpe
    assert overfit.deflated < DSR_THRESHOLD


@pytest.mark.parametrize("seed", _SEEDS)
def test_the_planted_edge_survives_deflation(seed: int) -> None:
    """The direction nobody notices: a lab that rejects everything passes the §13 gate."""
    honest = run_exit_gate(seed=seed, n_bars=_TEST_BARS).honest
    assert honest.accepted
    assert honest.deflated >= DSR_THRESHOLD
    assert honest.hurdle < honest.best_sharpe


def test_the_honest_search_is_charged_a_smaller_hurdle() -> None:
    """Four a-priori variants must not cost what seventy snooped ones cost."""
    report = run_exit_gate(seed=DEFAULT_SEED, n_bars=_TEST_BARS)
    assert report.honest.n_variants < report.overfit.n_variants
    assert report.honest.hurdle < report.overfit.hurdle


def test_cpcv_produces_a_distribution_not_a_number() -> None:
    """Five paths from C(6,2) splits — the spread is the evidence, per §8."""
    report = run_exit_gate(seed=DEFAULT_SEED, n_bars=_TEST_BARS)
    assert len(report.overfit.path_sharpes) == 5
    assert len(report.honest.path_sharpes) == 5


def test_every_variant_is_recorded_not_only_the_winner() -> None:
    """Recording after a result looks good reproduces the bias the ledger removes."""
    report = run_exit_gate(seed=DEFAULT_SEED, n_bars=_TEST_BARS)
    assert report.overfit.n_variants == 70
    assert report.honest.n_variants == 4


def test_the_gate_leaves_no_trials_behind() -> None:
    """Seventy worthless crossovers in the real ledger would raise the hurdle forever.

    The ledger is append-only by design, so there would be no undoing it.
    """
    before = set(Path.cwd().rglob("ledger.jsonl"))
    run_exit_gate(seed=DEFAULT_SEED, n_bars=_TEST_BARS)
    assert set(Path.cwd().rglob("ledger.jsonl")) == before


# ── Reproducibility ──────────────────────────────────────────────────────


def test_two_runs_of_one_seed_agree_digest_for_digest() -> None:
    first = run_exit_gate(seed=DEFAULT_SEED, n_bars=_TEST_BARS)
    second = run_exit_gate(seed=DEFAULT_SEED, n_bars=_TEST_BARS)
    assert first.digest() == second.digest()


def test_a_different_seed_gives_a_different_digest() -> None:
    assert (
        run_exit_gate(seed=DEFAULT_SEED, n_bars=_TEST_BARS).digest()
        != run_exit_gate(seed=DEFAULT_SEED + 1, n_bars=_TEST_BARS).digest()
    )


# ── The verdict logic ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("deflated", "accepted"),
    [(0.0, False), (0.949, False), (DSR_THRESHOLD, True), (1.0, True)],
)
def test_acceptance_turns_on_the_deflated_sharpe(deflated: float, accepted: bool) -> None:
    assert replace(_REJECTED_SNOOP, deflated=deflated).accepted is accepted


@pytest.mark.parametrize("pbo", [0.0, 0.49, 0.5, 1.0])
def test_pbo_is_reported_but_does_not_vote(pbo: float) -> None:
    """Measured over ten seeds it does not separate the two controls; see `accepted`."""
    outcome = replace(_REJECTED_SNOOP, pbo=pbo, deflated=1.0)
    assert outcome.accepted
    assert outcome.is_overfit is (pbo >= 0.5)


def test_an_accepted_snoop_is_reported_as_a_failure() -> None:
    report = GateReport(
        seed=1,
        overfit=replace(_REJECTED_SNOOP, deflated=0.99),
        honest=_ACCEPTED_EDGE,
    )
    assert not report.passed
    assert "cannot tell a snooped result from an edge" in report.failures[0]


def test_a_search_that_found_nothing_does_not_count_as_a_rejection() -> None:
    """`accepted is False` is satisfied by a control nobody would have been fooled by."""
    report = GateReport(
        seed=1,
        overfit=replace(_REJECTED_SNOOP, best_sharpe=-0.05),
        honest=_ACCEPTED_EDGE,
    )
    assert not report.passed
    assert any("never found anything to be tempted by" in problem for problem in report.failures)


def test_a_hurdle_below_the_best_sharpe_is_reported_as_a_failure() -> None:
    """If deflation stops charging for the size of the search, nothing else catches it."""
    report = GateReport(
        seed=1,
        overfit=replace(_REJECTED_SNOOP, best_sharpe=0.40, hurdle=0.35, deflated=0.1),
        honest=_ACCEPTED_EDGE,
    )
    assert not report.passed
    assert any("not charging for the size of the search" in problem for problem in report.failures)


def test_a_rejected_edge_is_reported_as_a_failure() -> None:
    report = GateReport(
        seed=1,
        overfit=_REJECTED_SNOOP,
        honest=replace(
            _REJECTED_SNOOP, control="honest", deflated=0.4, hurdle=0.05, best_sharpe=0.27
        ),
    )
    assert not report.passed
    assert any("rejects real edges too" in problem for problem in report.failures)


def test_a_clean_report_lists_no_failures() -> None:
    report = GateReport(
        seed=1,
        overfit=_REJECTED_SNOOP,
        honest=_ACCEPTED_EDGE,
    )
    assert report.passed
    assert report.failures == ()


def test_an_outcome_prints_its_verdict() -> None:
    """Anything a command prints gets a `__str__`, or the terminal gets the repr."""
    rendered = str(_REJECTED_SNOOP)
    assert "REJECTED" in rendered
    assert "sma 5/40" in rendered
