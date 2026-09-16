"""Tests for `Universe`.

Heavier on rejection cases than on happy paths: the point of `Universe` is to
make "TD.NYSE and TD.TSX silently collapsed into one row" unrepresentable, so
what matters is that construction actually refuses to lose information, not
that the happy path returns symbols back.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import date

import pytest

from neurotrade.core.types import Symbol, Venue
from neurotrade.core.universe import Universe, UniverseHistory, UniverseMembership

AAPL = Symbol("AAPL", Venue.NASDAQ)
MSFT = Symbol("MSFT", Venue.NASDAQ)
TD_NYSE = Symbol("TD", Venue.NYSE)
TD_TSX = Symbol("TD", Venue.TSX)


# ── Construction ─────────────────────────────────────────────


def test_construction_sorts() -> None:
    assert Universe([MSFT, AAPL]).symbols == (AAPL, MSFT)


def test_construction_deduplicates() -> None:
    assert Universe([AAPL, AAPL, MSFT]).symbols == (AAPL, MSFT)


def test_order_of_input_does_not_affect_equality_or_digest() -> None:
    """Two universes built from the same symbols in different orders are the same universe."""
    forwards = Universe([AAPL, MSFT])
    backwards = Universe([MSFT, AAPL])
    assert forwards == backwards
    assert forwards.digest == backwards.digest


def test_same_ticker_on_two_venues_is_two_distinct_instruments() -> None:
    """TD on NYSE (USD) and TD on TSX (CAD) — the entire point of the module."""
    universe = Universe([TD_NYSE, TD_TSX])
    assert len(universe) == 2
    assert TD_NYSE in universe
    assert TD_TSX in universe


def test_empty_iterable_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one symbol"):
        Universe([])


def test_is_frozen() -> None:
    universe = Universe([AAPL])
    with pytest.raises(AttributeError):
        universe.symbols = (MSFT,)  # type: ignore[misc]


# ── venues ───────────────────────────────────────────────────


def test_venues_are_sorted_and_deduplicated() -> None:
    universe = Universe([TD_TSX, AAPL, MSFT])  # two NASDAQ names + one TSX
    assert universe.venues == (Venue.NASDAQ, Venue.TSX)


# ── by_venue ─────────────────────────────────────────────────


def test_by_venue_returns_universe_order() -> None:
    universe = Universe([MSFT, TD_NYSE, AAPL])
    assert universe.by_venue(Venue.NASDAQ) == (AAPL, MSFT)


def test_by_venue_is_empty_for_a_venue_with_no_listings() -> None:
    universe = Universe([AAPL, MSFT])
    assert universe.by_venue(Venue.TSX) == ()


# ── digest ───────────────────────────────────────────────────


def test_digest_is_16_hex_chars() -> None:
    digest = Universe([AAPL]).digest
    assert len(digest) == 16
    int(digest, 16)  # raises ValueError if not hex


def test_digest_changes_when_membership_changes() -> None:
    assert Universe([AAPL]).digest != Universe([AAPL, MSFT]).digest


def test_digest_distinguishes_the_same_ticker_on_different_venues() -> None:
    """A digest collision here would mean TD.NYSE and TD.TSX hashed as one."""
    assert Universe([TD_NYSE]).digest != Universe([TD_TSX]).digest


def test_digest_is_stable_across_processes_and_hash_seeds() -> None:
    """`hash()` is salted per process; BLAKE2b must not be affected.

    Runs the same construction in two subprocesses with different
    PYTHONHASHSEED values, mirroring `test_ids.py`. If this ever fails, a
    universe's digest stops identifying *what* was crawled — silently, and
    invisibly from inside a single process.
    """
    script = (
        "from neurotrade.core.universe import Universe, UniverseHistory, UniverseMembership;"
        "from neurotrade.core.types import Symbol, Venue;"
        "print(Universe([Symbol('MSFT', Venue.NASDAQ), Symbol('AAPL', Venue.NASDAQ),"
        " Symbol('TD', Venue.TSX)]).digest)"
    )
    outputs = {
        subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
            env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
        ).stdout.strip()
        for seed in ("0", "1", "random")
    }
    assert len(outputs) == 1, f"digest varied across hash seeds: {outputs}"


# ── container protocol ───────────────────────────────────────


def test_len_counts_symbols() -> None:
    assert len(Universe([AAPL, MSFT])) == 2


def test_iter_yields_universe_order() -> None:
    assert list(Universe([MSFT, AAPL])) == [AAPL, MSFT]


def test_contains_a_member() -> None:
    assert AAPL in Universe([AAPL, MSFT])


def test_contains_a_non_member() -> None:
    assert TD_TSX not in Universe([AAPL, MSFT])


def test_contains_a_non_symbol_object_returns_false() -> None:
    """`in` must not raise on a stray object; it must simply say no."""
    assert "AAPL" not in Universe([AAPL, MSFT])


# ── repr ─────────────────────────────────────────────────────


def test_repr_shape() -> None:
    universe = Universe([TD_NYSE, TD_TSX])
    assert repr(universe) == f"Universe(2 symbols, NYSE+TSX, digest={universe.digest})"


# ── UniverseMembership ───────────────────────────────────────

AAPL = Symbol("AAPL", Venue.NASDAQ)
MSFT = Symbol("MSFT", Venue.NASDAQ)
JULY_1 = date(2024, 7, 1)
JULY_2 = date(2024, 7, 2)


def test_a_membership_sorts_and_deduplicates_like_a_universe() -> None:
    membership = UniverseMembership(JULY_1, [MSFT, AAPL, AAPL])
    assert membership.symbols == (AAPL, MSFT)


def test_a_membership_may_be_empty() -> None:
    """Unlike `Universe`. A screen that admitted nobody is a real answer, and
    the row is the evidence the date was looked at."""
    assert len(UniverseMembership(JULY_1, [])) == 0


# ── UniverseHistory ──────────────────────────────────────────


def test_rows_are_ordered_by_date_however_they_arrive() -> None:
    history = UniverseHistory(
        [UniverseMembership(JULY_2, [MSFT]), UniverseMembership(JULY_1, [AAPL])],
        survivorship_biased=False,
    )
    assert history.dates == (JULY_1, JULY_2)


def test_two_rows_on_one_date_are_refused() -> None:
    """Which membership governs that session is not something to guess at."""
    with pytest.raises(ValueError, match="same session date"):
        UniverseHistory(
            [UniverseMembership(JULY_1, [AAPL]), UniverseMembership(JULY_1, [MSFT])],
            survivorship_biased=False,
        )


def test_membership_is_carried_forward_to_a_later_date() -> None:
    history = UniverseHistory([UniverseMembership(JULY_1, [AAPL])], survivorship_biased=False)
    assert history.as_of(date(2024, 12, 25)) == (AAPL,)


def test_asking_before_the_history_starts_raises() -> None:
    """Answering "nobody" would read as a screen that rejected everyone, and a
    backtest would quietly trade nothing instead of failing."""
    history = UniverseHistory([UniverseMembership(JULY_1, [AAPL])], survivorship_biased=False)
    with pytest.raises(LookupError, match="history starts 2024-07-01"):
        history.as_of(date(2024, 6, 30))


def test_the_bias_flag_changes_the_digest() -> None:
    """Two histories with identical membership are not the same artifact if one
    of them could only see survivors."""
    rows = [UniverseMembership(JULY_1, [AAPL])]
    assert (
        UniverseHistory(rows, survivorship_biased=True).digest
        != UniverseHistory(rows, survivorship_biased=False).digest
    )


def test_the_digest_is_stable_across_row_order() -> None:
    forwards = UniverseHistory(
        [UniverseMembership(JULY_1, [AAPL]), UniverseMembership(JULY_2, [MSFT])],
        survivorship_biased=False,
    )
    backwards = UniverseHistory(
        [UniverseMembership(JULY_2, [MSFT]), UniverseMembership(JULY_1, [AAPL])],
        survivorship_biased=False,
    )
    assert forwards.digest == backwards.digest


def test_repr_names_the_span_and_the_bias() -> None:
    history = UniverseHistory([UniverseMembership(JULY_1, [AAPL])], survivorship_biased=True)
    assert "survivorship-biased" in repr(history)
