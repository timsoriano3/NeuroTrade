"""The universe selector: which names are trading unusually heavily.

The floor is the design. Without it the top five of a quiet morning are "in
play" by definition, and the ORB strategy built on this would be measuring the
universe's ordering rather than unusual activity — which is exactly how the
published edge gets quoted for a universe that cannot produce it.
"""

from __future__ import annotations

import pytest

from neurotrade.core.types import Symbol, Venue
from neurotrade.features.cross_section import CrossSection, InstrumentSnapshot
from neurotrade.strategies.selection import (
    DEFAULT_SELECTION_FLOOR,
    DEFAULT_SELECTION_SIZE,
    stocks_in_play,
)

NAMES = [Symbol(ticker, Venue.NASDAQ) for ticker in ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")]


def section(*rvols: float | None) -> CrossSection:
    return CrossSection(
        as_of=0,
        rows={
            symbol: InstrumentSnapshot(symbol, 0.01, None, rvol, None)
            for symbol, rvol in zip(NAMES, rvols, strict=False)
        },
    )


def test_the_heaviest_names_come_back_heaviest_first() -> None:
    chosen = stocks_in_play(section(3.0, 5.0, 4.0), size=2)
    assert [symbol.ticker for symbol in chosen] == ["BBB", "CCC"]


def test_a_quiet_morning_selects_nothing() -> None:
    """The finding, not a failure — and the reason the floor exists."""
    assert stocks_in_play(section(1.1, 1.2, 0.9)) == ()


def test_the_floor_excludes_rather_than_the_slice() -> None:
    """A name under the floor must not displace a qualifying one out of the top `size`.

    Ranking first and slicing after would let a 1.1x name take a seat from a
    2.5x name whenever it happened to sort higher.
    """
    chosen = stocks_in_play(section(1.5, 9.0, 1.9, 2.5), size=2)
    assert [symbol.ticker for symbol in chosen] == ["BBB", "DDD"]


def test_a_cold_relative_volume_is_excluded_not_ranked_low() -> None:
    assert [s.ticker for s in stocks_in_play(section(None, 3.0))] == ["BBB"]


def test_the_selection_is_capped_at_size() -> None:
    chosen = stocks_in_play(section(3.0, 4.0, 5.0, 6.0, 7.0, 8.0), size=3)
    assert len(chosen) == 3


def test_ties_break_on_the_symbol_so_two_runs_agree() -> None:
    assert [s.ticker for s in stocks_in_play(section(3.0, 3.0), size=1)] == ["AAA"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"size": 0}, "size 0 must be at least 1"),
        ({"floor": 0.0}, r"floor 0\.0 must be positive"),
    ],
)
def test_impossible_settings_are_refused(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        stocks_in_play(section(3.0), **kwargs)  # type: ignore[arg-type]


def test_the_shipped_defaults_are_the_ones_the_strategies_share() -> None:
    """One implementation for the nightly selector and the ORB filter (§3.6)."""
    assert DEFAULT_SELECTION_SIZE == 5
    assert DEFAULT_SELECTION_FLOOR == 2.0
