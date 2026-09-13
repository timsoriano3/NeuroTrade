"""Tests for `UniverseFile`, the YAML-backed `UniversePort`.

Heavier on rejection than on happy paths: the module's whole job is refusing a
malformed file loudly at startup rather than letting it silently short the
universe by a venue or a symbol — see the module docstring. So most of this
file is bad input, one case at a time, each asserting both the exception type
and that the message names the offending venue or ticker.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from neurotrade.adapters.universe.universe_file import InvalidUniverseFile, UniverseFile
from neurotrade.core.types import Symbol, Venue

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "universe.yaml"
    path.write_text(text)
    return path


# ── Well-formed files ─────────────────────────────────────────


def test_parses_symbols_across_more_than_one_venue(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "version: 1\nvenues:\n  NASDAQ: [AAPL, MSFT]\n  TSX: [SHOP]\n",
    )
    universe = UniverseFile(path).universe()
    assert set(universe) == {
        Symbol("AAPL", Venue.NASDAQ),
        Symbol("MSFT", Venue.NASDAQ),
        Symbol("SHOP", Venue.TSX),
    }


def test_universe_returns_the_same_object_on_repeated_calls(tmp_path: Path) -> None:
    """`UniversePort` requires a stable answer within a run — see its docstring."""
    source = UniverseFile(_write(tmp_path, "version: 1\nvenues:\n  NYSE: [BA]\n"))
    assert source.universe() is source.universe()


def test_path_property_returns_the_constructor_argument(tmp_path: Path) -> None:
    path = _write(tmp_path, "version: 1\nvenues:\n  NYSE: [BA]\n")
    assert UniverseFile(path).path == path


def test_repr_shows_path_and_symbol_count(tmp_path: Path) -> None:
    path = _write(tmp_path, "version: 1\nvenues:\n  NASDAQ: [AAPL, MSFT]\n")
    assert repr(UniverseFile(path)) == f"UniverseFile(path={path}, symbols=2)"


# ── The committed universe.yaml ──────────────────────────────
#
# Regression guard for the shipped file. Deliberately asserts properties, not a
# fixed symbol count or digest: the file is expected to grow, and pinning
# either would make every future ticker addition break this test for no
# reason.


def test_committed_universe_file_loads_and_is_non_empty() -> None:
    universe = UniverseFile(_REPO_ROOT / "config" / "universe.yaml").universe()
    assert len(universe) > 0


def test_committed_universe_file_every_symbol_resolves_to_a_listing_venue() -> None:
    universe = UniverseFile(_REPO_ROOT / "config" / "universe.yaml").universe()
    for symbol in universe:
        assert symbol.venue is not Venue.SMART


def test_committed_universe_file_names_no_smart_venue_key() -> None:
    """SMART is an order route, never a listing key — belt and suspenders on
    top of the per-symbol check above, read straight from the raw YAML."""
    loaded = yaml.safe_load((_REPO_ROOT / "config" / "universe.yaml").read_text())
    assert "SMART" not in loaded["venues"]


# ── Missing or malformed file ─────────────────────────────────


def test_missing_path_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        UniverseFile(tmp_path / "does-not-exist.yaml")


def test_top_level_not_a_mapping_raises(tmp_path: Path) -> None:
    with pytest.raises(InvalidUniverseFile, match="must contain a mapping"):
        UniverseFile(_write(tmp_path, "- NASDAQ\n- NYSE\n"))


@pytest.mark.parametrize(
    "text",
    [
        "version: 2\nvenues:\n  NASDAQ: [AAPL]\n",
        "venues:\n  NASDAQ: [AAPL]\n",
    ],
    ids=["wrong-version", "missing-version"],
)
def test_wrong_or_missing_version_raises(tmp_path: Path, text: str) -> None:
    with pytest.raises(InvalidUniverseFile, match="unsupported version"):
        UniverseFile(_write(tmp_path, text))


@pytest.mark.parametrize(
    "text",
    [
        "version: 1\n",
        "version: 1\nvenues: [NASDAQ, NYSE]\n",
    ],
    ids=["missing-venues", "venues-not-a-mapping"],
)
def test_venues_missing_or_not_a_mapping_raises(tmp_path: Path, text: str) -> None:
    with pytest.raises(InvalidUniverseFile, match="'venues' must be a mapping"):
        UniverseFile(_write(tmp_path, text))


# ── Venue names ────────────────────────────────────────────────


def test_unknown_venue_raises_and_lists_the_valid_ones(tmp_path: Path) -> None:
    path = _write(tmp_path, "version: 1\nvenues:\n  NYSEE: [AAPL]\n")
    with pytest.raises(InvalidUniverseFile, match="unknown venue 'NYSEE'") as excinfo:
        UniverseFile(path)
    for venue in Venue:
        if venue is not Venue.SMART:
            assert venue.value in str(excinfo.value)


def test_smart_as_a_venue_key_raises_with_its_own_message(tmp_path: Path) -> None:
    """SMART is a real `Venue` member, so it must be caught before the
    unknown-venue branch and explained on its own terms rather than reported
    as merely unrecognised."""
    path = _write(tmp_path, "version: 1\nvenues:\n  SMART: [AAPL]\n")
    with pytest.raises(InvalidUniverseFile, match="order route, not a listing venue"):
        UniverseFile(path)


def test_venue_mapping_to_a_non_list_raises(tmp_path: Path) -> None:
    path = _write(tmp_path, "version: 1\nvenues:\n  NASDAQ: AAPL\n")
    with pytest.raises(InvalidUniverseFile, match="must map to a list of tickers"):
        UniverseFile(path)


# ── Ticker validation ─────────────────────────────────────────


def test_repeated_ticker_within_a_venue_raises_naming_venue_and_ticker(tmp_path: Path) -> None:
    path = _write(tmp_path, "version: 1\nvenues:\n  NASDAQ: [AAPL, AAPL]\n")
    with pytest.raises(InvalidUniverseFile, match="NASDAQ lists 'AAPL' twice"):
        UniverseFile(path)


@pytest.mark.parametrize(
    "ticker", ["aapl", " AAPL", "AAPL "], ids=["lowercase", "leading-ws", "trailing-ws"]
)
def test_malformed_ticker_raises_wrapping_symbols_own_rejection(
    tmp_path: Path, ticker: str
) -> None:
    """`Symbol.__post_init__` already rejects these; this only checks the
    rejection is wrapped as `InvalidUniverseFile` and names the venue."""
    path = _write(tmp_path, f"version: 1\nvenues:\n  NASDAQ: ['{ticker}']\n")
    with pytest.raises(InvalidUniverseFile, match="NASDAQ"):
        UniverseFile(path)


def test_file_describing_no_symbols_raises_invalid_universe_file_not_bare_value_error(
    tmp_path: Path,
) -> None:
    """`venues: {}` reaches `Universe([])`, which raises a bare `ValueError`.
    That must be caught and re-raised as `InvalidUniverseFile`, not leak through
    as the domain object's own exception type."""
    path = _write(tmp_path, "version: 1\nvenues: {}\n")
    with pytest.raises(InvalidUniverseFile, match="at least one symbol"):
        UniverseFile(path)


# ── The YAML boolean trap ─────────────────────────────────────
#
# Verified against the installed PyYAML before writing this: `safe_load`
# coerces an unquoted ON, OFF, NO, YES, TRUE and FALSE (any case) to `bool`,
# but leaves Y and N as strings. This is the highest-value test in the file —
# a silent `str(True)` here would crawl a ticker called "True" forever.


@pytest.mark.parametrize("ticker", ["ON", "NO", "OFF"])
def test_yaml_boolean_ticker_raises_telling_the_user_to_quote_it(
    tmp_path: Path, ticker: str
) -> None:
    path = _write(tmp_path, f"version: 1\nvenues:\n  NASDAQ: [{ticker}]\n")
    with pytest.raises(InvalidUniverseFile, match="quote it"):
        UniverseFile(path)


def test_quoting_the_boolean_looking_ticker_fixes_it(tmp_path: Path) -> None:
    """The escape hatch the error message points to actually works."""
    path = _write(tmp_path, "version: 1\nvenues:\n  NASDAQ: ['ON']\n")
    universe = UniverseFile(path).universe()
    assert Symbol("ON", Venue.NASDAQ) in universe
