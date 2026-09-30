"""The committed sector map, and every way the file can be wrong.

The real `config/sectors.yaml` is checked against the real universe here,
because a map short by one name leaves that name unresidualised — and a single
raw return in a column of residuals sits at an extreme of the ranking every day.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from neurotrade.adapters.universe.sector_file import InvalidSectorFile, SectorFile
from neurotrade.adapters.universe.universe_file import UniverseFile
from neurotrade.core.types import Symbol, Venue

SECTORS = Path("config/sectors.yaml")
UNIVERSE = Path("config/universe.yaml")

MINIMAL = """
version: 1
markets: {NASDAQ: SPY.ARCA}
assignments:
  - {symbol: AAPL.NASDAQ, sector: XLK.ARCA}
"""


def written(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "sectors.yaml"
    path.write_text(body)
    return path


# ── The committed file ──────────────────────────────────────────────────


def test_the_committed_file_parses() -> None:
    sectors = SectorFile(SECTORS).sectors()
    assert len(sectors) > 0
    assert sectors.markets[Venue.TSX] == Symbol("XIU", Venue.TSX)


def test_every_classified_instrument_is_in_the_universe() -> None:
    """A sector assignment for a name nobody crawls is a typo, not a plan."""
    sectors = SectorFile(SECTORS).sectors()
    universe = set(UniverseFile(UNIVERSE).universe().symbols)
    assert set(sectors.classified) <= universe


def test_the_only_unclassified_names_are_benchmarks() -> None:
    """Indices, sector ETFs and the volatility ETN are regressors, never members."""
    sectors = SectorFile(SECTORS).sectors()
    universe = set(UniverseFile(UNIVERSE).universe().symbols)
    unclassified = {symbol.ticker for symbol in universe - set(sectors.classified)}
    assert unclassified == {
        "DIA",
        "EEM",
        "IWM",
        "QQQ",
        "SPY",
        "VXX",
        "XIU",
        "XLB",
        "XLC",
        "XLE",
        "XLF",
        "XLI",
        "XLK",
        "XLP",
        "XLRE",
        "XLU",
        "XLV",
        "XLY",
    }


def test_the_payment_networks_change_sector_inside_the_seed_window() -> None:
    """GICS 2023, and the reason the file is dated at all.

    The seed window is 2022-09-30..2023-09-29, so a map without this would
    residualise V and MA against XLF for five months of it.
    """
    sectors = SectorFile(SECTORS).sectors()
    for ticker in ("V", "MA"):
        symbol = Symbol(ticker, Venue.NYSE)
        assert sectors.sector_of(symbol, on=date(2022, 10, 3)) == Symbol("XLK", Venue.ARCA)
        assert sectors.sector_of(symbol, on=date(2023, 9, 1)) == Symbol("XLF", Venue.ARCA)


def test_both_td_listings_are_classified_separately() -> None:
    """One company, two instruments, two currencies — and two rows."""
    sectors = SectorFile(SECTORS).sectors()
    xlf = Symbol("XLF", Venue.ARCA)
    assert sectors.sector_of(Symbol("TD", Venue.NYSE), on=date(2024, 7, 8)) == xlf
    assert sectors.sector_of(Symbol("TD", Venue.TSX), on=date(2024, 7, 8)) == xlf


def test_the_committed_map_is_parsed_once() -> None:
    source = SectorFile(SECTORS)
    assert source.sectors() is source.sectors()


# ── Rejections ──────────────────────────────────────────────────────────


def test_a_missing_file_is_a_deployment_problem_not_a_content_one() -> None:
    with pytest.raises(FileNotFoundError, match="sector file not found"):
        SectorFile(Path("config/no-such-sectors.yaml"))


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("- not a mapping\n", "must contain a mapping"),
        ("version: 2\nmarkets: {}\nassignments: []\n", "unsupported version"),
        ("version: 1\nassignments: []\n", "'markets' must map a venue"),
        ("version: 1\nmarkets: {}\n", "'assignments' must be a list"),
        (
            "version: 1\nmarkets: {MOON: SPY.ARCA}\nassignments: []\n",
            "unknown listing venue 'MOON'",
        ),
        (
            "version: 1\nmarkets: {SMART: SPY.ARCA}\nassignments: []\n",
            "unknown listing venue 'SMART'",
        ),
        (
            "version: 1\nmarkets: {}\nassignments: [{symbol: AAPL, sector: XLK.ARCA}]\n",
            "is not TICKER.VENUE",
        ),
        (
            "version: 1\nmarkets: {}\nassignments: [{symbol: XLK.ARCA, sector: XLK.ARCA}]\n",
            "cannot be its own sector",
        ),
        (
            "version: 1\nmarkets: {}\nassignments: [[AAPL.NASDAQ, XLK.ARCA]]\n",
            "each assignment must be a mapping",
        ),
    ],
)
def test_a_malformed_file_says_what_is_wrong(tmp_path: Path, body: str, message: str) -> None:
    with pytest.raises(InvalidSectorFile, match=message):
        SectorFile(written(tmp_path, body))


def test_a_misspelled_key_is_refused_rather_than_ignored(tmp_path: Path) -> None:
    """`form:` instead of `from:` would backdate a reclassification by a century."""
    body = (
        "version: 1\nmarkets: {}\n"
        "assignments: [{symbol: V.NYSE, sector: XLF.ARCA, form: 2023-03-17}]\n"
    )
    with pytest.raises(InvalidSectorFile, match=r"unknown key\(s\) \['form'\]"):
        SectorFile(written(tmp_path, body))


def test_a_quoted_date_is_rejected_rather_than_parsed(tmp_path: Path) -> None:
    """Quoting it is the mistake; accepting both forms is how one stops being tested."""
    body = (
        "version: 1\nmarkets: {}\n"
        'assignments: [{symbol: V.NYSE, sector: XLF.ARCA, from: "2023-03-17"}]\n'
    )
    with pytest.raises(InvalidSectorFile, match="is not a date"):
        SectorFile(written(tmp_path, body))


def test_an_unquoted_boolean_ticker_is_refused(tmp_path: Path) -> None:
    """`ON` is ON Semiconductor; unquoted, YAML makes it `True`."""
    body = "version: 1\nmarkets: {}\nassignments: [{symbol: ON, sector: XLK.ARCA}]\n"
    with pytest.raises(InvalidSectorFile, match=r"must be a quoted TICKER\.VENUE string"):
        SectorFile(written(tmp_path, body))


def test_overlapping_spans_in_a_file_are_a_content_error(tmp_path: Path) -> None:
    body = (
        "version: 1\nmarkets: {}\nassignments:\n"
        "  - {symbol: V.NYSE, sector: XLK.ARCA}\n"
        "  - {symbol: V.NYSE, sector: XLF.ARCA, from: 2023-03-17}\n"
    )
    with pytest.raises(InvalidSectorFile, match="is in two sectors at once"):
        SectorFile(written(tmp_path, body))


def test_a_minimal_file_is_enough(tmp_path: Path) -> None:
    sectors = SectorFile(written(tmp_path, MINIMAL)).sectors()
    assert sectors.sector_of(Symbol("AAPL", Venue.NASDAQ), on=date(2024, 7, 8)) == Symbol(
        "XLK", Venue.ARCA
    )
