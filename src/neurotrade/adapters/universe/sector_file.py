"""`SectorMap` backed by a committed YAML file.

The counterpart to `universe_file.py`: the domain type lives in `core`, the
parsing of a hand-edited file lives here. Same strictness for the same reason —
a sector map silently short by one name would leave that name unresidualised,
and a cross-sectional ranking with one raw return in a column of residuals puts
that name at an extreme of the ranking every single day.

**Symbols are written `TICKER.VENUE`.** The universe file keys tickers under a
venue heading, which works because every name there has exactly one listing in
its section. A sector file cannot do that: TD is in XLF on both NYSE and TSX and
they are two instruments, so the venue has to travel with the ticker. `TD.NYSE`
and `TD.TSX` are two rows, deliberately.

**Dates are parsed by YAML, not by us.** `2023-03-17` unquoted is a `date` to
`yaml.safe_load`, so a malformed one arrives as a string and is rejected by
name. Writing it quoted would silently make it a string, which is why the
rejection says what it got.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Final

import yaml

from neurotrade.core.sectors import UNKNOWN_SINCE, SectorAssignment, SectorMap
from neurotrade.core.types import Symbol, Venue

__all__ = ["InvalidSectorFile", "SectorFile"]

_SUPPORTED_VERSION: Final = 1
"""The only file layout this understands, checked rather than assumed."""

_LISTING_VENUES: Final[dict[str, Venue]] = {
    member.value: member for member in Venue if member is not Venue.SMART
}
"""Venues a symbol may be listed on. SMART is IBKR's router, never a listing."""


class InvalidSectorFile(ValueError):
    """The file exists but does not describe a usable sector map.

    Distinct from `FileNotFoundError`, which means the path is wrong.
    """


class SectorFile:
    """A sector map read from YAML, parsed once at construction.

    Example:
        >>> import tempfile
        >>> body = '''
        ... version: 1
        ... markets: {NASDAQ: SPY.ARCA}
        ... assignments:
        ...   - {symbol: AAPL.NASDAQ, sector: XLK.ARCA}
        ... '''
        >>> with tempfile.TemporaryDirectory() as directory:
        ...     path = Path(directory) / "sectors.yaml"
        ...     _ = path.write_text(body)
        ...     sectors = SectorFile(path).sectors()
        ...     print(sectors.sector_of(Symbol("AAPL", Venue.NASDAQ), on=date(2024, 7, 8)))
        XLK.ARCA
    """

    def __init__(self, path: Path) -> None:
        """Read and validate a sector file.

        Args:
            path: The YAML file to read.

        Raises:
            FileNotFoundError: If the path does not exist.
            InvalidSectorFile: If the contents are malformed, name an unknown
                venue, assign an instrument to itself, or put one instrument in
                two sectors on the same day.
        """
        self._path = path
        self._sectors = _parse(path)

    @property
    def path(self) -> Path:
        """The file this was read from, for error messages and logs."""
        return self._path

    def sectors(self) -> SectorMap:
        """The map parsed at construction.

        Returns the same object every time, so a long run cannot see the file
        change underneath it — the guarantee `UniverseFile.universe` makes for
        the same reason.
        """
        return self._sectors

    def __repr__(self) -> str:
        return f"SectorFile(path={self._path}, assignments={len(self._sectors)})"


def _parse(path: Path) -> SectorMap:
    """Turn a YAML file into a validated `SectorMap`."""
    if not path.exists():
        raise FileNotFoundError(f"sector file not found: {path}")

    loaded = yaml.safe_load(path.read_text())
    if not isinstance(loaded, dict):
        raise InvalidSectorFile(f"{path} must contain a mapping, got {type(loaded).__name__}")

    version = loaded.get("version")
    if version != _SUPPORTED_VERSION:
        raise InvalidSectorFile(
            f"{path}: unsupported version {version!r}, expected {_SUPPORTED_VERSION}"
        )

    markets = loaded.get("markets")
    if not isinstance(markets, dict):
        raise InvalidSectorFile(
            f"{path}: 'markets' must map a venue to its index, got {type(markets).__name__}"
        )
    resolved = {
        _venue(path, name): _symbol(path, value, field="markets") for name, value in markets.items()
    }

    rows = loaded.get("assignments")
    if not isinstance(rows, list):
        raise InvalidSectorFile(f"{path}: 'assignments' must be a list, got {type(rows).__name__}")

    try:
        return SectorMap([_assignment(path, row) for row in rows], resolved)
    except ValueError as error:
        # `SectorMap` raises on a self-assignment and on two memberships that
        # overlap in time. Both are content errors in this file, so they are
        # re-raised as such rather than escaping as a bare ValueError from core.
        raise InvalidSectorFile(f"{path}: {error}") from error


def _assignment(path: Path, row: object) -> SectorAssignment:
    """One `assignments` entry."""
    if not isinstance(row, dict):
        raise InvalidSectorFile(f"{path}: each assignment must be a mapping, got {row!r}")
    unknown = set(row) - {"symbol", "sector", "from", "until"}
    if unknown:
        # Loudly, because a misspelled `form:` would silently become
        # UNKNOWN_SINCE and backdate a reclassification by a century.
        raise InvalidSectorFile(f"{path}: unknown key(s) {sorted(unknown)} in {row!r}")
    return SectorAssignment(
        symbol=_symbol(path, row.get("symbol"), field="symbol"),
        sector=_symbol(path, row.get("sector"), field="sector"),
        effective=_date(path, row.get("from")) or UNKNOWN_SINCE,
        until=_date(path, row.get("until")),
    )


def _symbol(path: Path, value: object, *, field: str) -> Symbol:
    """A `TICKER.VENUE` string as a `Symbol`."""
    if not isinstance(value, str):
        # `str(True)` is 'True', a plausible-looking ticker that would resolve
        # nowhere — the same trap `universe_file.py` documents for `ON`.
        raise InvalidSectorFile(
            f"{path}: {field} must be a quoted TICKER.VENUE string, got {value!r}"
        )
    ticker, _, venue = value.partition(".")
    if not ticker or not venue:
        raise InvalidSectorFile(f"{path}: {field} {value!r} is not TICKER.VENUE")
    return Symbol(ticker, _venue(path, venue))


def _venue(path: Path, name: object) -> Venue:
    """A venue name as a `Venue`."""
    if not isinstance(name, str) or name not in _LISTING_VENUES:
        raise InvalidSectorFile(
            f"{path}: unknown listing venue {name!r}; have {sorted(_LISTING_VENUES)}"
        )
    return _LISTING_VENUES[name]


def _date(path: Path, value: object) -> date | None:
    """An optional date. YAML parses a bare `2023-03-17` into a `date` already.

    A quoted `"2023-03-17"` arrives as a string and is rejected by name rather
    than parsed, because quoting it is the mistake and silently accepting both
    forms is how one of them stops being tested.
    """
    if value is None:
        return None
    if not isinstance(value, date):
        raise InvalidSectorFile(
            f"{path}: {value!r} is not a date — write it unquoted, as 2023-03-17"
        )
    return value
