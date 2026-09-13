"""`UniversePort` backed by a committed YAML file.

The crawler's work queue has two axes: which days a venue traded, and which
instruments to ask about. `adapters/calendar` supplies the first. This supplies
the second, and for Phase 0 it is simply a list someone wrote down — §12.1
stage 1 starts the crawl in week 1, while the yfinance-derived universe history
that would otherwise populate it is stage 3.

**Every malformed file raises.** A universe silently short by one venue would
produce a corpus silently short by one venue, and nothing downstream could tell
that apart from a venue that genuinely had no data. The file is small, hand
edited, and read once at startup, so strictness costs nothing and catches typos
at the only moment anyone is looking.

**YAML coerces some real tickers.** `ON`, `NO` and `OFF` parse as booleans, and
`ON` is ON Semiconductor on NASDAQ. This module rejects a non-string ticker and
says to quote it rather than calling `str()` on it, because `str(True)` is
`'True'` — a plausible-looking ticker that would be crawled forever and never
resolve.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import yaml

from neurotrade.core.types import Symbol, Venue
from neurotrade.core.universe import Universe

__all__ = ["InvalidUniverseFile", "UniverseFile"]

_SUPPORTED_VERSION: Final = 1
"""The only file layout this understands. Checked rather than assumed so that a
future layout change fails loudly on an old file instead of reading half of it."""

_LISTING_VENUES: Final[dict[str, Venue]] = {
    member.value: member for member in Venue if member is not Venue.SMART
}
"""Venue names accepted as keys in the file. SMART is excluded because it is
IBKR's order router — a destination, never a listing — and `Symbol` rejects it
anyway; naming it here lets the error say why rather than "unknown venue"."""


class InvalidUniverseFile(ValueError):
    """The file exists but does not describe a usable universe.

    Distinct from `FileNotFoundError`, which means the path is wrong — a
    deployment problem rather than a content one.
    """


class UniverseFile:
    """A universe read from YAML, parsed once at construction.

    Satisfies `UniversePort` structurally. Parsing eagerly means a typo in the
    file breaks startup rather than the first fetch, which is the difference
    between a clear error and a crawl that stops an hour in.

    Example:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as directory:
        ...     path = Path(directory) / "universe.yaml"
        ...     _ = path.write_text("version: 1\\nvenues:\\n  NASDAQ: [AAPL, MSFT]\\n")
        ...     [str(symbol) for symbol in UniverseFile(path).universe()]
        ['AAPL.NASDAQ', 'MSFT.NASDAQ']
    """

    def __init__(self, path: Path) -> None:
        """Read and validate a universe file.

        Args:
            path: The YAML file to read.

        Raises:
            FileNotFoundError: If the path does not exist.
            InvalidUniverseFile: If the contents are malformed, name an unknown
                venue, repeat a ticker, or describe no symbols at all.
        """
        self._path = path
        self._universe = _parse(path)

    @property
    def path(self) -> Path:
        """The file this was read from, for error messages and logs."""
        return self._path

    def universe(self) -> Universe:
        """The instruments in scope.

        Returns the universe parsed at construction, so repeated calls cannot
        disagree even if the file changes underneath a running crawl — which
        `UniversePort` requires.

        Example:
            >>> import tempfile
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     path = Path(directory) / "universe.yaml"
            ...     _ = path.write_text("version: 1\\nvenues:\\n  TSX: [SHOP]\\n")
            ...     source = UniverseFile(path)
            ...     source.universe() is source.universe()
            True
        """
        return self._universe

    def __repr__(self) -> str:
        return f"UniverseFile(path={self._path}, symbols={len(self._universe)})"


def _parse(path: Path) -> Universe:
    """Turn a YAML file into a validated `Universe`."""
    if not path.exists():
        raise FileNotFoundError(f"universe file not found: {path}")

    loaded = yaml.safe_load(path.read_text())
    if not isinstance(loaded, dict):
        raise InvalidUniverseFile(f"{path} must contain a mapping, got {type(loaded).__name__}")

    version = loaded.get("version")
    if version != _SUPPORTED_VERSION:
        raise InvalidUniverseFile(
            f"{path}: unsupported version {version!r}, expected {_SUPPORTED_VERSION}"
        )

    venues = loaded.get("venues")
    if not isinstance(venues, dict):
        raise InvalidUniverseFile(
            f"{path}: 'venues' must be a mapping of venue to tickers, got {type(venues).__name__}"
        )

    symbols: list[Symbol] = []
    for name, tickers in venues.items():
        symbols.extend(_venue_symbols(path, _venue(path, name), tickers))

    try:
        return Universe(symbols)
    except ValueError as error:
        # The only way to get here is an empty file: every other rejection is
        # raised above with the offending venue or ticker named.
        raise InvalidUniverseFile(f"{path}: {error}") from error


def _venue(path: Path, name: object) -> Venue:
    """Resolve a venue key, rejecting unknown names and the order router.

    Takes `object` rather than `str` because YAML decides the type of a key on
    its own: an unquoted `ON` arrives as the boolean true, and an unquoted date
    as a `date`. Both are wrong here, and both must say so rather than crash.
    """
    if name == Venue.SMART.value:
        raise InvalidUniverseFile(f"{path}: SMART is an order route, not a listing venue")
    if not isinstance(name, str) or name not in _LISTING_VENUES:
        listed = ", ".join(_LISTING_VENUES)
        raise InvalidUniverseFile(f"{path}: unknown venue {name!r}; expected one of {listed}")
    return _LISTING_VENUES[name]


def _venue_symbols(path: Path, venue: Venue, tickers: object) -> list[Symbol]:
    """Validate one venue's ticker list into symbols.

    Duplicates are rejected here even though `Universe` would collapse them.
    The domain object has set semantics and is right to; a hand-edited file
    listing a name twice is a typo, and naming it with its venue is the
    difference between a two-second fix and a diff review.
    """
    if not isinstance(tickers, list):
        raise InvalidUniverseFile(
            f"{path}: {venue.value} must map to a list of tickers, got {type(tickers).__name__}"
        )

    seen: set[str] = set()
    symbols: list[Symbol] = []
    for ticker in tickers:
        if not isinstance(ticker, str):
            raise InvalidUniverseFile(
                f"{path}: {venue.value} ticker {ticker!r} is a "
                f"{type(ticker).__name__}, not a string — quote it "
                "(YAML reads ON, NO and OFF as booleans)"
            )
        if ticker in seen:
            raise InvalidUniverseFile(f"{path}: {venue.value} lists {ticker!r} twice")
        seen.add(ticker)
        try:
            symbols.append(Symbol(ticker, venue))
        except ValueError as error:
            raise InvalidUniverseFile(f"{path}: {venue.value} {ticker!r}: {error}") from error
    return symbols
