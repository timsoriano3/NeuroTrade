"""Loader for `config/seed_sources.yaml` — vendor file to `Symbol` mapping.

Neither `firstrate.py` nor `kibot.py` know which file backs which symbol; that
mapping is data, hand-written the same way `config/universe.yaml` is
(`adapters/universe/universe_file.py`), and this module turns it into value
objects the same way that one does.

**Every malformed file raises.** A typo here silently drops or mis-labels an
instrument, and nothing downstream could tell that apart from a vendor that
genuinely had no data for it. The file is small and read once at startup, so
strictness costs nothing.

**YAML coerces some real tickers.** See the gotcha in `universe_file.py`'s
module docstring — this loader rejects a non-string ticker the same way,
rather than calling `str()` on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

import yaml

from neurotrade.core.types import Symbol, Venue

__all__ = ["InvalidSeedSourcesFile", "SeedEntry", "SeedSource", "SeedSourcesFile"]

_SUPPORTED_VERSION: Final = 1
"""The only file layout this understands — see `universe_file.py` for why this
is checked rather than assumed."""

_LISTING_VENUES: Final[dict[str, Venue]] = {
    member.value: member for member in Venue if member is not Venue.SMART
}
"""Venue names accepted in the file. SMART excluded — see `universe_file.py`."""


class SeedSource(StrEnum):
    """Which free-sample vendor served a file.

    Example:
        >>> SeedSource.FIRSTRATE.value
        'firstrate'
    """

    FIRSTRATE = "firstrate"
    KIBOT = "kibot"


_SOURCES: Final[dict[str, SeedSource]] = {member.value: member for member in SeedSource}
"""Source names accepted in the file, keyed by their YAML value."""


@dataclass(frozen=True, slots=True)
class SeedEntry:
    """One vendor file, mapped to the instrument its rows describe.

    Example:
        >>> entry = SeedEntry(
        ...     source=SeedSource.KIBOT, file="IBM_unadjusted.txt", symbol=Symbol("IBM", Venue.NYSE)
        ... )
        >>> str(entry.symbol)
        'IBM.NYSE'
    """

    source: SeedSource  # which vendor served the file
    file: str  # the vendor's file name, as vendor_download.py is asked to keep it
    symbol: Symbol  # the instrument this file's rows describe


class InvalidSeedSourcesFile(ValueError):
    """The file exists but does not describe usable seed sources.

    Distinct from `FileNotFoundError`, which means the path is wrong — a
    deployment problem rather than a content one.
    """


class SeedSourcesFile:
    """Vendor file to `Symbol` entries, read from YAML once at construction.

    Parsing eagerly means a typo in the file breaks startup rather than a
    later fetch — the same trade `UniverseFile` makes, for the same reason.

    Example:
        >>> import tempfile
        >>> text = (
        ...     "version: 1\\nentries:\\n"
        ...     "  - source: firstrate\\n    ticker: AAPL\\n    venue: NASDAQ\\n"
        ...     "    file: AAPL_1min_sample_firstratedata.zip\\n"
        ... )
        >>> with tempfile.TemporaryDirectory() as directory:
        ...     path = Path(directory) / "seed_sources.yaml"
        ...     _ = path.write_text(text)
        ...     [entry.file for entry in SeedSourcesFile(path).entries()]
        ['AAPL_1min_sample_firstratedata.zip']
    """

    def __init__(self, path: Path) -> None:
        """Read and validate a seed sources file.

        Args:
            path: The YAML file to read.

        Raises:
            FileNotFoundError: If the path does not exist.
            InvalidSeedSourcesFile: If the contents are malformed, name an
                unknown source or venue, repeat a symbol, or describe no
                entries at all.
        """
        self._path = path
        self._entries = _parse(path)

    @property
    def path(self) -> Path:
        """The file this was read from, for error messages and logs."""
        return self._path

    def entries(self, source: SeedSource | None = None) -> tuple[SeedEntry, ...]:
        """The parsed entries, optionally restricted to one vendor.

        Args:
            source: Keep only this vendor's entries. Omitted returns all.

        Returns:
            Entries in file order.

        Example:
            >>> import tempfile
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     path = Path(directory) / "seed_sources.yaml"
            ...     _ = path.write_text(
            ...         "version: 1\\nentries:\\n  - source: kibot\\n"
            ...         "    ticker: IBM\\n    venue: NYSE\\n    file: IBM_unadjusted.txt\\n"
            ...     )
            ...     len(SeedSourcesFile(path).entries(SeedSource.KIBOT))
            1
        """
        if source is None:
            return self._entries
        return tuple(entry for entry in self._entries if entry.source is source)

    def __repr__(self) -> str:
        return f"SeedSourcesFile(path={self._path}, entries={len(self._entries)})"


def _parse(path: Path) -> tuple[SeedEntry, ...]:
    """Turn a YAML file into validated `SeedEntry` objects."""
    if not path.exists():
        raise FileNotFoundError(f"seed sources file not found: {path}")

    loaded = yaml.safe_load(path.read_text())
    if not isinstance(loaded, dict):
        raise InvalidSeedSourcesFile(f"{path} must contain a mapping, got {type(loaded).__name__}")

    version = loaded.get("version")
    if version != _SUPPORTED_VERSION:
        raise InvalidSeedSourcesFile(
            f"{path}: unsupported version {version!r}, expected {_SUPPORTED_VERSION}"
        )

    raw_entries = loaded.get("entries")
    if not isinstance(raw_entries, list):
        raise InvalidSeedSourcesFile(
            f"{path}: 'entries' must be a list, got {type(raw_entries).__name__}"
        )

    entries: list[SeedEntry] = []
    seen: set[Symbol] = set()
    for raw in raw_entries:
        entry = _entry(path, raw)
        if entry.symbol in seen:
            raise InvalidSeedSourcesFile(f"{path}: {entry.symbol} listed twice")
        seen.add(entry.symbol)
        entries.append(entry)

    if not entries:
        raise InvalidSeedSourcesFile(f"{path}: describes no entries")
    return tuple(entries)


def _entry(path: Path, raw: object) -> SeedEntry:
    """Validate one list item into a `SeedEntry`."""
    if not isinstance(raw, dict):
        raise InvalidSeedSourcesFile(f"{path}: entry must be a mapping, got {type(raw).__name__}")

    source = _source(path, raw.get("source"))
    venue = _venue(path, raw.get("venue"))
    ticker = _ticker(path, raw.get("ticker"))
    file_name = raw.get("file")
    if not isinstance(file_name, str) or not file_name:
        raise InvalidSeedSourcesFile(f"{path}: entry for {ticker!r} has no valid 'file' name")

    try:
        symbol = Symbol(ticker, venue)
    except ValueError as error:
        raise InvalidSeedSourcesFile(f"{path}: {venue.value} {ticker!r}: {error}") from error

    return SeedEntry(source=source, file=file_name, symbol=symbol)


def _source(path: Path, name: object) -> SeedSource:
    """Resolve a source name, rejecting anything but `SeedSource`'s values."""
    if isinstance(name, str) and name in _SOURCES:
        return _SOURCES[name]
    supported = ", ".join(_SOURCES)
    raise InvalidSeedSourcesFile(f"{path}: unknown source {name!r}; expected one of {supported}")


def _venue(path: Path, name: object) -> Venue:
    """Resolve a venue name — see `universe_file._venue` for why this takes
    `object` rather than `str`."""
    if name == Venue.SMART.value:
        raise InvalidSeedSourcesFile(f"{path}: SMART is an order route, not a listing venue")
    if not isinstance(name, str) or name not in _LISTING_VENUES:
        listed = ", ".join(_LISTING_VENUES)
        raise InvalidSeedSourcesFile(f"{path}: unknown venue {name!r}; expected one of {listed}")
    return _LISTING_VENUES[name]


def _ticker(path: Path, ticker: object) -> str:
    """Reject a non-string ticker rather than coercing it — see the module
    docstring's YAML-boolean gotcha."""
    if not isinstance(ticker, str):
        raise InvalidSeedSourcesFile(
            f"{path}: ticker {ticker!r} is a {type(ticker).__name__}, not a string — quote it "
            "(YAML reads ON, NO and OFF as booleans)"
        )
    return ticker
