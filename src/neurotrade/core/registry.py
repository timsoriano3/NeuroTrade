"""A versioned registry for plugins.

§3.7 makes everything a versioned plugin: adding or changing a strategy is a new
file plus config, never a change to the core. This is the mechanism.

**Versions are part of the key, not metadata on the value.** Two versions of one
strategy must coexist in a running process, because §10.2's promotion flow runs
a challenger beside the champion on identical input. If a registry held one entry
per name, registering the challenger would evict the champion and the comparison
would be against nothing.

**The snapshot feeds the config hash.** §4.2 requires "which plugins were
loaded" to be part of the audit record, so that a trade months old can be tied
not just to a configuration but to the exact set of strategies that were running
when it happened. The snapshot is deterministically ordered for that reason —
an unordered one would hash differently run to run.

**Registration closes.** `freeze` makes the registry immutable, and startup
calls it once plugin discovery is done. Registering a strategy afterwards would
change the plugin set *after* its hash had already been stamped onto orders,
which would make the audit record quietly false.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Self

__all__ = [
    "DuplicateRegistration",
    "Registry",
    "RegistryFrozen",
    "Version",
]

_SEMVER = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


class DuplicateRegistration(ValueError):
    """Raised when a name and version are registered twice.

    Overwriting silently would mean the journal names a strategy that is not the
    one that ran — the version recorded on a trade would point at code that had
    been replaced in memory.
    """


class RegistryFrozen(RuntimeError):
    """Raised when registering into a registry that has been frozen."""


@dataclass(frozen=True, slots=True, order=True)
class Version:
    """A semantic version, ordered so `latest` is well defined.

    Strict `MAJOR.MINOR.PATCH` only. Pre-release and build suffixes are rejected
    because they make ordering ambiguous, and an ambiguous "latest" is how a
    canary sleeve ends up running the wrong build.

    Example:
        >>> Version.parse("1.2.10") > Version.parse("1.2.9")
        True
    """

    major: int  # breaking change to the plugin's contract or its meaning
    minor: int  # new behaviour, backwards compatible
    patch: int  # fix with no intended behaviour change

    @classmethod
    def parse(cls, text: str) -> Self:
        """Parse a `MAJOR.MINOR.PATCH` string.

        Args:
            text: e.g. `"1.0.0"`.

        Returns:
            The parsed version.

        Raises:
            ValueError: If the string is not exactly three dot-separated
                integers. `"1.0"`, `"v1.0.0"` and `"1.0.0-rc1"` are all refused.

        Example:
            >>> Version.parse("2.3.1")
            Version(major=2, minor=3, patch=1)
        """
        match = _SEMVER.match(text)
        if match is None:
            raise ValueError(f"not a MAJOR.MINOR.PATCH version: {text!r}")
        major, minor, patch = (int(part) for part in match.groups())
        return cls(major=major, minor=minor, patch=patch)

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


@dataclass(frozen=True, slots=True, order=True)
class _Key:
    """Registry key. Ordered by name then version so snapshots are stable."""

    name: str
    version: Version


class Registry[T]:
    """A name-and-version keyed collection of plugins.

    Not a global. Each layer owns its own instance — features, strategies,
    models — so tests can build an isolated one and two registries cannot
    collide over a shared name.

    Example:
        >>> registry: Registry[str] = Registry("greeting")
        >>> registry.add("hello", "1.0.0", "hi")
        >>> registry.get("hello")
        'hi'
    """

    __slots__ = ("_frozen", "_items", "_kind")

    def __init__(self, kind: str) -> None:
        """Create an empty registry.

        Args:
            kind: What this registry holds, e.g. `"strategy"`. Used only in
                error messages, where knowing which registry rejected a lookup
                saves a debugging round trip.
        """
        self._kind = kind
        self._items: dict[_Key, T] = {}
        self._frozen = False

    # ── Registration ─────────────────────────────────────────

    def add(self, name: str, version: str | Version, item: T) -> None:
        """Register an item under a name and version.

        Args:
            name: Plugin name, e.g. `"orb_stocks_in_play"`.
            version: `MAJOR.MINOR.PATCH`, as a string or `Version`.
            item: The thing being registered.

        Raises:
            DuplicateRegistration: If that name and version are already taken.
            RegistryFrozen: If registration has been closed.
            ValueError: If the name is empty or the version is malformed.

        Example:
            >>> registry: Registry[int] = Registry("answer")
            >>> registry.add("meaning", "1.0.0", 42)
            >>> registry.get("meaning")
            42
        """
        if self._frozen:
            raise RegistryFrozen(
                f"{self._kind} registry is frozen; {name} cannot be registered after startup"
            )
        if not name.strip():
            raise ValueError(f"{self._kind} name must not be empty")

        key = _Key(name, version if isinstance(version, Version) else Version.parse(version))
        if key in self._items:
            raise DuplicateRegistration(f"{self._kind} {name}@{key.version} is already registered")
        self._items[key] = item

    def register(self, name: str, version: str | Version) -> Callable[[T], T]:
        """Decorator form of `add`, for registration at import time.

        Args:
            name: Plugin name.
            version: `MAJOR.MINOR.PATCH`.

        Returns:
            A decorator that registers and then returns its argument unchanged,
            so the decorated object stays directly usable and importable.

        Example:
            >>> registry: Registry[type] = Registry("strategy")
            >>> @registry.register("orb", "1.0.0")
            ... class Orb: pass
            >>> registry.get("orb") is Orb
            True
        """

        def decorate(item: T) -> T:
            self.add(name, version, item)
            return item

        return decorate

    def freeze(self) -> None:
        """Close registration.

        Called once at startup, after plugin discovery. Anything registered
        afterwards would not be reflected in the plugin hash already stamped on
        this run's records, so `add` raises `RegistryFrozen` from here on.

        Example:
            >>> registry: Registry[int] = Registry("answer")
            >>> registry.freeze()
            >>> registry.frozen
            True
        """
        self._frozen = True

    @property
    def frozen(self) -> bool:
        """Whether registration has been closed."""
        return self._frozen

    # ── Lookup ───────────────────────────────────────────────

    def get(self, name: str, version: str | Version | None = None) -> T:
        """Look up one item.

        Args:
            name: Plugin name.
            version: Exact version. When omitted, the highest registered version
                is returned — convenient interactively, but production config
                should pin a version so a new registration cannot silently
                change what runs.

        Returns:
            The registered item.

        Raises:
            KeyError: If the name, or that specific version, is not registered.

        Example:
            >>> registry: Registry[str] = Registry("strategy")
            >>> registry.add("orb", "1.0.0", "old")
            >>> registry.add("orb", "1.1.0", "new")
            >>> (registry.get("orb"), registry.get("orb", "1.0.0"))
            ('new', 'old')
        """
        if version is None:
            return self._items[_Key(name, self.latest_version(name))]

        key = _Key(name, version if isinstance(version, Version) else Version.parse(version))
        try:
            return self._items[key]
        except KeyError:
            raise KeyError(f"{self._kind} {name}@{key.version} is not registered") from None

    def latest_version(self, name: str) -> Version:
        """Highest registered version of a name.

        Raises:
            KeyError: If nothing is registered under that name.

        Example:
            >>> registry: Registry[str] = Registry("strategy")
            >>> registry.add("orb", "1.9.0", "a")
            >>> registry.add("orb", "1.10.0", "b")
            >>> str(registry.latest_version("orb"))     # numeric, not lexical
            '1.10.0'
        """
        versions = [key.version for key in self._items if key.name == name]
        if not versions:
            raise KeyError(f"no {self._kind} registered under {name!r}")
        return max(versions)

    def names(self) -> tuple[str, ...]:
        """Every registered name, sorted.

        Example:
            >>> registry: Registry[int] = Registry("n")
            >>> registry.add("b", "1.0.0", 1); registry.add("a", "1.0.0", 2)
            >>> registry.names()
            ('a', 'b')
        """
        return tuple(sorted({key.name for key in self._items}))

    def snapshot(self) -> tuple[str, ...]:
        """Deterministic record of everything registered, as `name@version`.

        Folded into the config hash so that "which plugins were loaded" is part
        of the audit record (§4.2). Sorted, because an unordered snapshot would
        hash differently between runs and make every comparison meaningless.

        Example:
            >>> registry: Registry[int] = Registry("strategy")
            >>> registry.add("vwap", "2.0.0", 1); registry.add("orb", "1.0.0", 2)
            >>> registry.snapshot()
            ('orb@1.0.0', 'vwap@2.0.0')
        """
        return tuple(f"{key.name}@{key.version}" for key in sorted(self._items))

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and any(key.name == name for key in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[tuple[str, Version, T]]:
        """Iterate registrations in deterministic order."""
        for key in sorted(self._items):
            yield key.name, key.version, self._items[key]

    def __repr__(self) -> str:
        return f"Registry({self._kind!r}, {len(self._items)} registered)"
