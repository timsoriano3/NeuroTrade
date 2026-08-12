"""Configuration: profiles, environment overrides, and the settings object.

**Twelve-factor from day one** (§19). Nothing in the system hardcodes a path, a
host or a threshold. Everything comes from a profile file plus environment
overrides, which is what makes moving off this laptop onto the training rig — or
onto a VPS when §19's uptime trigger fires — a deployment change rather than a
port.

**Three profiles, matching the three environments in §10.1:**

- ``research`` — historical data, modelled fills, no capital.
- ``paper`` — live data, simulated fills against the real book, no capital.
- ``live`` — live data, real fills, real capital.

**Precedence: environment beats file.** Values are layered in this order, later
winning:

1. ``config/base.yaml`` — settings common to every profile
2. ``config/<profile>.yaml`` — what makes that profile different
3. ``.env``
4. real environment variables

That ordering is deliberate and is the twelve-factor rule: a container or a
systemd unit sets ``NEUROTRADE_DATA_ROOT`` and the file it inherited becomes
irrelevant, with no image rebuild. Nested values use a double underscore, so
``NEUROTRADE_IBKR__PORT=4002`` reaches ``settings.ibkr.port``.
"""

from __future__ import annotations

import hashlib
import json
import os
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

__all__ = [
    "Profile",
    "Settings",
    "config_hash",
    "load_settings",
]

ENVIRONMENTAL: dict[str, Any] = {"environmental": True}
"""Marks a field as describing *where* the system runs, not *what it decides*.

Attach with ``Field(json_schema_extra=ENVIRONMENTAL)``. Such fields are excluded
from the config hash — see `config_hash` for why that distinction matters.
Fields are behavioural by default, so a new setting affects the hash unless it
explicitly opts out."""

ENV_PREFIX = "NEUROTRADE_"

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = _REPO_ROOT / "config"
"""Where profile files live. Overridable so tests and containers can point
elsewhere without mutating the working tree."""


class Profile(StrEnum):
    """Which of §10.1's three environments this process is running as.

    Example:
        >>> Profile.RESEARCH.value
        'research'
    """

    RESEARCH = "research"  # historical data, modelled fills, no capital
    PAPER = "paper"  # live data, simulated fills, no capital
    LIVE = "live"  # live data, real fills, real capital


class StorageSettings(BaseModel):
    """Where the corpus lives on this machine.

    Split into raw and derived because §12.1 makes the distinction structural:
    raw is immutable and re-downloadable, derived is recomputable from raw and
    may be deleted at any time without loss.

    Example:
        >>> StorageSettings(data_root=Path("/tmp/x")).raw_dir
        PosixPath('/tmp/x/raw')
    """

    data_root: Path  # everything the corpus needs; the one path that moves between machines

    @property
    def raw_dir(self) -> Path:
        """Immutable source data. Never written twice, never edited in place."""
        return self.data_root / "raw"

    @property
    def derived_dir(self) -> Path:
        """Recomputable artifacts. Safe to delete; rebuilt from `raw_dir`."""
        return self.data_root / "derived"


class Settings(BaseSettings):
    """Fully resolved configuration for one process.

    Built by `load_settings`, not constructed directly in application code —
    there should be exactly one of these per process, created at startup and
    passed down, so that everything runs against the same resolved values.

    Example:
        >>> settings = load_settings(Profile.RESEARCH)
        >>> (settings.profile, settings.allow_live_orders)
        (<Profile.RESEARCH: 'research'>, False)
    """

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_nested_delimiter="__",
        env_file=".env",
        extra="forbid",  # a typo in a profile file is an error, not a silent no-op
        frozen=True,  # resolved config must not drift mid-run; the hash would lie
    )

    profile: Profile

    storage: StorageSettings = Field(json_schema_extra=ENVIRONMENTAL)
    """Where data lives on this machine. Environmental: moving the corpus to a
    different disk changes nothing about what the system decides to trade."""

    log_level: str = Field(default="INFO", json_schema_extra=ENVIRONMENTAL)
    """How loudly to log. Environmental: verbosity cannot change a decision."""

    allow_live_orders: bool = Field(default=False)
    """Whether this process may place orders that move real money.

    False in every profile except `live`. A structural guard rather than a
    preference: §6.2 requires that risk limits cannot be talked around, and the
    cheapest version of that is a research process being *unable* to trade even
    if a bug routes an order to a real broker adapter.
    """

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Rank the settings sources so the environment beats the profile files.

        Sources are consulted in the order returned, first match winning.
        Profile YAML arrives through `init_settings`, so ranking it *below* env
        and dotenv is what makes `NEUROTRADE_LOG_LEVEL=DEBUG` override a file
        without editing the file. Pydantic's default order puts `init_settings`
        first, which would invert this.
        """
        return (env_settings, dotenv_settings, init_settings, file_secret_settings)


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file into a dict, treating absent and empty as `{}`.

    Args:
        path: File to read. Missing files are not an error — only `base.yaml`
            is required, and that is checked by the caller.

    Returns:
        The parsed mapping.

    Raises:
        ValueError: If the file parses to something other than a mapping.
    """
    if not path.exists():
        return {}
    loaded = yaml.safe_load(path.read_text()) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a mapping, got {type(loaded).__name__}")
    return loaded


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Merge `overlay` onto `base`, recursing into nested mappings.

    A shallow merge would mean a profile setting one key of `storage` silently
    dropped every other key of it.

    Example:
        >>> _deep_merge({"a": {"x": 1, "y": 2}}, {"a": {"y": 3}})
        {'a': {'x': 1, 'y': 3}}
    """
    merged = dict(base)
    for key, value in overlay.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def load_settings(
    profile: Profile | str | None = None,
    *,
    config_dir: Path | None = None,
) -> Settings:
    """Resolve configuration for one profile.

    Args:
        profile: Which profile to load. Falls back to `NEUROTRADE_PROFILE`, then
            to `research` — the only default that cannot spend money.
        config_dir: Where the profile files live. Defaults to `config/` beside
            the repo root.

    Returns:
        A frozen `Settings` with file values overridden by the environment.

    Raises:
        FileNotFoundError: If `base.yaml` or the profile's file is missing. Both
            are required: a silent fallback to defaults is how a live process
            ends up running research settings.
        ValidationError: If a required value is absent or a key is unknown.

    Example:
        >>> load_settings("research").storage.raw_dir.name
        'raw'
    """
    resolved = Profile(profile or os.environ.get(f"{ENV_PREFIX}PROFILE") or Profile.RESEARCH)
    directory = config_dir or DEFAULT_CONFIG_DIR

    base_path = directory / "base.yaml"
    profile_path = directory / f"{resolved.value}.yaml"
    for path in (base_path, profile_path):
        if not path.exists():
            raise FileNotFoundError(f"config file not found: {path}")

    values = _deep_merge(_read_yaml(base_path), _read_yaml(profile_path))
    values["profile"] = resolved.value
    return Settings(**values)


def describe(settings: Settings) -> str:
    """Render settings for a human, one key per line.

    Used by `neurotrade config show` so that "what is this process actually
    running with" is answerable without attaching a debugger.

    Example:
        >>> print(describe(load_settings("research")).splitlines()[0])
        profile              research
    """
    lines = []
    for name, value in settings.model_dump().items():
        if isinstance(value, dict):
            for sub, sub_value in value.items():
                lines.append(f"{name}.{sub:<20} {sub_value}")
        else:
            lines.append(f"{name:<20} {value}")
    return "\n".join(lines)


def _canonical(value: object) -> object:
    """Reduce a value to something `json.dumps` renders identically everywhere.

    Paths become strings, enums become their values, sets become sorted lists.
    Without this, a `PosixPath` on this Mac and a `WindowsPath` on the rig would
    serialise differently and produce different hashes for identical settings.
    """
    if isinstance(value, dict):
        return {k: _canonical(v) for k, v in sorted(value.items())}
    if isinstance(value, list | tuple):
        return [_canonical(v) for v in value]
    if isinstance(value, set | frozenset):
        # key=repr because arbitrary objects are not mutually comparable, and an
        # unordered set would otherwise hash differently run to run.
        return sorted((_canonical(v) for v in value), key=repr)
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    return value


def behavioural_values(settings: Settings) -> dict[str, object]:
    """The subset of settings that can change what the system decides.

    Fields marked with `ENVIRONMENTAL` are omitted. Everything else is included,
    because the safe default for a new setting is to affect the hash.

    Example:
        >>> sorted(behavioural_values(load_settings("research")))
        ['allow_live_orders', 'profile']
    """
    dumped = settings.model_dump()
    return {
        name: _canonical(dumped[name])
        for name, field in type(settings).model_fields.items()
        if (field.json_schema_extra or {}) != ENVIRONMENTAL
    }


def config_hash(settings: Settings) -> str:
    """Stable fingerprint of the settings that affect trading decisions.

    §6.3 requires every order to carry this, so that a trade can be
    reconstructed months later and checked against the configuration that
    produced it. §10.3 requires it on every trade record for the same reason.

    **Environmental settings are excluded on purpose.** Hashing `data_root`
    would mean the same strategy running from an external NVMe produced a
    different hash than one running from the internal disk — so the journal
    would show a configuration change where none happened, and the hash would
    stop being a signal that something meaningful moved.

    Deterministic across processes and machines: canonical JSON with sorted
    keys, hashed with BLAKE2b. Python's `hash()` is salted per process and
    would give a different answer on every run.

    Args:
        settings: Resolved settings, normally from `load_settings`.

    Returns:
        A string like ``cfg_2f8a1c…``, 16 hex characters behind a prefix.

    Example:
        >>> config_hash(load_settings("research")) == config_hash(load_settings("research"))
        True
        >>> config_hash(load_settings("research")) != config_hash(load_settings("live"))
        True
    """
    payload = json.dumps(behavioural_values(settings), sort_keys=True, separators=(",", ":"))
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=32).hexdigest()[:16]
    return f"cfg_{digest}"
