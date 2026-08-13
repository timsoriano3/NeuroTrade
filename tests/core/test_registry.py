"""Tests for the versioned plugin registry.

Two properties carry weight: versions coexist (so champion and challenger can
run side by side), and the snapshot is deterministic (so it can feed the config
hash without making every run incomparable).
"""

from __future__ import annotations

import pytest

from neurotrade.core.registry import (
    DuplicateRegistration,
    Registry,
    RegistryFrozen,
    Version,
)


@pytest.fixture
def registry() -> Registry[str]:
    return Registry("strategy")


# ── Version ──────────────────────────────────────────────────


def test_version_parses_and_renders() -> None:
    assert str(Version.parse("1.2.3")) == "1.2.3"


@pytest.mark.parametrize("bad", ["1.0", "v1.0.0", "1.0.0-rc1", "1.0.0+build", "", "a.b.c"])
def test_malformed_versions_are_rejected(bad: str) -> None:
    """An ambiguous 'latest' is how a canary ends up running the wrong build."""
    with pytest.raises(ValueError, match=r"MAJOR\.MINOR\.PATCH"):
        Version.parse(bad)


def test_versions_order_numerically_not_lexically() -> None:
    """The classic bug: '1.10.0' sorts below '1.9.0' as a string."""
    assert Version.parse("1.10.0") > Version.parse("1.9.0")
    assert sorted([Version.parse("1.10.0"), Version.parse("1.9.0")])[0] == Version.parse("1.9.0")


def test_version_precedence_across_all_three_parts() -> None:
    assert Version.parse("2.0.0") > Version.parse("1.99.99")
    assert Version.parse("1.1.0") > Version.parse("1.0.99")


# ── Coexisting versions ──────────────────────────────────────


def test_two_versions_of_one_plugin_coexist(registry: Registry[str]) -> None:
    """§10.2 runs a challenger beside the champion on identical input.

    A registry holding one entry per name would evict the champion when the
    challenger registered, and the comparison would be against nothing.
    """
    registry.add("orb", "1.0.0", "champion")
    registry.add("orb", "1.1.0", "challenger")
    assert registry.get("orb", "1.0.0") == "champion"
    assert registry.get("orb", "1.1.0") == "challenger"
    assert len(registry) == 2


def test_lookup_without_a_version_returns_the_latest(registry: Registry[str]) -> None:
    registry.add("orb", "1.0.0", "old")
    registry.add("orb", "1.10.0", "newest")
    registry.add("orb", "1.9.0", "middle")
    assert registry.get("orb") == "newest"


def test_names_collapses_versions(registry: Registry[str]) -> None:
    registry.add("orb", "1.0.0", "a")
    registry.add("orb", "2.0.0", "b")
    registry.add("vwap", "1.0.0", "c")
    assert registry.names() == ("orb", "vwap")


# ── Registration guards ──────────────────────────────────────


def test_duplicate_registration_is_rejected(registry: Registry[str]) -> None:
    """Silent overwrite would make the journal name code that did not run."""
    registry.add("orb", "1.0.0", "first")
    with pytest.raises(DuplicateRegistration, match="already registered"):
        registry.add("orb", "1.0.0", "second")
    assert registry.get("orb", "1.0.0") == "first"


def test_empty_name_is_rejected(registry: Registry[str]) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        registry.add("   ", "1.0.0", "x")


def test_freeze_closes_registration(registry: Registry[str]) -> None:
    """Registering after the plugin hash is stamped would falsify the record."""
    registry.add("orb", "1.0.0", "a")
    registry.freeze()
    assert registry.frozen
    with pytest.raises(RegistryFrozen, match="frozen"):
        registry.add("vwap", "1.0.0", "b")


def test_freezing_does_not_disturb_lookups(registry: Registry[str]) -> None:
    registry.add("orb", "1.0.0", "a")
    registry.freeze()
    assert registry.get("orb") == "a"
    assert registry.snapshot() == ("orb@1.0.0",)


# ── Lookup failures ──────────────────────────────────────────


def test_unknown_name_raises(registry: Registry[str]) -> None:
    with pytest.raises(KeyError, match="no strategy registered"):
        registry.get("nope")


def test_unknown_version_raises_and_names_the_registry(registry: Registry[str]) -> None:
    """The error says which registry rejected it, saving a debugging round trip."""
    registry.add("orb", "1.0.0", "a")
    with pytest.raises(KeyError, match=r"strategy orb@2\.0\.0 is not registered"):
        registry.get("orb", "2.0.0")


def test_membership_is_by_name(registry: Registry[str]) -> None:
    registry.add("orb", "1.0.0", "a")
    assert "orb" in registry
    assert "vwap" not in registry


# ── Snapshot: what feeds the config hash ─────────────────────


def test_snapshot_is_sorted_regardless_of_registration_order() -> None:
    """An unordered snapshot would hash differently run to run."""
    forwards: Registry[int] = Registry("strategy")
    backwards: Registry[int] = Registry("strategy")
    entries = [("vwap", "2.0.0"), ("orb", "1.0.0"), ("orb", "1.10.0")]
    for name, version in entries:
        forwards.add(name, version, 1)
    for name, version in reversed(entries):
        backwards.add(name, version, 1)
    assert forwards.snapshot() == backwards.snapshot()


def test_snapshot_records_every_version(registry: Registry[str]) -> None:
    registry.add("orb", "1.0.0", "a")
    registry.add("orb", "1.1.0", "b")
    assert registry.snapshot() == ("orb@1.0.0", "orb@1.1.0")


def test_snapshot_sorts_versions_numerically() -> None:
    """Sorting the rendered strings would put 1.10.0 before 1.9.0."""
    registry: Registry[int] = Registry("strategy")
    registry.add("orb", "1.9.0", 1)
    registry.add("orb", "1.10.0", 2)
    assert registry.snapshot() == ("orb@1.9.0", "orb@1.10.0")


def test_snapshot_of_an_empty_registry_is_empty() -> None:
    assert Registry[int]("strategy").snapshot() == ()


def test_snapshot_changes_when_a_plugin_is_added(registry: Registry[str]) -> None:
    """The point: the audit record must reflect what was actually loaded."""
    registry.add("orb", "1.0.0", "a")
    before = registry.snapshot()
    registry.add("vwap", "1.0.0", "b")
    assert registry.snapshot() != before


# ── Decorator form ───────────────────────────────────────────


def test_decorator_registers_and_returns_the_object_unchanged() -> None:
    registry: Registry[type] = Registry("strategy")

    @registry.register("orb", "1.0.0")
    class Orb:
        pass

    assert registry.get("orb") is Orb
    assert Orb.__name__ == "Orb"  # still directly usable and importable


def test_decorator_rejects_duplicates() -> None:
    registry: Registry[type] = Registry("strategy")

    @registry.register("orb", "1.0.0")
    class First:
        pass

    with pytest.raises(DuplicateRegistration):

        @registry.register("orb", "1.0.0")
        class Second:
            pass

    assert registry.get("orb") is First


# ── Isolation ────────────────────────────────────────────────


def test_registries_do_not_share_state() -> None:
    """Not a global: two registries can hold the same name without colliding."""
    features: Registry[str] = Registry("feature")
    strategies: Registry[str] = Registry("strategy")
    features.add("vwap", "1.0.0", "the feature")
    strategies.add("vwap", "1.0.0", "the strategy")
    assert features.get("vwap") == "the feature"
    assert strategies.get("vwap") == "the strategy"


def test_iteration_is_deterministic(registry: Registry[str]) -> None:
    registry.add("vwap", "1.0.0", "b")
    registry.add("orb", "1.0.0", "a")
    assert [name for name, _, _ in registry] == ["orb", "vwap"]
