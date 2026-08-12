"""Tests for configuration loading.

The precedence tests carry the weight: if the environment stops overriding the
profile files, §19's claim that moving machines is a deployment change quietly
becomes false, and nothing else would fail to reveal it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from neurotrade.config import (
    DEFAULT_CONFIG_DIR,
    Profile,
    Settings,
    describe,
    load_settings,
)


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """A minimal profile set, isolated from the repo's real one."""
    (tmp_path / "base.yaml").write_text(
        "storage:\n  data_root: data\nlog_level: INFO\nallow_live_orders: false\n"
    )
    (tmp_path / "research.yaml").write_text("log_level: DEBUG\n")
    (tmp_path / "live.yaml").write_text("allow_live_orders: true\n")
    return tmp_path


# ── Layering ─────────────────────────────────────────────────


def test_profile_file_overrides_base(config_dir: Path) -> None:
    assert load_settings(Profile.RESEARCH, config_dir=config_dir).log_level == "DEBUG"


def test_base_supplies_what_the_profile_omits(config_dir: Path) -> None:
    """live.yaml sets only allow_live_orders; storage still comes from base."""
    settings = load_settings(Profile.LIVE, config_dir=config_dir)
    assert settings.storage.data_root == Path("data")
    assert settings.allow_live_orders is True


def test_environment_beats_the_profile_file(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The twelve-factor rule, and the reason for the custom source ordering.

    research.yaml says DEBUG. A container setting the variable must win, with no
    image rebuild and no file edit.
    """
    monkeypatch.setenv("NEUROTRADE_LOG_LEVEL", "WARNING")
    assert load_settings(Profile.RESEARCH, config_dir=config_dir).log_level == "WARNING"


def test_environment_reaches_nested_values(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Double underscore descends: NEUROTRADE_STORAGE__DATA_ROOT -> storage.data_root."""
    monkeypatch.setenv("NEUROTRADE_STORAGE__DATA_ROOT", "/Volumes/nvme/neurotrade")
    settings = load_settings(Profile.RESEARCH, config_dir=config_dir)
    assert settings.storage.data_root == Path("/Volumes/nvme/neurotrade")
    assert settings.storage.raw_dir == Path("/Volumes/nvme/neurotrade/raw")


def test_profile_comes_from_the_environment_when_unspecified(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NEUROTRADE_PROFILE", "live")
    assert load_settings(config_dir=config_dir).profile is Profile.LIVE


def test_default_profile_is_research(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The only default that cannot spend money."""
    monkeypatch.delenv("NEUROTRADE_PROFILE", raising=False)
    settings = load_settings(config_dir=config_dir)
    assert settings.profile is Profile.RESEARCH
    assert settings.allow_live_orders is False


# ── The live-orders guard ────────────────────────────────────


@pytest.mark.parametrize("profile", [Profile.RESEARCH, Profile.PAPER])
def test_only_live_may_place_real_orders(profile: Profile) -> None:
    """Checked against the real config/ directory, not a fixture.

    A profile file that accidentally enabled this would be a defect no unit
    test on a temp directory could catch.
    """
    assert load_settings(profile).allow_live_orders is False


def test_live_profile_enables_real_orders() -> None:
    assert load_settings(Profile.LIVE).allow_live_orders is True


# ── Failure modes ────────────────────────────────────────────


def test_missing_base_file_is_an_error(tmp_path: Path) -> None:
    """Silently falling back to defaults is how a live process runs research settings."""
    (tmp_path / "research.yaml").write_text("log_level: DEBUG\n")
    with pytest.raises(FileNotFoundError, match=r"base\.yaml"):
        load_settings(Profile.RESEARCH, config_dir=tmp_path)


def test_missing_profile_file_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text("storage:\n  data_root: data\n")
    with pytest.raises(FileNotFoundError, match=r"paper\.yaml"):
        load_settings(Profile.PAPER, config_dir=tmp_path)


def test_unknown_key_is_rejected(config_dir: Path) -> None:
    """A typo in a profile file must fail loudly, not be silently ignored."""
    (config_dir / "research.yaml").write_text("log_levl: DEBUG\n")
    with pytest.raises(ValidationError, match="log_levl"):
        load_settings(Profile.RESEARCH, config_dir=config_dir)


def test_unknown_profile_is_rejected(config_dir: Path) -> None:
    with pytest.raises(ValueError, match="backtest"):
        load_settings("backtest", config_dir=config_dir)


def test_non_mapping_yaml_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text("- just\n- a\n- list\n")
    (tmp_path / "research.yaml").write_text("")
    with pytest.raises(ValueError, match="must contain a mapping"):
        load_settings(Profile.RESEARCH, config_dir=tmp_path)


# ── Immutability ─────────────────────────────────────────────


def test_settings_are_frozen(config_dir: Path) -> None:
    """Config must not drift mid-run — the config hash would stop being true."""
    settings = load_settings(Profile.RESEARCH, config_dir=config_dir)
    with pytest.raises(ValidationError):
        settings.log_level = "ERROR"  # type: ignore[misc]  # frozen, per the plugin


# ── Storage layout ───────────────────────────────────────────


def test_raw_and_derived_are_separate_trees(config_dir: Path) -> None:
    """§12.1 makes the split structural: raw immutable, derived recomputable."""
    storage = load_settings(Profile.RESEARCH, config_dir=config_dir).storage
    assert storage.raw_dir != storage.derived_dir
    assert storage.raw_dir.parent == storage.data_root


# ── The repo's own profiles ──────────────────────────────────


@pytest.mark.parametrize("profile", list(Profile))
def test_every_shipped_profile_loads(profile: Profile) -> None:
    assert load_settings(profile).profile is profile


def test_shipped_profiles_exist_for_every_enum_member() -> None:
    """A Profile with no file is a runtime failure waiting for a deploy."""
    for profile in Profile:
        assert (DEFAULT_CONFIG_DIR / f"{profile.value}.yaml").exists()


def test_paper_and_live_differ_only_where_they_must() -> None:
    """§10.1: paper exists to be evidence about live.

    If these drift apart, paper results stop transferring — so any new
    difference should be a deliberate decision, not an accident.
    """
    paper = load_settings(Profile.PAPER).model_dump()
    live = load_settings(Profile.LIVE).model_dump()
    differing = {k for k in paper if paper[k] != live[k]}
    assert differing == {"profile", "allow_live_orders"}


# ── Rendering ────────────────────────────────────────────────


def test_describe_renders_nested_values(config_dir: Path) -> None:
    rendered = describe(load_settings(Profile.RESEARCH, config_dir=config_dir))
    assert "profile" in rendered
    assert "storage.data_root" in rendered


def test_settings_cannot_be_built_without_required_values() -> None:
    with pytest.raises(ValidationError):
        Settings()
