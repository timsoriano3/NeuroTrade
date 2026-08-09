"""Smoke tests for the package itself.

Cheap, but they fail loudly if the packaging or the toolchain is broken — which
is exactly the failure that is otherwise discovered three commits later.
"""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

import neurotrade


def test_package_imports() -> None:
    assert neurotrade.__version__


def test_module_version_matches_installed_metadata() -> None:
    """Catches drift between __version__ and the version pyproject declares."""
    assert neurotrade.__version__ == version("neurotrade")


def test_package_is_installed_editable() -> None:
    """The install must point at the working tree, not a copy in site-packages.

    If this ever fails, tests are passing against stale code.
    """
    module = Path(neurotrade.__file__ or "")
    assert module.is_relative_to(Path.cwd() / "src")
