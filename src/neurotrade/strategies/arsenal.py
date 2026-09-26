"""The registry every strategy plugin registers into.

Module-level rather than global, for the same reason `features.indicators`
keeps its own: a caller that wants a different arsenal — a test, or a discovery
run exploring variants — builds its own `StrategyRegistry` instead of mutating a
shared one. Nothing here reaches for a singleton.

Importing this module is not enough to populate it. A plugin registers when its
own module is imported, so a host that wants the whole arsenal imports the
package; one that wants a single strategy imports that strategy and gets only
it. That is deliberate — §10.2 runs a champion beside a challenger, and which
strategies exist in a given run is configuration, not a property of the code.
"""

from __future__ import annotations

from typing import Final

from neurotrade.strategies.base import StrategyRegistry

__all__ = ["arsenal"]

arsenal: Final = StrategyRegistry()
"""Every strategy plugin, by name and version."""
