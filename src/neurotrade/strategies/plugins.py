"""Every strategy this build ships, imported so the arsenal holds them.

`arsenal.py` on its own is an empty registry, deliberately: a plugin registers
when its own module is imported, so a run that wants one strategy imports that
strategy and gets only it. This module is the other half of that arrangement —
the explicit "give me everything shipped" import, for a host or a command that
has to resolve a strategy by name rather than by import.

**Adding a strategy is a new file and one line here.** Nothing else: not the
engine, not the lab, not `core`.
"""

from __future__ import annotations

from neurotrade.strategies.arsenal import arsenal
from neurotrade.strategies.gap_continuation import GapContinuation

__all__ = ["GapContinuation", "arsenal"]
