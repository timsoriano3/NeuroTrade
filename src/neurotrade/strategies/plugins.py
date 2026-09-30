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
from neurotrade.strategies.intraday_momentum import IntradayMomentum
from neurotrade.strategies.intraday_reversal import IntradayReversal
from neurotrade.strategies.momentum_ignition import MomentumIgnition
from neurotrade.strategies.opening_range_breakout import OpeningRangeBreakout
from neurotrade.strategies.orb_fade import OrbFade
from neurotrade.strategies.relative_strength import RelativeStrength
from neurotrade.strategies.residual_reversion import ResidualReversion
from neurotrade.strategies.vwap_band_reversion import VwapBandReversion

__all__ = [
    "GapContinuation",
    "IntradayMomentum",
    "IntradayReversal",
    "MomentumIgnition",
    "OpeningRangeBreakout",
    "OrbFade",
    "RelativeStrength",
    "ResidualReversion",
    "VwapBandReversion",
    "arsenal",
]
