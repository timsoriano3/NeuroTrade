"""Feature library: point-in-time-correct functions over market data.

One implementation, imported by both research and live (§3.6). This is the
invariant the whole design rests on — a feature computed one way in the backtest
and another way in production makes every backtest result a claim about
software that is not the software running.

Depends on: core.
"""
