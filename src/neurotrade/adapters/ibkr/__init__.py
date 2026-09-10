"""Interactive Brokers adapter.

Talks to IB Gateway over its socket API. Everything here sits behind the ports
in `core/ports.py`, so the trading logic never learns that IBKR exists.
"""
