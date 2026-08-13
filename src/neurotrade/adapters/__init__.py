"""Adapters: everything that touches the outside world.

IBKR, Parquet, DuckDB, Postgres, notifications. All of it sits behind the ports
defined in `neurotrade.core.ports`, so swapping a broker or a storage backend is
a new adapter rather than a change above this layer.

Depends on: core.
"""
