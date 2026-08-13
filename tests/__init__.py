"""Test suite.

Package markers exist so that same-named modules in different layers —
`tests/core/test_registry.py` and `tests/features/test_registry.py` — resolve as
distinct modules. Without them mypy sees one ambiguous `test_registry`.
"""
