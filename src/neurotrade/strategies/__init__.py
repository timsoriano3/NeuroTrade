"""Strategy plugins.

Each strategy produces `Intent`s — proposals carrying a side and an
invalidation level. None of them produces an order, and none of them decides a
size. Direction is the strategy's job; conviction and size belong to the model
and the risk engine (§7.3).

Depends on: core, features.
"""
