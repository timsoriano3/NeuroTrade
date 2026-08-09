"""NeuroTrade — autonomous day-trading system for US and Canadian equities.

Rules produce trade direction; ML produces conviction and size; hard risk limits
are structural and cannot be overridden by a model.

Subpackages are added as each layer is built. See CLAUDE.md for the dependency
direction between them, which is enforced in CI rather than by convention.
"""

__version__ = "0.0.0"
