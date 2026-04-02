"""Legacy pure-Python API namespace.

This module is retained for backward compatibility.
Prefer `hict.HiCTClient` for the maintained JVM-backed API.
"""

from __future__ import annotations

import warnings

warnings.warn(
    (
        "hict.api is deprecated and kept for compatibility only. "
        "Use hict.HiCTClient (JVM-backed API) for new development."
    ),
    category=DeprecationWarning,
    stacklevel=2,
)

from .ContactMatrixFacet import ContactMatrixFacet

__all__ = ["ContactMatrixFacet"]
