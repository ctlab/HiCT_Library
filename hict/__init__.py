"""HiCT Python API facade.

This package now defaults to the JVM-backed API client for production usage.
Legacy pure-Python modules remain importable from their original subpackages,
but new code should use :class:`HiCTClient`.
"""

from __future__ import annotations

import warnings

from hict_jvm_api.client import HiCTJVMClient
from hict_jvm_api.dataloader import HiCTRegionDataset, HiCTSignalDataset
from hict_jvm_api.exceptions import HiCTAPIError, HiCTClientStateError, HiddenCoordinateError
from hict_jvm_api.models import (
    AssemblyInfo,
    ContigDescriptor,
    OpenFileResponse,
    ScaffoldDescriptor,
    TileRanges,
    TileWithRanges,
    Unit,
)
from hict_jvm_api.units import UnitConverter

# Backward-compatible alias for the primary entrypoint.
HiCTClient = HiCTJVMClient

__all__ = [
    "HiCTClient",
    "HiCTJVMClient",
    "HiCTRegionDataset",
    "HiCTSignalDataset",
    "HiCTAPIError",
    "HiCTClientStateError",
    "HiddenCoordinateError",
    "AssemblyInfo",
    "ContigDescriptor",
    "OpenFileResponse",
    "ScaffoldDescriptor",
    "TileRanges",
    "TileWithRanges",
    "Unit",
    "UnitConverter",
]


def _warn_legacy_import(path: str) -> None:
    warnings.warn(
        (
            f"The legacy pure-Python API module '{path}' is kept for compatibility "
            "but is no longer the recommended API. Use 'hict.HiCTClient' "
            "(JVM-backed) for maintained functionality."
        ),
        category=DeprecationWarning,
        stacklevel=2,
    )
