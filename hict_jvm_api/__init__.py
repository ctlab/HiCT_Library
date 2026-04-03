"""HiCT JVM API client package.

This package provides a Python interface for communicating with a running
HiCT_JVM API server. It includes:

- :class:`hict_jvm_api.client.HiCTJVMClient` for server operations
- :class:`hict_jvm_api.units.UnitConverter` for fast BP/BINS/PIXELS conversion
- :class:`hict_jvm_api.dataloader.HiCTRegionDataset` for rendered RGBA region loading
- :class:`hict_jvm_api.dataloader.HiCTSignalDataset` for numeric signal matrix loading
"""

from .client import HiCTJVMClient
from .dataloader import HiCTRegionDataset, HiCTSignalDataset
from .exceptions import HiCTAPIError, HiCTClientStateError, HiddenCoordinateError
from .models import (
    AssemblyInfo,
    ContigDescriptor,
    OpenFileResponse,
    ScaffoldDescriptor,
    TileRanges,
    TileWithRanges,
    Unit,
)
from .units import UnitConverter

__all__ = [
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
