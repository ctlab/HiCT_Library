"""Typed models used by :mod:`hict_jvm_api`.

These dataclasses mirror the payloads returned by HiCT_JVM endpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np


class Unit(str, Enum):
    """Length units supported by HiCT queries and conversions."""

    BP = "BP"
    BINS = "BINS"
    PIXELS = "PIXELS"


@dataclass(frozen=True)
class ContigDescriptor:
    """Contig metadata for the current assembly state."""

    contig_id: int
    contig_name: str
    contig_original_name: str
    contig_source_name: str
    contig_offset_in_source: int
    contig_direction: int
    contig_length_bp: int
    contig_length_bins: Dict[int, int]
    contig_presence_at_resolution: Dict[int, int]

    @staticmethod
    def from_json(payload: Mapping[str, object]) -> "ContigDescriptor":
        length_bins_raw = payload.get("contigLengthBins", {})
        presence_raw = payload.get("contigPresenceAtResolution", {})
        return ContigDescriptor(
            contig_id=int(payload.get("contigId", 0)),
            contig_name=str(payload.get("contigName", "")),
            contig_original_name=str(payload.get("contigOriginalName", "")),
            contig_source_name=str(payload.get("contigSourceName", "")),
            contig_offset_in_source=int(payload.get("contigOffsetInSource", 0)),
            contig_direction=int(payload.get("contigDirection", 0)),
            contig_length_bp=int(payload.get("contigLengthBp", 0)),
            contig_length_bins={int(k): int(v) for k, v in dict(length_bins_raw).items()},
            contig_presence_at_resolution={int(k): int(v) for k, v in dict(presence_raw).items()},
        )


@dataclass(frozen=True)
class ScaffoldDescriptor:
    """Scaffold metadata for the current assembly state."""

    scaffold_id: int
    scaffold_name: str
    scaffold_original_name: str
    spacer_length: int
    scaffold_borders_bp: Optional[Dict[str, int]]

    @staticmethod
    def from_json(payload: Mapping[str, object]) -> "ScaffoldDescriptor":
        borders = payload.get("scaffoldBordersBP")
        normalized_borders: Optional[Dict[str, int]]
        if isinstance(borders, Mapping):
            normalized_borders = {str(k): int(v) for k, v in borders.items()}
        else:
            normalized_borders = None
        return ScaffoldDescriptor(
            scaffold_id=int(payload.get("scaffoldId", 0)),
            scaffold_name=str(payload.get("scaffoldName", "")),
            scaffold_original_name=str(payload.get("scaffoldOriginalName", "")),
            spacer_length=int(payload.get("spacerLength", 0)),
            scaffold_borders_bp=normalized_borders,
        )


@dataclass(frozen=True)
class AssemblyInfo:
    """Assembly descriptors returned by HiCT_JVM."""

    contigs: List[ContigDescriptor]
    scaffolds: List[ScaffoldDescriptor]

    @staticmethod
    def from_json(payload: Mapping[str, object]) -> "AssemblyInfo":
        contigs_raw = payload.get("contigDescriptors", [])
        scaffolds_raw = payload.get("scaffoldDescriptors", [])
        return AssemblyInfo(
            contigs=[ContigDescriptor.from_json(item) for item in list(contigs_raw)],
            scaffolds=[ScaffoldDescriptor.from_json(item) for item in list(scaffolds_raw)],
        )


@dataclass(frozen=True)
class OpenFileResponse:
    """Response returned by ``/open`` and embedded in ``/attach``."""

    status: str
    dtype: str
    resolutions: List[int]
    pixel_resolutions: List[float]
    tile_size: int
    assembly_info: AssemblyInfo
    matrix_sizes_bins: List[int]

    @staticmethod
    def from_json(payload: Mapping[str, object]) -> "OpenFileResponse":
        return OpenFileResponse(
            status=str(payload.get("status", "")),
            dtype=str(payload.get("dtype", "")),
            resolutions=[int(v) for v in list(payload.get("resolutions", []))],
            pixel_resolutions=[float(v) for v in list(payload.get("pixelResolutions", []))],
            tile_size=int(payload.get("tileSize", 256)),
            assembly_info=AssemblyInfo.from_json(dict(payload.get("assemblyInfo", {}))),
            matrix_sizes_bins=[int(v) for v in list(payload.get("matrixSizesBins", []))],
        )

    def matrix_size_for_resolution(self, bp_resolution: int) -> int:
        """Return matrix size in bins for a given bp resolution."""
        if bp_resolution not in self.resolutions:
            raise KeyError(f"Resolution {bp_resolution} is not present in opened dataset")
        idx = self.resolutions.index(bp_resolution)
        return self.matrix_sizes_bins[idx]


@dataclass(frozen=True)
class TileRanges:
    """Per-resolution signal ranges returned together with tile image."""

    lower_bounds: Dict[int, float]
    upper_bounds: Dict[int, float]

    @staticmethod
    def from_json(payload: Mapping[str, object]) -> "TileRanges":
        lower = payload.get("lowerBounds", {})
        upper = payload.get("upperBounds", {})
        return TileRanges(
            lower_bounds={int(k): float(v) for k, v in dict(lower).items()},
            upper_bounds={int(k): float(v) for k, v in dict(upper).items()},
        )


@dataclass(frozen=True)
class TileWithRanges:
    """Tile image together with signal ranges metadata."""

    image: np.ndarray
    ranges: TileRanges
    image_data_url: str


@dataclass(frozen=True)
class SecondarySourceStatus:
    """Secondary source status returned by ``/secondary/status`` and related endpoints."""

    attached: bool
    filename: str
    assembly_source: str
    compatibility: Optional[Mapping[str, object]]
    warnings: Sequence[str]

    @staticmethod
    def from_json(payload: Mapping[str, object]) -> "SecondarySourceStatus":
        warnings = payload.get("warnings", [])
        return SecondarySourceStatus(
            attached=bool(payload.get("attached", False)),
            filename=str(payload.get("filename", "")),
            assembly_source=str(payload.get("assemblySource", "PRIMARY")),
            compatibility=payload.get("compatibility") if isinstance(payload.get("compatibility"), Mapping) else None,
            warnings=[str(item) for item in list(warnings)],
        )
