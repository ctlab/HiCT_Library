"""Coordinate conversion utilities for HiCT units.

The converter uses assembly descriptors returned by HiCT_JVM (`/open` or `/attach`)
to translate between:

- base pairs (BP)
- bins at a specific bp resolution (BINS)
- visible pixels (PIXELS), i.e. bins belonging only to contigs marked as SHOWN
  at this resolution.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple, Union

from .exceptions import HiddenCoordinateError
from .models import AssemblyInfo, OpenFileResponse, Unit


@dataclass(frozen=True)
class _Span:
    bp_start: int
    bp_end: int
    bin_start: int
    bin_end: int
    visible_start: int
    visible_end: int
    shown: bool


class UnitConverter:
    """Fast converter between BP/BINS/PIXELS for a given assembly and resolution.

    Parameters
    ----------
    assembly_info:
        Assembly information from :class:`hict_jvm_api.models.OpenFileResponse`.
    bp_resolution:
        Resolution in bp used for BINS and PIXELS conversion.
    """

    def __init__(self, assembly_info: AssemblyInfo, bp_resolution: int):
        if bp_resolution <= 0:
            raise ValueError("bp_resolution must be positive")
        self.bp_resolution = int(bp_resolution)
        self._spans: List[_Span] = []
        self._bp_ends: List[int] = []
        self._bin_ends: List[int] = []
        self._shown_spans: List[_Span] = []
        self._visible_ends: List[int] = []

        bp_cursor = 0
        bin_cursor = 0
        visible_cursor = 0

        for contig in assembly_info.contigs:
            bins = contig.contig_length_bins.get(self.bp_resolution)
            if bins is None:
                # Fall back to ceil(length_bp / bp_resolution) for robustness.
                bins = max(1, (int(contig.contig_length_bp) + self.bp_resolution - 1) // self.bp_resolution)
            bins = int(bins)
            if bins <= 0:
                continue
            bp_len = int(contig.contig_length_bp)
            shown = int(contig.contig_presence_at_resolution.get(self.bp_resolution, 1)) == 1

            span = _Span(
                bp_start=bp_cursor,
                bp_end=bp_cursor + bp_len,
                bin_start=bin_cursor,
                bin_end=bin_cursor + bins,
                visible_start=visible_cursor,
                visible_end=visible_cursor + (bins if shown else 0),
                shown=shown,
            )
            self._spans.append(span)
            self._bp_ends.append(span.bp_end)
            self._bin_ends.append(span.bin_end)
            if shown:
                self._shown_spans.append(span)
                self._visible_ends.append(span.visible_end)

            bp_cursor = span.bp_end
            bin_cursor = span.bin_end
            visible_cursor = span.visible_end

        self.total_bp = bp_cursor
        self.total_bins = bin_cursor
        self.total_visible_pixels = visible_cursor

    @classmethod
    def from_open_file_response(cls, response: OpenFileResponse, bp_resolution: int) -> "UnitConverter":
        """Build a converter from an opened-session response."""
        return cls(response.assembly_info, bp_resolution)

    @staticmethod
    def _normalize_unit(unit: Union[Unit, str]) -> Unit:
        if isinstance(unit, Unit):
            return unit
        token = str(unit).strip().upper()
        if token in {"BP", "BASE_PAIRS", "BASEPAIR", "BASEPAIRS"}:
            return Unit.BP
        if token in {"BIN", "BINS"}:
            return Unit.BINS
        if token in {"PX", "PIXEL", "PIXELS"}:
            return Unit.PIXELS
        raise ValueError("Unsupported unit: %s" % unit)

    @staticmethod
    def _guard_range(value: int, size: int, unit: str, clamp: bool) -> int:
        if size <= 0:
            raise ValueError("Cannot convert with empty assembly")
        if clamp:
            return min(max(int(value), 0), size - 1)
        ivalue = int(value)
        if ivalue < 0 or ivalue >= size:
            raise ValueError("%s coordinate %d is out of range [0, %d)" % (unit, ivalue, size))
        return ivalue

    def _span_by_bp(self, bp: int) -> _Span:
        idx = bisect_right(self._bp_ends, bp)
        if idx >= len(self._spans):
            raise ValueError("BP coordinate %d is outside assembly" % bp)
        return self._spans[idx]

    def _span_by_bin(self, bin_coord: int) -> _Span:
        idx = bisect_right(self._bin_ends, bin_coord)
        if idx >= len(self._spans):
            raise ValueError("BIN coordinate %d is outside assembly" % bin_coord)
        return self._spans[idx]

    def _shown_span_by_pixel(self, pixel: int) -> _Span:
        idx = bisect_right(self._visible_ends, pixel)
        if idx >= len(self._shown_spans):
            raise ValueError("PIXEL coordinate %d is outside visible range" % pixel)
        return self._shown_spans[idx]

    def bp_to_bins(self, bp: int, clamp: bool = False) -> int:
        """Convert BP coordinate to BINS coordinate."""
        bp = self._guard_range(bp, self.total_bp, "BP", clamp)
        span = self._span_by_bp(bp)
        offset_bp = bp - span.bp_start
        candidate = span.bin_start + (offset_bp // self.bp_resolution)
        return min(candidate, span.bin_end - 1)

    def bins_to_bp(self, bins: int, clamp: bool = False) -> int:
        """Convert BINS coordinate to BP coordinate."""
        bins = self._guard_range(bins, self.total_bins, "BINS", clamp)
        span = self._span_by_bin(bins)
        offset_bin = bins - span.bin_start
        candidate = span.bp_start + offset_bin * self.bp_resolution
        return min(candidate, span.bp_end - 1)

    def bins_to_pixels(self, bins: int, clamp: bool = False) -> int:
        """Convert BINS coordinate to PIXELS coordinate.

        Raises
        ------
        HiddenCoordinateError
            If `bins` belongs to a hidden contig at this resolution.
        """
        bins = self._guard_range(bins, self.total_bins, "BINS", clamp)
        span = self._span_by_bin(bins)
        if not span.shown:
            raise HiddenCoordinateError(
                "BIN coordinate %d belongs to hidden contig at %dbp resolution" % (bins, self.bp_resolution)
            )
        return span.visible_start + (bins - span.bin_start)

    def pixels_to_bins(self, pixels: int, clamp: bool = False) -> int:
        """Convert PIXELS coordinate to BINS coordinate."""
        pixels = self._guard_range(pixels, self.total_visible_pixels, "PIXELS", clamp)
        span = self._shown_span_by_pixel(pixels)
        return span.bin_start + (pixels - span.visible_start)

    def bp_to_pixels(self, bp: int, clamp: bool = False) -> int:
        """Convert BP coordinate to PIXELS coordinate."""
        return self.bins_to_pixels(self.bp_to_bins(bp, clamp=clamp), clamp=clamp)

    def pixels_to_bp(self, pixels: int, clamp: bool = False) -> int:
        """Convert PIXELS coordinate to BP coordinate."""
        return self.bins_to_bp(self.pixels_to_bins(pixels, clamp=clamp), clamp=clamp)

    def convert(self, value: int, from_unit: Union[Unit, str], to_unit: Union[Unit, str], clamp: bool = False) -> int:
        """Convert a single coordinate between units."""
        src = self._normalize_unit(from_unit)
        dst = self._normalize_unit(to_unit)
        if src == dst:
            return int(value)

        if src == Unit.BP:
            if dst == Unit.BINS:
                return self.bp_to_bins(value, clamp=clamp)
            return self.bp_to_pixels(value, clamp=clamp)

        if src == Unit.BINS:
            if dst == Unit.BP:
                return self.bins_to_bp(value, clamp=clamp)
            return self.bins_to_pixels(value, clamp=clamp)

        # src == PIXELS
        if dst == Unit.BINS:
            return self.pixels_to_bins(value, clamp=clamp)
        return self.pixels_to_bp(value, clamp=clamp)

    def convert_interval(
        self,
        start: int,
        end_exclusive: int,
        from_unit: Union[Unit, str],
        to_unit: Union[Unit, str],
        clamp: bool = False,
    ) -> Tuple[int, int]:
        """Convert a half-open interval ``[start, end_exclusive)`` between units."""
        if end_exclusive <= start:
            raise ValueError("end_exclusive must be larger than start")
        dst_start = self.convert(start, from_unit, to_unit, clamp=clamp)
        dst_end_inclusive = self.convert(end_exclusive - 1, from_unit, to_unit, clamp=clamp)
        return dst_start, dst_end_inclusive + 1

    @property
    def spans(self) -> Sequence[_Span]:
        """Read-only sequence of conversion spans (mainly for debugging/tests)."""
        return tuple(self._spans)
