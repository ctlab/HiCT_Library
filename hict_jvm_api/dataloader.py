"""PyTorch-friendly datasets for HiCT_JVM region sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .client import HiCTJVMClient
from .models import Unit


@dataclass(frozen=True)
class RegionRequest:
    """A single region request in visible pixel coordinates."""

    row_px: int
    col_px: int


class HiCTRegionDataset:
    """Random-access dataset that fetches regions from a running HiCT_JVM session.

    Parameters
    ----------
    client:
        Active :class:`hict_jvm_api.client.HiCTJVMClient` instance.
    bp_resolution:
        Resolution used for tile/region extraction.
    window_px:
        Square crop size in pixels.
    num_samples:
        Number of samples generated when ``regions`` is not provided.
    regions:
        Explicit pixel coordinates to fetch. If set, ``num_samples`` is ignored.
    seed:
        RNG seed used for synthetic sampling.
    return_torch:
        Return ``torch.Tensor`` instead of ``numpy.ndarray``.
    channel_first:
        If ``return_torch=True``, output layout is ``(C,H,W)`` when enabled,
        otherwise ``(H,W,C)``.
    transform:
        Optional callable applied to each sample.
    """

    def __init__(
        self,
        *,
        client: HiCTJVMClient,
        bp_resolution: int,
        window_px: int,
        num_samples: int = 1024,
        regions: Optional[Sequence[Tuple[int, int]]] = None,
        seed: int = 0,
        return_torch: bool = False,
        channel_first: bool = True,
        transform: Optional[Callable[[np.ndarray], object]] = None,
    ):
        if window_px <= 0:
            raise ValueError("window_px must be positive")
        if num_samples <= 0 and not regions:
            raise ValueError("num_samples must be positive when regions are not provided")

        self.client = client
        self.bp_resolution = int(bp_resolution)
        self.window_px = int(window_px)
        self.return_torch = bool(return_torch)
        self.channel_first = bool(channel_first)
        self.transform = transform

        converter = self.client.unit_converter(self.bp_resolution)
        self._visible_size = int(converter.total_visible_pixels)
        max_coord = max(0, self._visible_size - self.window_px)

        if regions is not None:
            self._regions = [RegionRequest(int(r), int(c)) for r, c in regions]
        else:
            rng = np.random.default_rng(seed)
            self._regions = [
                RegionRequest(
                    row_px=int(rng.integers(0, max_coord + 1)),
                    col_px=int(rng.integers(0, max_coord + 1)),
                )
                for _ in range(int(num_samples))
            ]

    def __len__(self) -> int:
        return len(self._regions)

    def __getitem__(self, index: int):
        region = self._regions[index]
        image = self.client.fetch_region_pixels(
            start_row_px=region.row_px,
            start_col_px=region.col_px,
            rows=self.window_px,
            cols=self.window_px,
            bp_resolution=self.bp_resolution,
        )

        if self.transform is not None:
            image = self.transform(image)
            return image

        if not self.return_torch:
            return image

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "return_torch=True requires torch to be installed. "
                "Install optional dependency: pip install 'hict[torch]'"
            ) from exc

        tensor = torch.from_numpy(np.ascontiguousarray(image))
        if self.channel_first:
            tensor = tensor.permute(2, 0, 1)
        return tensor.float() / 255.0


class HiCTSignalDataset:
    """Random-access dataset fetching numeric Hi-C signal matrices.

    Regions are sampled in a selected coordinate unit and fetched through
    ``POST /matrix/query`` using the JVM-backed client.
    """

    def __init__(
        self,
        *,
        client: HiCTJVMClient,
        bp_resolution: int,
        window_size: int,
        unit: Union[Unit, str] = Unit.PIXELS,
        signal_mode: str = "TRADITIONAL_NORMALIZED",
        dtype: str = "float32",
        num_samples: int = 1024,
        regions: Optional[Sequence[Tuple[int, int]]] = None,
        seed: int = 0,
        return_torch: bool = False,
        transform: Optional[Callable[[np.ndarray], object]] = None,
    ):
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if num_samples <= 0 and not regions:
            raise ValueError("num_samples must be positive when regions are not provided")

        self.client = client
        self.bp_resolution = int(bp_resolution)
        self.window_size = int(window_size)
        self.unit = unit
        self.signal_mode = str(signal_mode)
        self.dtype = str(dtype)
        self.return_torch = bool(return_torch)
        self.transform = transform

        converter = self.client.unit_converter(self.bp_resolution)
        if isinstance(unit, Unit):
            normalized = unit
        else:
            normalized = Unit(str(unit).upper())

        if normalized == Unit.PIXELS:
            total_size = int(converter.total_visible_pixels)
        elif normalized == Unit.BINS:
            total_size = int(converter.total_bins)
        else:
            total_size = int(converter.total_bp)

        max_coord = max(0, total_size - self.window_size)

        if regions is not None:
            self._regions = [RegionRequest(int(r), int(c)) for r, c in regions]
        else:
            rng = np.random.default_rng(seed)
            self._regions = [
                RegionRequest(
                    row_px=int(rng.integers(0, max_coord + 1)),
                    col_px=int(rng.integers(0, max_coord + 1)),
                )
                for _ in range(int(num_samples))
            ]

    def __len__(self) -> int:
        return len(self._regions)

    def __getitem__(self, index: int):
        region = self._regions[index]
        matrix = self.client.fetch_region_signal(
            start_row=region.row_px,
            start_col=region.col_px,
            rows=self.window_size,
            cols=self.window_size,
            bp_resolution=self.bp_resolution,
            unit=self.unit,
            signal_mode=self.signal_mode,
            dtype=self.dtype,
        )

        if self.transform is not None:
            matrix = self.transform(matrix)
            return matrix

        if not self.return_torch:
            return matrix

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "return_torch=True requires torch to be installed. "
                "Install optional dependency: pip install 'hict[torch]'"
            ) from exc

        return torch.from_numpy(np.ascontiguousarray(matrix))
