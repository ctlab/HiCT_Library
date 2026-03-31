import numpy as np

from hict_jvm_api.dataloader import HiCTRegionDataset


class _DummyConverter:
    def __init__(self, total_visible_pixels: int = 128):
        self.total_visible_pixels = total_visible_pixels


class _DummyClient:
    def __init__(self) -> None:
        self.requests = []

    def unit_converter(self, bp_resolution: int):
        assert bp_resolution == 50000
        return _DummyConverter(total_visible_pixels=128)

    def fetch_region_pixels(self, *, start_row_px: int, start_col_px: int, rows: int, cols: int, bp_resolution: int):
        self.requests.append((start_row_px, start_col_px, rows, cols, bp_resolution))
        arr = np.zeros((rows, cols, 4), dtype=np.uint8)
        arr[..., 0] = start_row_px % 255
        arr[..., 1] = start_col_px % 255
        arr[..., 2] = 100
        arr[..., 3] = 255
        return arr


def test_dataset_random_sampling_numpy_output() -> None:
    client = _DummyClient()
    ds = HiCTRegionDataset(client=client, bp_resolution=50000, window_px=16, num_samples=5, seed=123)

    assert len(ds) == 5
    sample = ds[0]
    assert sample.shape == (16, 16, 4)
    assert sample.dtype == np.uint8
    assert len(client.requests) == 1


def test_dataset_explicit_regions_and_transform() -> None:
    client = _DummyClient()
    ds = HiCTRegionDataset(
        client=client,
        bp_resolution=50000,
        window_px=8,
        regions=[(10, 20), (30, 40)],
        transform=lambda image: int(image[0, 0, 0]) + int(image[0, 0, 1]),
    )

    assert len(ds) == 2
    value = ds[1]
    assert value == 70
    assert client.requests[-1][:2] == (30, 40)
