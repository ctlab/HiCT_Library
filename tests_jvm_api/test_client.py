import base64
from io import BytesIO
from typing import Any, Dict, List, Mapping
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

from hict_jvm_api.client import HiCTJVMClient
from hict_jvm_api.exceptions import HiCTAPIError


class _FakeResponse:
    def __init__(self, status_code: int, *, json_payload: Any = None, text: str = "", content: bytes = b""):
        self.status_code = status_code
        self._json_payload = json_payload
        self.text = text if text else ("" if json_payload is not None else content.decode("utf-8", errors="ignore"))
        self.content = content

    def json(self) -> Any:
        if self._json_payload is None:
            raise ValueError("No JSON payload")
        return self._json_payload


class _FakeSession:
    def __init__(self, handler):
        self._handler = handler

    def request(self, method: str, url: str, *, params: Mapping[str, Any], json: Mapping[str, Any], timeout: float):
        parsed = urlparse(url)
        query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        merged_params = dict(query)
        merged_params.update({k: str(v) for k, v in dict(params).items()})
        return self._handler(method.upper(), parsed.path, merged_params, dict(json))


def _png_bytes(rgba: tuple[int, int, int, int], width: int = 2, height: int = 2) -> bytes:
    arr = np.zeros((height, width, 4), dtype=np.uint8)
    arr[:, :] = np.array(rgba, dtype=np.uint8)
    image = Image.fromarray(arr, mode="RGBA")
    bio = BytesIO()
    image.save(bio, format="PNG")
    return bio.getvalue()


def _data_url_from_png_bytes(content: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(content).decode("ascii")


def test_open_and_fetch_tile_with_ranges_decodes_image() -> None:
    png = _png_bytes((10, 20, 30, 255))
    data_url = _data_url_from_png_bytes(png)

    def handler(method: str, path: str, params: Mapping[str, str], payload: Mapping[str, Any]) -> _FakeResponse:
        if method == "POST" and path == "/open":
            assert payload["filename"] == "build/quad/combined_ind2_4DN.hict.hdf5"
            return _FakeResponse(
                200,
                json_payload={
                    "status": "Opened",
                    "dtype": "uint8",
                    "resolutions": [50000, 1000],
                    "pixelResolutions": [50.0, 1.0],
                    "tileSize": 256,
                    "assemblyInfo": {
                        "contigDescriptors": [
                            {
                                "contigId": 1,
                                "contigName": "ctg1",
                                "contigOriginalName": "ctg1",
                                "contigSourceName": "ctg1",
                                "contigOffsetInSource": 0,
                                "contigDirection": 0,
                                "contigLengthBp": 200,
                                "contigLengthBins": {"50000": 1, "1000": 1},
                                "contigPresenceAtResolution": {"50000": 1, "1000": 1},
                            }
                        ],
                        "scaffoldDescriptors": [],
                    },
                    "matrixSizesBins": [1, 1],
                },
            )

        if method == "GET" and path == "/get_tile":
            assert params["format"] == "JSON_PNG_WITH_RANGES"
            return _FakeResponse(
                200,
                json_payload={
                    "image": data_url,
                    "ranges": {
                        "lowerBounds": {"1": 0.0},
                        "upperBounds": {"1": 1.0},
                    },
                },
            )

        raise AssertionError(f"Unexpected request: {method} {path}")

    client = HiCTJVMClient("http://test", session=_FakeSession(handler))

    opened = client.open_file("build/quad/combined_ind2_4DN.hict.hdf5")
    assert opened.resolutions == [50000, 1000]

    tile = client.fetch_tile_with_ranges(row=0, col=0, bp_resolution=50000)
    assert tile.image.shape == (2, 2, 4)
    assert int(tile.image[0, 0, 0]) == 10
    assert tile.ranges.upper_bounds[1] == 1.0



def test_fetch_region_pixels_uses_png_by_pixels() -> None:
    png = _png_bytes((3, 4, 5, 255), width=4, height=3)

    def handler(method: str, path: str, params: Mapping[str, str], payload: Mapping[str, Any]) -> _FakeResponse:
        if method == "GET" and path == "/get_tile":
            assert params["format"] == "PNG_BY_PIXELS"
            assert params["rows"] == "3"
            assert params["cols"] == "4"
            return _FakeResponse(200, content=png)
        raise AssertionError(f"Unexpected request: {method} {path}")

    client = HiCTJVMClient("http://test", session=_FakeSession(handler))

    image = client.fetch_region_pixels(
        start_row_px=10,
        start_col_px=20,
        rows=3,
        cols=4,
        bp_resolution=50000,
    )
    assert image.shape == (3, 4, 4)
    assert int(image[0, 0, 1]) == 4



def test_start_conversion_job_sends_overwrite() -> None:
    seen_payloads: List[Dict[str, Any]] = []

    def handler(method: str, path: str, params: Mapping[str, str], payload: Mapping[str, Any]) -> _FakeResponse:
        if method == "POST" and path == "/convert/jobs":
            seen_payloads.append(dict(payload))
            return _FakeResponse(200, json_payload={"status": "submitted", "jobId": "job-1"})
        raise AssertionError(f"Unexpected request: {method} {path}")

    client = HiCTJVMClient("http://test", session=_FakeSession(handler))

    response = client.start_conversion_job(filename="x.mcool", overwrite=True)
    assert response["jobId"] == "job-1"
    assert seen_payloads[-1]["overwrite"] is True



def test_api_error_payload_raises_exception() -> None:
    def handler(method: str, path: str, params: Mapping[str, str], payload: Mapping[str, Any]) -> _FakeResponse:
        return _FakeResponse(200, json_payload={"error": "Something failed"})

    client = HiCTJVMClient("http://test", session=_FakeSession(handler))

    try:
        client.list_files()
        assert False, "Expected HiCTAPIError"
    except HiCTAPIError as err:
        assert "Something failed" in str(err)


def test_openapi_spec_yaml_fetch() -> None:
    def handler(method: str, path: str, params: Mapping[str, str], payload: Mapping[str, Any]) -> _FakeResponse:
        if method == "GET" and path == "/api/v1/openapi.yaml":
            return _FakeResponse(200, text="openapi: 3.0.3\npaths: {}\n")
        raise AssertionError(f"Unexpected request: {method} {path}")

    client = HiCTJVMClient("http://test", session=_FakeSession(handler))
    spec = client.openapi_spec_yaml()
    assert "openapi: 3.0.3" in spec
