"""High-level Python client for HiCT_JVM HTTP API."""

from __future__ import annotations

import base64
import time
from io import BytesIO
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import numpy as np
import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from .exceptions import HiCTAPIError, HiCTClientStateError
from .models import OpenFileResponse, SecondarySourceStatus, TileRanges, TileWithRanges, Unit
from .units import UnitConverter


class HiCTJVMClient:
    """Client for communicating with a running HiCT_JVM API server.

    The client uses a persistent HTTP session (`requests.Session`) for connection
    reuse and better throughput in region-heavy workloads.
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        session: Optional[requests.Session] = None,
        *,
        pool_maxsize: int = 64,
        max_retries: int = 2,
        retry_backoff_sec: float = 0.2,
    ):
        if not base_url:
            raise ValueError("base_url must be provided")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._own_session = session is None
        if session is not None:
            self._http = session
        else:
            self._http = self._create_default_session(
                pool_maxsize=int(max(1, pool_maxsize)),
                max_retries=int(max(0, max_retries)),
                retry_backoff_sec=float(max(0.0, retry_backoff_sec)),
            )
        self._open_file_response: Optional[OpenFileResponse] = None

    @staticmethod
    def _create_default_session(
        *,
        pool_maxsize: int,
        max_retries: int,
        retry_backoff_sec: float,
    ) -> requests.Session:
        session = requests.Session()
        retries = Retry(
            total=max_retries,
            connect=max_retries,
            read=max_retries,
            status=max_retries,
            backoff_factor=retry_backoff_sec,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET", "POST"}),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(
            max_retries=retries,
            pool_connections=pool_maxsize,
            pool_maxsize=pool_maxsize,
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update(
            {
                "Accept": "application/json",
                "Accept-Encoding": "gzip, deflate",
                "User-Agent": "hict-python/1 (jvm-api)",
            }
        )
        return session

    def close(self) -> None:
        """Close the underlying HTTP session if owned by this client."""
        if self._own_session and hasattr(self._http, "close"):
            self._http.close()

    def __enter__(self) -> "HiCTJVMClient":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def _url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return self.base_url + path

    @staticmethod
    def _raise_if_error_payload(payload: Any, status_code: Optional[int] = None) -> None:
        if isinstance(payload, Mapping):
            error = payload.get("error")
            if isinstance(error, str) and error.strip():
                raise HiCTAPIError(error, status_code=status_code, payload=payload)

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> requests.Response:
        method_upper = method.upper()
        resolved_params = dict(params or {})
        resolved_payload = dict(payload or {})
        request_kwargs: Dict[str, Any] = {
            "method": method_upper,
            "url": self._url(path),
            "params": resolved_params,
            "json": resolved_payload,
            "timeout": self.timeout,
        }
        response = self._http.request(**request_kwargs)
        if response.status_code >= 400:
            message = response.text
            parsed_payload: Any = None
            try:
                parsed_payload = response.json()
                if isinstance(parsed_payload, Mapping) and isinstance(parsed_payload.get("error"), str):
                    message = str(parsed_payload["error"])
            except ValueError:
                parsed_payload = response.text
            raise HiCTAPIError(message, status_code=response.status_code, payload=parsed_payload)
        return response

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        response = self._request(method, path, params=params, payload=payload)
        try:
            data = response.json()
        except ValueError as exc:
            raise HiCTAPIError(
                "Expected JSON response from %s %s" % (method, path),
                status_code=response.status_code,
                payload=response.text,
            ) from exc
        self._raise_if_error_payload(data, status_code=response.status_code)
        return data

    def _request_text(self, method: str, path: str, payload: Optional[Mapping[str, Any]] = None) -> str:
        response = self._request(method, path, payload=payload)
        return response.text

    @staticmethod
    def _header_int(response: requests.Response, header_name: str, fallback: int = 0) -> int:
        raw = response.headers.get(header_name)
        if raw is None:
            return int(fallback)
        try:
            return int(raw)
        except (TypeError, ValueError):
            return int(fallback)

    @staticmethod
    def _normalize_unit(unit: Union[Unit, str]) -> str:
        if isinstance(unit, Unit):
            return unit.value
        value = str(unit).strip().upper()
        if value in {"PX", "PIXEL", "PIXELS"}:
            return Unit.PIXELS.value
        if value in {"BIN", "BINS"}:
            return Unit.BINS.value
        if value in {"BP", "BASE_PAIRS", "BASEPAIR", "BASEPAIRS"}:
            return Unit.BP.value
        raise ValueError(f"Unsupported unit: {unit}")

    @staticmethod
    def _matrix_format_for_dtype(dtype: str) -> str:
        normalized = str(dtype).strip().lower()
        if normalized in {"float32", "f32"}:
            return "BINARY_FLOAT32"
        if normalized in {"float64", "double", "f64"}:
            return "BINARY_FLOAT64"
        if normalized in {"int64", "i64", "long"}:
            return "BINARY_INT64"
        raise ValueError(f"Unsupported matrix dtype '{dtype}'. Use float32, float64 or int64.")

    @staticmethod
    def _numpy_dtype_for_binary_format(binary_format: str) -> np.dtype:
        normalized = binary_format.strip().upper()
        if normalized == "BINARY_FLOAT32":
            return np.dtype("<f4")
        if normalized == "BINARY_FLOAT64":
            return np.dtype("<f8")
        if normalized == "BINARY_INT64":
            return np.dtype("<i8")
        raise ValueError(f"Unsupported matrix binary format '{binary_format}'")

    @staticmethod
    def _decode_png_bytes(png_bytes: bytes) -> np.ndarray:
        with Image.open(BytesIO(png_bytes)) as image:
            rgba = image.convert("RGBA")
            return np.asarray(rgba, dtype=np.uint8)

    @staticmethod
    def _decode_data_url_png(image_data_url: str) -> np.ndarray:
        prefix = "data:image/png;base64,"
        if not image_data_url.startswith(prefix):
            raise HiCTAPIError("Unexpected tile image payload format", payload=image_data_url)
        encoded = image_data_url[len(prefix) :]
        return HiCTJVMClient._decode_png_bytes(base64.b64decode(encoded))

    def require_open_file_response(self) -> OpenFileResponse:
        """Return cached open-file metadata or raise when session is not attached."""
        if self._open_file_response is None:
            raise HiCTClientStateError(
                "No opened session metadata is cached. Call open_file(...) or attach_session() first."
            )
        return self._open_file_response

    # -------------------------
    # Session and file control
    # -------------------------
    def version(self) -> Mapping[str, Any]:
        """Get HiCT_JVM version payload from ``GET /version``."""
        payload = self._request_json("GET", "/version")
        return payload

    def openapi_spec_yaml(self) -> str:
        """Get backend OpenAPI v1 YAML from ``GET /api/v1/openapi.yaml``."""
        return self._request_text("GET", "/api/v1/openapi.yaml")

    def list_files(self) -> Sequence[str]:
        payload = self._request_json("POST", "/list_files", payload={})
        return [str(item) for item in payload]

    def list_files_detailed(self) -> Sequence[Mapping[str, Any]]:
        payload = self._request_json("POST", "/list_files_detailed", payload={})
        return list(payload)

    def list_coolers(self) -> Sequence[str]:
        payload = self._request_json("POST", "/list_coolers", payload={})
        return [str(item) for item in payload]

    def list_fasta_files(self) -> Sequence[str]:
        payload = self._request_json("POST", "/list_fasta_files", payload={})
        return [str(item) for item in payload]

    def list_agp_files(self) -> Sequence[str]:
        payload = self._request_json("POST", "/list_agp_files", payload={})
        return [str(item) for item in payload]

    def open_file(self, filename: str, fasta_filename: Optional[str] = None) -> OpenFileResponse:
        body: Dict[str, Any] = {"filename": filename}
        if fasta_filename:
            body["fastaFilename"] = fasta_filename
        payload = self._request_json("POST", "/open", payload=body)
        parsed = OpenFileResponse.from_json(payload)
        self._open_file_response = parsed
        return parsed

    def open_progress(self) -> Mapping[str, Any]:
        return self._request_json("POST", "/open_progress", payload={})

    def attach_session(self) -> Mapping[str, Any]:
        payload = self._request_json("POST", "/attach", payload={})
        open_response_raw = payload.get("openFileResponse") if isinstance(payload, Mapping) else None
        if isinstance(open_response_raw, Mapping):
            self._open_file_response = OpenFileResponse.from_json(open_response_raw)
        return payload

    def close_session(self) -> Mapping[str, Any]:
        payload = self._request_json("POST", "/close", payload={})
        self._open_file_response = None
        return payload

    # -------------------------
    # Secondary source control
    # -------------------------
    def get_secondary_status(self) -> SecondarySourceStatus:
        payload = self._request_json("POST", "/secondary/status", payload={})
        return SecondarySourceStatus.from_json(payload)

    def open_secondary_source(self, filename: str, allow_mismatch: bool = False) -> Mapping[str, Any]:
        return self._request_json(
            "POST",
            "/secondary/open",
            payload={"filename": filename, "allowMismatch": bool(allow_mismatch)},
        )

    def close_secondary_source(self) -> SecondarySourceStatus:
        payload = self._request_json("POST", "/secondary/close", payload={})
        return SecondarySourceStatus.from_json(payload)

    def set_assembly_source(self, assembly_source: str) -> Mapping[str, Any]:
        return self._request_json(
            "POST", "/secondary/set_assembly_source", payload={"assemblySource": assembly_source}
        )

    # -------------------------
    # Map tile and region fetch
    # -------------------------
    def fetch_tile_with_ranges(
        self,
        row: int,
        col: int,
        *,
        bp_resolution: Optional[int] = None,
        level: Optional[int] = None,
        version: int = 0,
        tile_size: int = 256,
    ) -> TileWithRanges:
        if bp_resolution is None and level is None:
            raise ValueError("Either bp_resolution or level must be specified")
        params: Dict[str, Any] = {
            "row": int(row),
            "col": int(col),
            "version": int(version),
            "tile_size": int(tile_size),
            "format": "JSON_PNG_WITH_RANGES",
        }
        if bp_resolution is not None:
            params["bpResolution"] = int(bp_resolution)
        else:
            params["level"] = int(level)
        payload = self._request_json("GET", "/get_tile", params=params)
        image_data_url = str(payload["image"])
        return TileWithRanges(
            image=self._decode_data_url_png(image_data_url),
            ranges=TileRanges.from_json(dict(payload.get("ranges", {}))),
            image_data_url=image_data_url,
        )

    def fetch_tile_png(
        self,
        row: int,
        col: int,
        *,
        bp_resolution: Optional[int] = None,
        level: Optional[int] = None,
        version: int = 0,
        tile_size: int = 256,
    ) -> np.ndarray:
        if bp_resolution is None and level is None:
            raise ValueError("Either bp_resolution or level must be specified")
        params: Dict[str, Any] = {
            "row": int(row),
            "col": int(col),
            "version": int(version),
            "tile_size": int(tile_size),
            "format": "PNG",
        }
        if bp_resolution is not None:
            params["bpResolution"] = int(bp_resolution)
        else:
            params["level"] = int(level)
        response = self._request("GET", "/get_tile", params=params)
        return self._decode_png_bytes(response.content)

    def fetch_region_pixels(
        self,
        *,
        start_row_px: int,
        start_col_px: int,
        rows: int,
        cols: int,
        bp_resolution: int,
        version: int = 0,
    ) -> np.ndarray:
        params = {
            "row": int(start_row_px),
            "col": int(start_col_px),
            "rows": int(rows),
            "cols": int(cols),
            "bpResolution": int(bp_resolution),
            "version": int(version),
            "format": "PNG_BY_PIXELS",
        }
        response = self._request("GET", "/get_tile", params=params)
        return self._decode_png_bytes(response.content)

    def fetch_region_bins(
        self,
        *,
        start_row_bin: int,
        start_col_bin: int,
        rows: int,
        cols: int,
        bp_resolution: int,
        version: int = 0,
    ) -> np.ndarray:
        converter = self.unit_converter(bp_resolution)
        row_px = converter.convert(start_row_bin, Unit.BINS, Unit.PIXELS)
        col_px = converter.convert(start_col_bin, Unit.BINS, Unit.PIXELS)
        return self.fetch_region_pixels(
            start_row_px=row_px,
            start_col_px=col_px,
            rows=rows,
            cols=cols,
            bp_resolution=bp_resolution,
            version=version,
        )

    def fetch_region_bp(
        self,
        *,
        start_row_bp: int,
        start_col_bp: int,
        rows: int,
        cols: int,
        bp_resolution: int,
        version: int = 0,
    ) -> np.ndarray:
        converter = self.unit_converter(bp_resolution)
        row_px = converter.convert(start_row_bp, Unit.BP, Unit.PIXELS)
        col_px = converter.convert(start_col_bp, Unit.BP, Unit.PIXELS)
        return self.fetch_region_pixels(
            start_row_px=row_px,
            start_col_px=col_px,
            rows=rows,
            cols=cols,
            bp_resolution=bp_resolution,
            version=version,
        )

    def fetch_region_signal(
        self,
        *,
        start_row: int,
        start_col: int,
        rows: int,
        cols: int,
        bp_resolution: int,
        unit: Union[Unit, str] = Unit.PIXELS,
        signal_mode: str = "TRADITIONAL_NORMALIZED",
        dtype: str = "float32",
    ) -> np.ndarray:
        """Fetch numeric matrix region as a dense NumPy array.

        This method uses ``POST /matrix/query`` with binary payload transfer
        for performance-sensitive ML/data-loader workflows.
        """
        if rows <= 0 or cols <= 0:
            raise ValueError("rows and cols must be positive")

        normalized_unit = self._normalize_unit(unit)
        binary_format = self._matrix_format_for_dtype(dtype)
        payload: Dict[str, Any] = {
            "bpResolution": int(bp_resolution),
            "unit": normalized_unit,
            "startRow": int(start_row),
            "startCol": int(start_col),
            "endRow": int(start_row) + int(rows),
            "endCol": int(start_col) + int(cols),
            "signalMode": str(signal_mode).strip().upper(),
            "format": binary_format,
        }
        response = self._request("POST", "/matrix/query", payload=payload)
        response_dtype = self._numpy_dtype_for_binary_format(binary_format)
        rows_out = self._header_int(response, "x-hict-rows", rows)
        cols_out = self._header_int(response, "x-hict-cols", cols)
        data = np.frombuffer(response.content, dtype=response_dtype)
        expected = int(rows_out) * int(cols_out)
        if data.size != expected:
            raise HiCTAPIError(
                f"Unexpected matrix payload size: got {data.size}, expected {expected} "
                f"for shape ({rows_out}, {cols_out})"
            )
        return data.reshape((rows_out, cols_out))

    def fetch_region_signal_torch(
        self,
        *,
        start_row: int,
        start_col: int,
        rows: int,
        cols: int,
        bp_resolution: int,
        unit: Union[Unit, str] = Unit.PIXELS,
        signal_mode: str = "TRADITIONAL_NORMALIZED",
        dtype: str = "float32",
    ):
        """Fetch numeric matrix region as a ``torch.Tensor``."""
        matrix = self.fetch_region_signal(
            start_row=start_row,
            start_col=start_col,
            rows=rows,
            cols=cols,
            bp_resolution=bp_resolution,
            unit=unit,
            signal_mode=signal_mode,
            dtype=dtype,
        )
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "fetch_region_signal_torch requires torch to be installed. "
                "Install optional dependency: pip install 'hict[torch]'"
            ) from exc
        return torch.from_numpy(np.ascontiguousarray(matrix))

    # -------------------------
    # Unit conversion utilities
    # -------------------------
    def unit_converter(self, bp_resolution: int) -> UnitConverter:
        return UnitConverter.from_open_file_response(self.require_open_file_response(), bp_resolution)

    def convert_units(
        self,
        value: int,
        *,
        from_unit: Union[Unit, str],
        to_unit: Union[Unit, str],
        bp_resolution: int,
        clamp: bool = False,
    ) -> int:
        return self.unit_converter(bp_resolution).convert(value, from_unit, to_unit, clamp=clamp)

    # -------------------------
    # Visualization options/pipeline
    # -------------------------
    def get_visualization_options(self) -> Mapping[str, Any]:
        return self._request_json("POST", "/get_visualization_options", payload={})

    def set_visualization_options(self, options: Mapping[str, Any]) -> Mapping[str, Any]:
        return self._request_json("POST", "/set_visualization_options", payload=dict(options))

    def get_render_pipeline(self) -> Mapping[str, Any]:
        return self._request_json("POST", "/render_pipeline/get", payload={})

    def set_render_pipeline(self, config: Mapping[str, Any]) -> Mapping[str, Any]:
        return self._request_json("POST", "/render_pipeline/set", payload=dict(config))

    def reset_render_pipeline(self) -> Mapping[str, Any]:
        return self._request_json("POST", "/render_pipeline/reset", payload={})

    # -------------------------
    # Scaffolding operations
    # -------------------------
    def reverse_selection_range(self, start_bp: int, end_bp: int) -> Mapping[str, Any]:
        return self._request_json(
            "POST", "/reverse_selection_range", payload={"startBP": int(start_bp), "endBP": int(end_bp)}
        )

    def move_selection_range(self, start_bp: int, end_bp: int, target_start_bp: int) -> Mapping[str, Any]:
        return self._request_json(
            "POST",
            "/move_selection_range",
            payload={"startBP": int(start_bp), "endBP": int(end_bp), "targetStartBP": int(target_start_bp)},
        )

    def split_contig_at_bin(self, split_px: int, bp_resolution: int) -> Mapping[str, Any]:
        return self._request_json(
            "POST",
            "/split_contig_at_bin",
            payload={"splitPx": int(split_px), "bpResolution": int(bp_resolution)},
        )

    def group_contigs_into_scaffold(self, start_bp: int, end_bp: int) -> Mapping[str, Any]:
        return self._request_json(
            "POST", "/group_contigs_into_scaffold", payload={"startBP": int(start_bp), "endBP": int(end_bp)}
        )

    def ungroup_contigs_from_scaffold(self, start_bp: int, end_bp: int) -> Mapping[str, Any]:
        return self._request_json(
            "POST", "/ungroup_contigs_from_scaffold", payload={"startBP": int(start_bp), "endBP": int(end_bp)}
        )

    def move_selection_to_debris(self, start_bp: int, end_bp: int) -> Mapping[str, Any]:
        return self._request_json(
            "POST", "/move_selection_to_debris", payload={"startBP": int(start_bp), "endBP": int(end_bp)}
        )

    # -------------------------
    # FASTA / AGP operations
    # -------------------------
    def link_fasta(self, fasta_filename: str, allow_mismatch: bool = False) -> Mapping[str, Any]:
        return self._request_json(
            "POST",
            "/link_fasta",
            payload={"fastaFilename": fasta_filename, "allowMismatch": bool(allow_mismatch)},
        )

    def export_fasta_for_assembly(self) -> str:
        return self._request_text("POST", "/get_fasta_for_assembly", payload={})

    def export_fasta_for_selection(
        self, from_bp_x: int, from_bp_y: int, to_bp_x: int, to_bp_y: int
    ) -> str:
        return self._request_text(
            "POST",
            "/get_fasta_for_selection",
            payload={
                "fromBpX": int(from_bp_x),
                "fromBpY": int(from_bp_y),
                "toBpX": int(to_bp_x),
                "toBpY": int(to_bp_y),
            },
        )

    def export_agp_for_assembly(self, default_spacer_length: int = 1000) -> str:
        return self._request_text(
            "POST", "/get_agp_for_assembly", payload={"defaultSpacerLength": int(default_spacer_length)}
        )

    def load_agp(self, agp_filename: str) -> Mapping[str, Any]:
        return self._request_json("POST", "/load_agp", payload={"agpFilename": agp_filename})

    # -------------------------
    # Conversion operations
    # -------------------------
    def start_conversion_job(
        self,
        *,
        filename: str,
        direction: str = "mcool-to-hict",
        overwrite: bool = False,
        resolutions: Optional[str] = None,
        compression: Optional[int] = None,
        compression_algorithm: Optional[str] = None,
        chunk_size: Optional[int] = None,
        parallelism: Optional[int] = None,
    ) -> Mapping[str, Any]:
        payload: Dict[str, Any] = {
            "filename": filename,
            "direction": direction,
            "overwrite": bool(overwrite),
        }
        if resolutions is not None:
            payload["resolutions"] = resolutions
        if compression is not None:
            payload["compression"] = int(compression)
        if compression_algorithm is not None:
            payload["compressionAlgorithm"] = compression_algorithm
        if chunk_size is not None:
            payload["chunkSize"] = int(chunk_size)
        if parallelism is not None:
            payload["parallelism"] = int(parallelism)
        return self._request_json("POST", "/convert/jobs", payload=payload)

    def start_batch_conversion_jobs(
        self,
        *,
        files: Sequence[str],
        parallel_jobs: int,
        parallelism: int,
        overwrite: bool = False,
        resolutions: Optional[str] = None,
        compression: Optional[int] = None,
        compression_algorithm: Optional[str] = None,
        chunk_size: Optional[int] = None,
    ) -> Mapping[str, Any]:
        payload: Dict[str, Any] = {
            "files": list(files),
            "parallelJobs": int(parallel_jobs),
            "parallelism": int(parallelism),
            "overwrite": bool(overwrite),
        }
        if resolutions is not None:
            payload["resolutions"] = resolutions
        if compression is not None:
            payload["compression"] = int(compression)
        if compression_algorithm is not None:
            payload["compressionAlgorithm"] = compression_algorithm
        if chunk_size is not None:
            payload["chunkSize"] = int(chunk_size)
        return self._request_json("POST", "/convert/jobs/batch", payload=payload)

    def list_conversion_jobs(self) -> Sequence[Mapping[str, Any]]:
        return list(self._request_json("POST", "/convert/jobs/list", payload={}))

    def get_conversion_job(self, job_id: str) -> Mapping[str, Any]:
        return self._request_json("GET", "/convert/jobs/%s" % job_id)

    def stop_conversion_job(self, job_id: str) -> Mapping[str, Any]:
        return self._request_json("POST", "/convert/jobs/%s/stop" % job_id, payload={})

    def wait_for_conversion_job(
        self,
        job_id: str,
        *,
        poll_interval_sec: float = 1.0,
        timeout_sec: float = 600.0,
    ) -> Mapping[str, Any]:
        deadline = time.monotonic() + timeout_sec
        while True:
            payload = self.get_conversion_job(job_id)
            status = str(payload.get("status", ""))
            if status in {"finished", "failed", "cancelled"}:
                return payload
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for conversion job %s" % job_id)
            time.sleep(poll_interval_sec)

    # -------------------------
    # Misc diagnostics
    # -------------------------
    def worker_diagnostics(self) -> Mapping[str, Any]:
        return self._request_json("POST", "/diagnostics/workers", payload={})
