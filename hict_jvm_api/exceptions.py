"""Exceptions for the HiCT JVM API client."""

from typing import Optional


class HiCTAPIError(RuntimeError):
    """Raised when the HiCT_JVM API returns an HTTP or API-level error."""

    def __init__(self, message: str, status_code: Optional[int] = None, payload: Optional[object] = None):
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload


class HiCTClientStateError(RuntimeError):
    """Raised when a client operation requires an opened session but none is available."""


class HiddenCoordinateError(ValueError):
    """Raised when converting to PIXELS for coordinates located in hidden contigs."""
