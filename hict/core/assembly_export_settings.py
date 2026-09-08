import os


def output_gap_length_bp() -> int:
    """Gap policy shared with the JVM launcher and AGP/FASTA exporters."""
    value = os.environ.get("HICT_EXPORT_GAP_LENGTH_BP", "1000").strip() or "1000"
    if not value.isascii() or not value.isdecimal():
        raise ValueError("HICT_EXPORT_GAP_LENGTH_BP must be a non-negative integer in base pairs")
    return int(value)
