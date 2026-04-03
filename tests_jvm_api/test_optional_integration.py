import os

import pytest

from hict_jvm_api.client import HiCTJVMClient
from hict_jvm_api.models import Unit

BASE_URL = os.getenv("HICT_JVM_API_BASE_URL", "").strip()
DATASET_FILE = os.getenv("HICT_DATASET_FILE", "").strip()
FASTA_FILE = os.getenv("HICT_FASTA_FILE", "").strip()
AGP_FILE = os.getenv("HICT_AGP_FILE", "").strip()
ALLOW_MUTATION = os.getenv("HICT_JVM_API_ALLOW_MUTATION", "false").strip().lower() in {"1", "true", "yes"}

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def client():
    if not BASE_URL:
        pytest.skip("Set HICT_JVM_API_BASE_URL to run optional integration tests")
    c = HiCTJVMClient(BASE_URL)
    if DATASET_FILE:
        c.open_file(DATASET_FILE)
    yield c
    try:
        c.close_session()
    except Exception:
        pass
    c.close()


def test_open_and_fetch_region(client: HiCTJVMClient) -> None:
    if not DATASET_FILE:
        pytest.skip("Set HICT_DATASET_FILE to run dataset integration tests")
    opened = client.require_open_file_response()
    coarse_resolution = opened.resolutions[0]

    image = client.fetch_region_pixels(
        start_row_px=0,
        start_col_px=0,
        rows=32,
        cols=32,
        bp_resolution=coarse_resolution,
    )
    assert image.shape == (32, 32, 4)

    converter = client.unit_converter(coarse_resolution)
    assert converter.convert(0, Unit.BP, Unit.BINS) == 0


def test_open_and_fetch_numeric_signal_region(client: HiCTJVMClient) -> None:
    if not DATASET_FILE:
        pytest.skip("Set HICT_DATASET_FILE to run dataset integration tests")
    opened = client.require_open_file_response()
    coarse_resolution = opened.resolutions[0]

    matrix = client.fetch_region_signal(
        start_row=0,
        start_col=0,
        rows=32,
        cols=32,
        bp_resolution=coarse_resolution,
        unit=Unit.PIXELS,
        signal_mode="TRADITIONAL_NORMALIZED",
        dtype="float32",
    )
    assert matrix.shape == (32, 32)


def test_scaffolding_reverse_twice(client: HiCTJVMClient) -> None:
    if not DATASET_FILE:
        pytest.skip("Set HICT_DATASET_FILE to run dataset integration tests")
    if not ALLOW_MUTATION:
        pytest.skip("Set HICT_JVM_API_ALLOW_MUTATION=true to run mutating scaffolding tests")

    opened = client.require_open_file_response()
    coarse_resolution = opened.resolutions[0]
    total_bp = client.unit_converter(coarse_resolution).total_bp
    if total_bp < 4:
        pytest.skip("Dataset is too small for reverse operation test")

    start_bp = 0
    end_bp = min(total_bp - 1, max(2, total_bp // 100))

    first = client.reverse_selection_range(start_bp, end_bp)
    second = client.reverse_selection_range(start_bp, end_bp)
    assert first["version"] < second["version"]


def test_fasta_link_and_export(client: HiCTJVMClient) -> None:
    if not DATASET_FILE:
        pytest.skip("Set HICT_DATASET_FILE to run dataset integration tests")
    if not FASTA_FILE:
        pytest.skip("Set HICT_FASTA_FILE to test FASTA integration")

    link_result = client.link_fasta(FASTA_FILE, allow_mismatch=True)
    assert "linked" in link_result or "warnings" in link_result

    fasta_text = client.export_fasta_for_selection(0, 0, 1_000, 1_000)
    assert isinstance(fasta_text, str)


def test_agp_load_and_export(client: HiCTJVMClient) -> None:
    if not DATASET_FILE:
        pytest.skip("Set HICT_DATASET_FILE to run dataset integration tests")
    if not AGP_FILE:
        pytest.skip("Set HICT_AGP_FILE to test AGP integration")

    _ = client.load_agp(AGP_FILE)
    agp_text = client.export_agp_for_assembly(default_spacer_length=1000)
    assert isinstance(agp_text, str)
    assert len(agp_text) > 0
