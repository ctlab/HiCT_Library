from hict_jvm_api.exceptions import HiddenCoordinateError
from hict_jvm_api.models import AssemblyInfo, ContigDescriptor, OpenFileResponse
from hict_jvm_api.units import UnitConverter


def _open_response_with_hidden_middle_contig() -> OpenFileResponse:
    contigs = [
        ContigDescriptor(
            contig_id=1,
            contig_name="c1",
            contig_original_name="c1",
            contig_source_name="c1",
            contig_offset_in_source=0,
            contig_direction=0,
            contig_length_bp=100,
            contig_length_bins={10: 10},
            contig_presence_at_resolution={10: 1},
        ),
        ContigDescriptor(
            contig_id=2,
            contig_name="c2",
            contig_original_name="c2",
            contig_source_name="c2",
            contig_offset_in_source=0,
            contig_direction=0,
            contig_length_bp=50,
            contig_length_bins={10: 5},
            contig_presence_at_resolution={10: 0},
        ),
        ContigDescriptor(
            contig_id=3,
            contig_name="c3",
            contig_original_name="c3",
            contig_source_name="c3",
            contig_offset_in_source=0,
            contig_direction=0,
            contig_length_bp=100,
            contig_length_bins={10: 10},
            contig_presence_at_resolution={10: 1},
        ),
    ]
    return OpenFileResponse(
        status="Opened",
        dtype="uint8",
        resolutions=[10],
        pixel_resolutions=[1.0],
        tile_size=256,
        assembly_info=AssemblyInfo(contigs=contigs, scaffolds=[]),
        matrix_sizes_bins=[25],
    )


def test_converter_total_lengths_and_visibility() -> None:
    converter = UnitConverter.from_open_file_response(_open_response_with_hidden_middle_contig(), bp_resolution=10)
    assert converter.total_bp == 250
    assert converter.total_bins == 25
    assert converter.total_visible_pixels == 20


def test_converter_bp_bin_pixel_roundtrip_on_shown_contig() -> None:
    converter = UnitConverter.from_open_file_response(_open_response_with_hidden_middle_contig(), bp_resolution=10)

    # c3 starts at bp=150 and bin=15, but visible pixels continue from c1 (10 pixels).
    assert converter.bp_to_bins(200) == 20
    assert converter.bins_to_bp(20) == 200
    assert converter.bins_to_pixels(20) == 15
    assert converter.pixels_to_bins(15) == 20
    assert converter.pixels_to_bp(15) == 200


def test_converter_hidden_contig_raises_for_pixel_conversion() -> None:
    converter = UnitConverter.from_open_file_response(_open_response_with_hidden_middle_contig(), bp_resolution=10)

    # bp=120 belongs to hidden contig c2.
    hidden_bin = converter.bp_to_bins(120)
    assert hidden_bin == 12

    try:
        converter.bins_to_pixels(hidden_bin)
        assert False, "Expected HiddenCoordinateError"
    except HiddenCoordinateError:
        pass
