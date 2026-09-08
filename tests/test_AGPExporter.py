from io import BytesIO
from types import SimpleNamespace

import pytest

from hict.core.AGPProcessor import AGPExporter, AGPparser
from hict.core.FASTAProcessor import FASTAProcessor
from hict.core.common import ContigDirection


def contig(name, length, offset=0):
    return SimpleNamespace(
        contig_name=name, contig_length_at_resolution={0: length},
        offset_inside_fasta_contig=offset, contig_name_in_source_fasta=name,
    )


@pytest.mark.parametrize("gap", [0, 1, 500, 1000, 2345])
def test_export_coordinates_parts_scaffold_reset_and_fasta(gap, tmp_path, monkeypatch):
    monkeypatch.setenv("HICT_EXPORT_GAP_LENGTH_BP", str(gap))
    contigs = [(contig("a", 3), ContigDirection.FORWARD),
               (contig("b", 2), ContigDirection.REVERSED),
               (contig("c", 1), ContigDirection.FORWARD)]
    scaffolds = [(SimpleNamespace(scaffold_name="scf"), SimpleNamespace(start_bp=0, end_bp=5))]
    output = BytesIO()
    AGPExporter().exportAGP(output, contigs, scaffolds)
    rows = [line.split("\t") for line in output.getvalue().decode().splitlines()]
    assert rows[0] == ["scf", "1", "3", "1", "W", "a", "1", "3", "+"]
    assert rows[-2] == ["scf", str(4 + gap), str(5 + gap), str(3 if gap else 2), "W", "b", "1", "2", "-"]
    assert rows[-1] == ["unscaffolded_c", "1", "1", "1", "W", "c", "1", "1", "+"]
    if gap:
        assert rows[1] == ["scf", "4", str(3 + gap), "2", "N", str(gap), "scaffold", "yes", "proximity_ligation"]
    source = tmp_path / "source.fa"
    source.write_text(">a\nACG\n>b\nAC\n>c\nT\n")
    fasta = BytesIO()
    FASTAProcessor(str(source)).get_fasta_for_range(fasta, contigs[:2], "scf")
    sequence = "".join(fasta.getvalue().decode().splitlines()[1:])
    assert sequence == "ACG" + "N" * gap + "GT"
    assert len(sequence) == int(rows[-2][2])
    assert FASTAProcessor(str(source)).get_dna_string_for_multiple_contigs_inside_scaffold(
        SimpleNamespace(spacer_length=500), contigs[:2]) == sequence


def test_parser_accepts_headers_unknown_gaps_and_non_wgs_components(tmp_path):
    source = tmp_path / "test.agp"
    source.write_text("##agp-version 2.1\n"
                      "chr\t1\t100\t1\tU\t100\ttelomere\tno\tna\n"
                      "chr\t101\t110\t2\tF\ta\t1\t10\t?\n")
    parsed = AGPparser(str(source))
    assert len(parsed.getAGPScaffoldRecords()) == 1
    assert parsed.getAGPContigRecords()[0].direction == ContigDirection.FORWARD


def test_short_source_is_rejected_instead_of_silently_dropping_bases(tmp_path):
    source = tmp_path / "short.fa"
    source.write_text(">a\nAC\n")
    with pytest.raises(ValueError, match="does not cover"):
        FASTAProcessor(str(source)).get_cropped_dna_string_for_single_contig(contig("a", 3), ContigDirection.FORWARD)
