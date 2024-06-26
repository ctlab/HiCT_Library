#  MIT License
#
#  Copyright (c) 2021-2024. Aleksandr Serdiukov, Anton Zamyatin, Aleksandr Sinitsyn, Vitalii Dravgelis and Computer Technologies Laboratory ITMO University team.
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

#  MIT License
#
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#
from typing import Tuple, NamedTuple, List, Dict, Optional, Union
from hict.core.common import ContigDescriptor, ContigDirection, ScaffoldBordersBP, ScaffoldDescriptor
import numpy as np


class AGPScaffoldRecord(NamedTuple):
    name: str
    start_ctg: str
    end_ctg: str


class AGPContigRecord(NamedTuple):
    name: str
    direction: ContigDirection
    start_position: int
    end_position: int


class AGPparser(object):
    def __init__(
        self,
        filename: str,
    ) -> None:
        self.contig_records_list: List[AGPContigRecord] = list()
        self.scaffold_records_list: List[AGPScaffoldRecord] = list()
        self.parseAGP(filename)

    def parseAGPLine(self, line: str) -> Tuple[str, str, str, int, int]:
        toks: List[str] = line.split()
        if toks[4] == 'N':
            gap_len: str = toks[5]
            return ('N_spacer', gap_len, '', 0, 0)
        elif toks[4] == 'W':
            seq_object_name: str = toks[0]
            component_name: str = toks[5]
            component_direction: str = toks[8]
            component_beg: int = int(toks[6])
            component_end: int = int(toks[7])
            return (seq_object_name, component_name, component_direction, component_beg, component_end)
        else:
            raise Exception(
                f'unexpected symbol in agp component_type column: {toks[4]}')

    def parseAGP(self, filename) -> None:
        with open(filename, 'r') as agp_file:
            scaf_name: str
            cur_scaf_name: str
            start_ctg: str
            end_ctg: str
            ctg_name: str
            ctg_dir: ContigDirection
            ctg_start_position: int
            ctg_end_position: int
            for i, line in enumerate(agp_file):
                scaf_name, ctg_name, ctg_dir_str, ctg_start_position, ctg_end_position = self.parseAGPLine(line)
                if scaf_name == 'N_spacer':
                    continue
                if ctg_dir_str not in ("+", "-"):
                    raise RuntimeError(
                        f'unexpected symbol in agp direction column: {ctg_dir}'
                    )
                ctg_dir = ContigDirection(
                    1) if ctg_dir_str == '+' else ContigDirection(0)
                self.contig_records_list.append(
                    AGPContigRecord(ctg_name, ctg_dir, ctg_start_position, ctg_end_position))
                if i == 0:
                    cur_scaf_name = scaf_name
                    start_ctg = ctg_name
                    end_ctg = ctg_name
                else:
                    if scaf_name == cur_scaf_name:
                        end_ctg = ctg_name
                    else:
                        self.scaffold_records_list.append(
                            AGPScaffoldRecord(cur_scaf_name, start_ctg, end_ctg))
                        cur_scaf_name = scaf_name
                        start_ctg = ctg_name
                        end_ctg = ctg_name
            self.scaffold_records_list.append(
                AGPScaffoldRecord(cur_scaf_name, start_ctg, end_ctg))

    def getAGPContigRecords(self) -> List[AGPContigRecord]:
        return self.contig_records_list

    def getAGPScaffoldRecords(self) -> List[AGPScaffoldRecord]:
        return self.scaffold_records_list


class AGPExporter(object):

    def exportAGP(
        self,
        writableStream,
        ordered_contig_descriptors: List[
            Tuple[
                ContigDescriptor,
                ContigDirection,
                # Dict[np.int64, Tuple[np.int64, np.int64]]
            ]
        ],
        scaffold_list: List[Tuple[ScaffoldDescriptor, ScaffoldBordersBP]],
        intercontig_spacer: str = 500*'N'
    ) -> None:
        agpString: str = ""
        prev_scaffold: str = ""
        prev_end: np.int64 = 0
        component_id: int = 1

        # contig_lengths: np.ndarray = np.zeros(shape=len(ordered_contig_descriptors), dtype=np.int64)
        # for i, cdt in enumerate(ordered_contig_descriptors):
        #     contig_lengths[i] = cdt[0].contig_length_at_resolution[0]

        # ord_contig_length_prefix_sum = np.cumsum(contig_lengths)

        position_bp: np.int64 = np.int64(0)
        position_in_scaffold_list = 0

        for contig, contig_direction in ordered_contig_descriptors:
            while position_in_scaffold_list < len(scaffold_list) and scaffold_list[position_in_scaffold_list][1].end_bp <= position_bp:
                position_in_scaffold_list += 1

            current_scaffold: str
            if position_in_scaffold_list < len(scaffold_list) and scaffold_list[position_in_scaffold_list][1].start_bp <= position_bp < scaffold_list[position_in_scaffold_list][1].end_bp:
                current_scaffold = scaffold_list[position_in_scaffold_list][0].scaffold_name
            else:
                current_scaffold = f"unscaffolded_{contig.contig_name}"

            contig_name: str = contig.contig_name
            contig_length_bp: np.int64 = contig.contig_length_at_resolution[np.int64(
                0)]
            dir_cond: bool = contig_direction == ContigDirection.FORWARD
            contig_direction_str = "+" if dir_cond else "-"
            if current_scaffold == prev_scaffold:
                component_id += 1
                agpString += "\t".join(
                    map(
                        str,
                        [
                            current_scaffold,
                            prev_end + 1,
                            prev_end +
                            len(intercontig_spacer),
                            component_id,
                            "N", len(intercontig_spacer),
                            "scaffold", "yes",
                            "proximity_ligation"
                        ]
                    )
                )
                prev_end = prev_end + len(intercontig_spacer) - 1
                agpString += '\n'
                component_id += 1
            else:
                component_id = 1
            agpString += "\t".join(
                map(
                    lambda e: str(e),
                    (
                        current_scaffold,
                        prev_end + 1,
                        prev_end + contig_length_bp - 1,
                        component_id,
                        "W", contig_name,
                        1, contig_length_bp,
                        contig_direction_str
                    )
                )
            )
            prev_end = prev_end + contig_length_bp - 1
            prev_scaffold = current_scaffold
            agpString += '\n'
            position_bp += contig_length_bp
        out_record: bytes = agpString.encode(encoding='utf-8')
        writableStream.write(out_record)
