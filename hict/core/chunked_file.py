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
import gc
import threading
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Iterable, Set, Union
import multiprocessing
import multiprocessing.managers
import copy

import h5py
from matplotlib.pyplot import sca
import numpy as np
# from cachetools import LRUCache, cachedmethod
# from cachetools.keys import hashkey
from readerwriterlock import rwlock
import scipy
from scipy.sparse import coo_array, csr_array, csc_array
from hict.core.scaffold_tree import ScaffoldTree

from hict.core.AGPProcessor import *
from hict.core.FASTAProcessor import FASTAProcessor
from hict.core.common import ATUDescriptor, ATUDirection, ScaffoldBordersBP, StripeDescriptor, ContigDescriptor, ScaffoldDescriptor, \
    FinalizeRecordType, ContigHideType, QueryLengthUnit
from hict.core.contig_tree import ContigTree
# from hict.core.stripe_tree import StripeTree
from hict.util.h5helpers import *


additional_dataset_creation_args = {
    'compression': 'lzf',
    'shuffle': True,
    'chunks': True,
}


def constrain_coordinate(x_bins: Union[np.int64, int], lower: Union[np.int64, int],
                         upper: Union[np.int64, int]) -> np.int64:
    return max(min(x_bins, upper), lower)


# BLOCK_CACHE_SIZE: int = 1024


class ChunkedFile(object):
    class FileState(Enum):
        CLOSED = 0
        OPENED = 1
        INCORRECT = 2

    def __init__(
            self,
            filepath: Union[Path, str],
            block_cache_size: int = 64,
            multithreading_pool_size: int = 8,
            mp_manager: Optional[multiprocessing.managers.SyncManager] = None
    ) -> None:
        super().__init__()
        self.filepath: Path = Path(filepath).absolute()
        self.stripes: Dict[np.int64, List[StripeDescriptor]] = dict()
        self.atl: Dict[np.int64, List[ATUDescriptor]] = dict()
        self.contig_names: List[str] = []
        self.contig_name_to_contig_id: Dict[str, np.int64] = dict()
        self.contig_lengths_bp: Dict[np.int64, np.int64] = dict()
        self.resolutions: List[np.int64] = []
        self.contig_tree: Optional[ContigTree] = None
        self.state: ChunkedFile.FileState = ChunkedFile.FileState.CLOSED
        self.dense_submatrix_size: Dict[np.int64,
                                        np.int64] = dict()  # Resolution -> MSS
        self.block_cache_size = block_cache_size
        # self.block_cache = LRUCache(maxsize=self.block_cache_size)
        # self.block_intersection_cache = LRUCache(maxsize=self.block_cache_size)
        # self.block_cache_lock: Lock = threading.Lock()
        # self.block_intersection_cache_lock: Lock = threading.Lock()
        self.dtype: Optional[np.dtype] = None
        self.mp_manager = mp_manager
        if mp_manager is not None:
            lock_factory = mp_manager.RLock
        else:
            lock_factory = threading.RLock
        self.hdf_file_lock: rwlock.RWLockWrite = rwlock.RWLockWrite(
            lock_factory=lock_factory)
        
        self.fasta_processor: Optional[FASTAProcessor] = None
        self.fasta_file_lock: rwlock.RWLockFair = rwlock.RWLockFair(
            lock_factory=lock_factory)
        self.multithreading_pool_size = multithreading_pool_size
        self.scaffold_tree: Optional[ScaffoldTree] = None
        self.contig_id_to_contig_descriptor: Dict[np.int64, ContigDescriptor] = dict(
        )

    def open(self) -> None:
        # NOTE: When file is opened in this method, we assert no one writes to it
        self.opened_hdf_file: h5py.File = h5py.File(
            self.filepath,
            mode='r',
            swmr=True,
            libver='latest'
        )        
        contig_id_to_length_by_resolution: Dict[np.int64,
                                                Dict[np.int64, np.int64]] = dict()
        contig_id_to_hide_type_by_resolution: Dict[np.int64,
                                                   Dict[np.int64, ContigHideType]] = dict()
        contig_id_to_direction: List[ContigDirection] = []
        contig_id_to_scaffold_id: List[Optional[np.int64]] = []
        ordered_contig_ids: np.ndarray
        with self.hdf_file_lock.gen_rlock():
            f = self.opened_hdf_file
            resolutions = np.array(
                [np.int64(sdn) for sdn in sorted(
                    filter(lambda s: s.isnumeric(), f['resolutions'].keys()))],
                dtype=np.int64
            )
            self.resolutions = resolutions
            self.dtype = f[f'resolutions/{str(max(resolutions))}/treap_coo/block_vals'].dtype

            for resolution in resolutions:
                (
                    self.stripes[resolution],
                    self.dense_submatrix_size[resolution]
                ) = self.read_stripe_data(f, resolution)

            self.atl = self.read_atl(f)

            (
                contig_id_to_contig_length_bp,
                resolution_to_contig_length_bins,
                resolution_to_contig_hide_type,
                contig_id_to_atus,
                contig_names
            ) = self.read_contig_data(f)
            self.contig_names = contig_names
            for contig_id, contig_name in enumerate(contig_names):
                self.contig_name_to_contig_id[contig_name] = contig_id

            contig_count: np.int64 = len(contig_names)
            for contig_id in range(0, contig_count):
                contig_id_to_length_by_resolution[contig_id] = dict()
                contig_id_to_hide_type_by_resolution[contig_id] = dict()

            for resolution in resolutions:
                contig_id_to_contig_length_bins_at_resolution = resolution_to_contig_length_bins[
                    resolution]
                contig_id_to_contig_hide_type_at_resolution = resolution_to_contig_hide_type[
                    resolution]
                for contig_id, (
                        contig_length_bins_at_resolution,
                        contig_hide_type_at_resolution
                ) in enumerate(zip(
                    contig_id_to_contig_length_bins_at_resolution,
                    contig_id_to_contig_hide_type_at_resolution
                )):
                    contig_id_to_length_by_resolution[contig_id][resolution] = (
                        contig_length_bins_at_resolution
                    )
                    contig_id_to_hide_type_by_resolution[contig_id][resolution] = ContigHideType(
                        contig_hide_type_at_resolution
                    )

            contig_info_group: h5py.Group = f['/contig_info/']
            ordered_contig_ids: h5py.Dataset = contig_info_group['ordered_contig_ids']
            contig_direction_ds: h5py.Dataset = contig_info_group['contig_direction']
            contig_scaffold_ids: h5py.Dataset = contig_info_group['contig_scaffold_id']

            for (
                    contig_direction,
                    contig_scaff_id
            ) in zip(
                contig_direction_ds,
                contig_scaffold_ids
            ):
                contig_id_to_direction.append(
                    ContigDirection(contig_direction))
                contig_id_to_scaffold_id.append(
                    contig_scaff_id if contig_scaff_id >= 0 else None)

            self.contig_tree = ContigTree(self.resolutions)

            contig_id_to_contig_descriptor: List[ContigDescriptor] = []

            for (
                    contig_id,
                    resolution_to_contig_length
            ) in contig_id_to_length_by_resolution.items():
                contig_presence_at_resolution: Dict[
                    np.int64,
                    ContigHideType
                ] = contig_id_to_hide_type_by_resolution[contig_id]
                contig_presence_at_resolution[0] = ContigHideType.FORCED_SHOWN

                # Hide small contigs only at zoomed resolutions
                for res in resolutions[1:]:
                    if contig_id_to_contig_length_bp[contig_id] < res:
                        contig_presence_at_resolution[res] = ContigHideType.AUTO_HIDDEN
                contig_descriptor: ContigDescriptor = ContigDescriptor.make_contig_descriptor(
                    contig_id=contig_id,
                    contig_name=contig_names[contig_id],
                    # direction=contig_id_to_direction[contig_id],
                    contig_length_bp=contig_id_to_contig_length_bp[contig_id],
                    contig_length_at_resolution=resolution_to_contig_length,
                    contig_presence_in_resolution=contig_presence_at_resolution,
                    # {resolution: list(map(lambda ati: self.atl[resolution][ati], contig_id_to_atus[contig_id][resolution])) for resolution in resolutions},
                    atus=contig_id_to_atus[contig_id],
                    # scaffold_id=contig_id_to_scaffold_id[contig_id]
                )
                contig_id_to_contig_descriptor.append(contig_descriptor)

            for contig_id in ordered_contig_ids:
                contig_descriptor = contig_id_to_contig_descriptor[contig_id]
                self.contig_id_to_contig_descriptor[contig_id] = contig_descriptor
                self.contig_tree.insert_at_position(
                    contig_descriptor,
                    self.contig_tree.get_node_count(),
                    direction=contig_id_to_direction[contig_id],
                )
            # self.contig_tree.update_tree()
            total_assembly_length_bp = self.contig_tree.root.get_sizes()[0][0]
            self.scaffold_tree = ScaffoldTree(
                total_assembly_length_bp, self.mp_manager)
            # self.restore_scaffolds(f)

        self.state = ChunkedFile.FileState.OPENED

    def read_atl(
        self,
        f: h5py.File
    ) -> Dict[np.int64, List[ATUDescriptor]]:
        resolution_atus: Dict[np.int64, List[ATUDescriptor]] = dict()

        for resolution in self.resolutions:
            atl_group: h5py.Group = f[f'/resolutions/{resolution}/atl']
            basis_atu: h5py.Dataset = atl_group['basis_atu']

            atus = [
                ATUDescriptor.make_atu_descriptor(
                    stripe_descriptor=self.stripes[resolution][row[0]],
                    start_index_in_stripe_incl=row[1],
                    end_index_in_stripe_excl=row[2],
                    direction=ATUDirection(row[3])
                ) for row in basis_atu
            ]

            resolution_atus[resolution] = atus

        return resolution_atus

    def read_contig_data(
            self,
            f: h5py.File
    ) -> Tuple[
        np.ndarray,
        Dict[np.int64, np.ndarray],
        Dict[np.int64, np.ndarray],
        List[Dict[np.int64, List[ATUDescriptor]]],
        List[str]
    ]:
        contig_info_group: h5py.Group = f['/contig_info/']
        contig_names_ds: h5py.Dataset = contig_info_group['contig_name']
        contig_lengths_bp: h5py.Dataset = contig_info_group['contig_length_bp']

        contig_count: np.int64 = np.int64(len(contig_names_ds))

        assert len(
            contig_lengths_bp) == contig_count, "Different contig count in different datasets??"

        # Resolution -> [ContigId -> ContigLengthBp]
        resolution_to_contig_length_bins: Dict[np.int64, np.ndarray] = dict()
        # Resolution -> [ContigId -> ContigHideType]
        resolution_to_contig_hide_type: Dict[np.int64, np.ndarray] = dict()
        # resolution_to_contig_atus: Dict[np.int64, List[List[ATUDescriptor]]] = dict()
        contig_id_to_atus: List[Dict[np.int64, List[ATUDescriptor]]] = [
            {resolution: [] for resolution in self.resolutions} for _ in range(contig_count)]
        for resolution in self.resolutions:
            contigs_group: h5py.Group = f[f'/resolutions/{resolution}/contigs/']
            contig_length_bins_ds: h5py.Dataset = contigs_group['contig_length_bins']
            contig_hide_type_ds: h5py.Dataset = contigs_group['contig_hide_type']
            contig_atus: h5py.Dataset = contigs_group['atl']

            assert len(
                contig_length_bins_ds) == contig_count, "Different contig count in different datasets??"

            for contig_id, basis_atu_id in contig_atus:
                contig_id_to_atus[contig_id][resolution].append(
                    self.atl[resolution][basis_atu_id])

            resolution_to_contig_length_bins[resolution] = np.array(
                contig_length_bins_ds[:].astype(np.int64),
                dtype=np.int64
            )

            resolution_to_contig_hide_type[resolution] = np.array(
                contig_hide_type_ds[:].astype(np.int64),
                dtype=np.int8
            )

        contig_id_to_contig_length_bp: np.ndarray = np.array(
            contig_lengths_bp[:].astype(np.int64), dtype=np.int64)
        contig_names: List[str] = [bytes(contig_name).decode(
            'utf-8') for contig_name in contig_names_ds]

        return (
            contig_id_to_contig_length_bp,
            resolution_to_contig_length_bins,
            resolution_to_contig_hide_type,
            contig_id_to_atus,
            contig_names
        )

    def read_stripe_data(
            self,
            f: h5py.File,
            resolution: np.int64
    ) -> Tuple[
        List[StripeDescriptor],
        np.int64
    ]:
        stripes_group: h5py.Group = f[f'/resolutions/{resolution}/stripes']
        stripe_lengths_bins: h5py.Dataset = stripes_group['stripe_length_bins']
        stripes_bin_weights: Optional[h5py.Dataset] = (
            stripes_group['stripes_bin_weights']
        ) if 'stripes_bin_weights' in stripes_group.keys() else None

        stripes: List[StripeDescriptor] = [
            StripeDescriptor.make_stripe_descriptor(
                np.int64(stripe_id),
                stripe_length_bins,
                np.array(
                    np.nan_to_num(
                        stripes_bin_weights[stripe_id, :stripe_length_bins], copy=False),
                    dtype=np.float64
                ) if stripes_bin_weights is not None else np.ones(stripe_length_bins, dtype=np.float64)
            ) for (
                stripe_id, stripe_length_bins
            ) in enumerate(stripe_lengths_bins)
        ]

        dense_submatrix_size: np.int64 = max(stripe_lengths_bins)

        return stripes, dense_submatrix_size

    def sparse_to_dense(self, sparse_mx: Union[coo_array, csr_array, csc_array]) -> np.ndarray:
        return sparse_mx.todense()

    def process_flips(
        self,
        mx_as_array: np.ndarray,
        row_atu: ATUDescriptor,
        col_atu: ATUDescriptor
    ) -> np.ndarray:
        if row_atu.direction == ATUDirection.REVERSED:
            mx_as_array = np.flip(mx_as_array, axis=0)
        if col_atu.direction == ATUDirection.REVERSED:
            mx_as_array = np.flip(mx_as_array, axis=1)
        return mx_as_array

    def get_stripe_intersection_for_atus_as_raw_dense_matrix(
            self,
            resolution: np.int64,
            row_atu: ATUDescriptor,
            col_atu: ATUDescriptor
    ) -> np.ndarray:
        row_stripe: StripeDescriptor = row_atu.stripe_descriptor
        col_stripe: StripeDescriptor = col_atu.stripe_descriptor
        needs_transpose: bool = False
        if row_stripe.stripe_id > col_stripe.stripe_id:
            row_stripe, col_stripe = col_stripe, row_stripe
            needs_transpose = True

        mx_as_array: np.ndarray
        is_empty: bool

        r: np.int64 = row_stripe.stripe_id
        c: np.int64 = col_stripe.stripe_id

        with self.hdf_file_lock.gen_rlock():
            blocks_dir: h5py.Group = self.opened_hdf_file[
                f'/resolutions/{resolution}/treap_coo']
            stripes_count: np.int64 = blocks_dir.attrs['stripes_count']
            block_index_in_datasets: np.int64 = r * stripes_count + c

            block_lengths: h5py.Dataset = blocks_dir['block_length']
            block_length = block_lengths[block_index_in_datasets]
            is_empty = (block_length == 0)

            if is_empty:
                mx_as_array = np.zeros(
                    shape=(
                        row_atu.end_index_in_stripe_excl - row_atu.start_index_in_stripe_incl,
                        col_atu.end_index_in_stripe_excl - col_atu.start_index_in_stripe_incl
                    ),
                    dtype=self.dtype
                )
            else:
                block_offsets: h5py.Dataset = blocks_dir['block_offset']
                block_offset = block_offsets[block_index_in_datasets]
                is_dense: bool = (block_offset < 0)

                if is_dense:
                    dense_blocks: h5py.Dataset = blocks_dir['dense_blocks']
                    index_in_dense_blocks: np.int64 = -(block_offset + 1)
                    mx_as_array = dense_blocks[index_in_dense_blocks, 0, :, :]
                else:
                    block_vals: h5py.Dataset = blocks_dir['block_vals']
                    block_finish = block_offset + block_length
                    block_rows: h5py.Dataset = blocks_dir['block_rows']
                    block_cols: h5py.Dataset = blocks_dir['block_cols']
                    mx = coo_array(
                        (
                            block_vals[block_offset:block_finish],
                            (
                                block_rows[block_offset:block_finish],
                                block_cols[block_offset:block_finish]
                            )
                        ),
                        shape=(row_stripe.stripe_length_bins,
                               col_stripe.stripe_length_bins)
                    )
                    mx_as_array = self.sparse_to_dense(mx)

                if row_atu.stripe_descriptor.stripe_id == col_atu.stripe_descriptor.stripe_id:
                    assert (
                        row_atu.stripe_descriptor == col_atu.stripe_descriptor
                    ), "Fetched stripe descriptors have the same ids, but are not equal??"
                    mx_as_array = np.where(
                        mx_as_array, mx_as_array, mx_as_array.T)

                if needs_transpose:
                    mx_as_array = mx_as_array.T

                mx_as_array = mx_as_array[
                    row_atu.start_index_in_stripe_incl:row_atu.end_index_in_stripe_excl,
                    col_atu.start_index_in_stripe_incl:col_atu.end_index_in_stripe_excl,
                ]

                mx_as_array = self.process_flips(mx_as_array, row_atu, col_atu)

        return mx_as_array

    def get_submatrix(
            self,
            resolution: np.int64,
            start_row_incl: np.int64,
            start_col_incl: np.int64,
            end_row_excl: np.int64,
            end_col_excl: np.int64,
            exclude_hidden_contigs: bool
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert (
            self.contig_tree is not None
        ), "Contig tree is not present?"
        total_assembly_length = self.contig_tree.get_sizes(
        )[2 if exclude_hidden_contigs else 0][resolution]

        start_row_incl = constrain_coordinate(
            start_row_incl, 0, total_assembly_length)
        end_row_excl = constrain_coordinate(
            end_row_excl, 0, total_assembly_length)
        start_col_incl = constrain_coordinate(
            start_col_incl, 0, total_assembly_length)
        end_col_excl = constrain_coordinate(
            end_col_excl, 0, total_assembly_length)

        row_atus: List[ATUDescriptor] = self.get_atus_for_range(
            resolution,
            start_row_incl,
            end_row_excl,
            exclude_hidden_contigs
        )
        col_atus: List[ATUDescriptor] = self.get_atus_for_range(
            resolution,
            start_col_incl,
            end_col_excl,
            exclude_hidden_contigs
        )

        query_rows_count = end_row_excl - start_row_incl
        query_cols_count = end_col_excl - start_col_incl

        if start_row_incl < end_row_excl and 0 <= start_row_incl < total_assembly_length:
            assert (
                len(row_atus) > 0
            ), "Query is correct but no rows were found??"

        if start_col_incl < end_col_excl and 0 <= start_col_incl < total_assembly_length:
            assert (
                len(col_atus) > 0
            ), "Query is correct but no columns were found??"

        row_matrices: List[np.ndarray] = []
        row_subweights: List[np.ndarray] = []

        row_subtotals: Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        for row_atu in row_atus:
            def load_intersection(col_atu: ATUDescriptor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
                return self.get_atu_intersection(
                    resolution=resolution,
                    row_atu=row_atu,
                    col_atu=col_atu
                )
            # with Pool(processes=self.multithreading_pool_size) as P:
                # row_subtotals = P.map(load_intersection, col_atus)
            row_subtotals = list(map(load_intersection, col_atus))
            row_submatrices: List[np.ndarray] = [t[0] for t in row_subtotals]
            if len(col_atus) > 0:
                assert (
                    len(row_subtotals) > 0
                ), "There were ATUs but no intersection??"
                assert all(
                    (
                        sbm.shape[0] == row_submatrices[0].shape[0]
                        for sbm in row_submatrices
                    )
                ), "Not all submatrices in row have the same row count??"
                assert (
                    row_submatrices[0].shape[0] == (
                        row_atu.end_index_in_stripe_excl - row_atu.start_index_in_stripe_incl
                    )
                ), "Row height is not equal to what ATU describes??"
                row = (
                    np.hstack(row_submatrices)
                )
                row_subweights.append(row_subtotals[0][1])
            else:
                assert (
                    query_cols_count <= 0
                ), "No column ATUs are present, but query is non-trivial for columns??"
                row = np.zeros(shape=(
                    row_atu.end_index_in_stripe_excl - row_atu.start_index_in_stripe_incl, 0))
            row_matrices.append(row)

        if len(row_subweights) > 0:
            row_weights = np.hstack(row_subweights)
        else:
            row_weights = np.ones(shape=max(np.int64(0), query_rows_count))

        col_subweights = [t[2] for t in row_subtotals]
        if len(col_subweights) > 0:
            col_weights = np.hstack(col_subweights)
        else:
            col_weights = np.ones(shape=max(np.int64(0), query_cols_count))

        if query_rows_count > 0 and query_cols_count > 0:
            result = np.vstack(row_matrices)
            assert (
                len(row_subweights) > 0
            ), "No row weights were fetched, but query is non-trivial for rows??"
            assert (
                len(col_subweights) > 0
            ), "No column weights were fetched, but query is non-trivial for columns??"
        else:
            result = np.zeros(
                shape=(max(0, query_rows_count), max(0, query_cols_count)))

        assert (
            result.shape[0] == (end_row_excl-start_row_incl)
        ), "Row count is not as queried??"

        assert (
            result.shape[1] == (end_col_excl-start_col_incl)
        ), "Column count is not as queried??"

        assert (
            len(row_weights) == (end_row_excl-start_row_incl)
        ), "Row weights count is not as queried??"

        assert (
            len(col_weights) == (end_col_excl-start_col_incl)
        ), "Column weights count is not as queried??"

        return result, row_weights, col_weights

    def get_atu_intersection(
        self,
        resolution: np.int64,
        row_atu: ATUDescriptor,
        col_atu: ATUDescriptor,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        atu_intersection_dense: np.ndarray = self.get_stripe_intersection_for_atus_as_raw_dense_matrix(
            resolution,
            row_atu,
            col_atu
        )

        row_weights = row_atu.stripe_descriptor.bin_weights[
            int(row_atu.start_index_in_stripe_incl):int(row_atu.end_index_in_stripe_excl)]
        col_weights = col_atu.stripe_descriptor.bin_weights[
            int(col_atu.start_index_in_stripe_incl):int(col_atu.end_index_in_stripe_excl)]
        if row_atu.direction == ATUDirection.REVERSED:
            row_weights = np.flip(row_weights)
        if col_atu.direction == ATUDirection.REVERSED:
            col_weights = np.flip(col_weights)

        return atu_intersection_dense, row_weights, col_weights

    def get_atus_for_range(
        self,
        resolution: np.int64,
        start_px_incl: np.int64,
        end_px_excl: np.int64,
        exclude_hidden_contigs: bool,
    ) -> List[ATUDescriptor]:
        assert (
            self.state == ChunkedFile.FileState.OPENED and self.contig_tree is not None
        ), "File must be opened for reading ATUs"

        total_assembly_length = self.contig_tree.get_sizes(
        )[2 if exclude_hidden_contigs else 0][resolution]
        start_px_incl = constrain_coordinate(
            start_px_incl, 0, total_assembly_length)
        end_px_excl = constrain_coordinate(
            end_px_excl, 0, total_assembly_length)

        es: ContigTree.ExposedSegment = self.contig_tree.expose_segment(
            resolution,
            # 1+start_px_incl,
            start_px_incl,
            end_px_excl,
            units=QueryLengthUnit.PIXELS if exclude_hidden_contigs else QueryLengthUnit.BINS
        )

        result_atus: List[ATUDescriptor]

        query_length: np.int64 = end_px_excl - start_px_incl
        if query_length <= 0:
            return []

        if es.segment is None:
            assert query_length <= 0, "Query is not zero-length, but no ATUs were found?"
            result_atus = []
        else:
            # TODO: maybe no update_sizes
            segment_size = es.segment.get_sizes(
            )[2 if exclude_hidden_contigs else 0][resolution]
            less_size: np.int64
            if es.less is not None:
                less_size = es.less.get_sizes(
                )[2 if exclude_hidden_contigs else 0][resolution]
            else:
                less_size = np.int64(0)

            delta_px_between_segment_first_contig_start_and_query_start: np.int64 = start_px_incl - less_size
            assert delta_px_between_segment_first_contig_start_and_query_start >= 0

            total_segment_length_px: np.int64 = es.segment.get_sizes(
            )[2 if exclude_hidden_contigs else 0][resolution]

            atus: List[ATUDescriptor] = []

            def traverse_fn(node: ContigTree.Node) -> None:
                contig_atus = node.contig_descriptor.atus[resolution]
                contig_direction = node.true_direction()
                if contig_direction == ContigDirection.REVERSED:
                    true_contig_atus = []
                    for atu in reversed(contig_atus):
                        new_atu = atu.clone()
                        new_atu.direction = ATUDirection(1-atu.direction.value)
                        true_contig_atus.append(new_atu)
                else:
                    true_contig_atus = contig_atus
                atus.extend(true_contig_atus)
                # atus.extend(contig_atus)

            ContigTree.traverse_nodes_at_resolution(
                es.segment,
                resolution,
                exclude_hidden_contigs,
                traverse_fn
            )

            all_atus_debug = copy.deepcopy(atus)

            total_exposed_atu_length = sum(
                map(
                    lambda atu: atu.end_index_in_stripe_excl -
                    atu.start_index_in_stripe_incl,
                    atus
                )
            )

            assert (
                total_exposed_atu_length == total_segment_length_px
            ), "ATUs total length is not equal to exposed segment length??"

            # TODO: maybe no push is needed
            first_contig_node_in_segment: Optional[ContigTree.Node] = es.segment.leftmost(
                push=False)

            assert first_contig_node_in_segment is not None, "Segment is not empty but has no leftmost contig??"

            first_contig_in_segment: ContigDescriptor = first_contig_node_in_segment.contig_descriptor

            reversed_first_contig_atus_prefix_sum = first_contig_in_segment.atu_prefix_sum_length_bins[
                resolution]

            if first_contig_node_in_segment.direction == ContigDirection.REVERSED:
                reversed_first_contig_atus_prefix_sum = reversed_first_contig_atus_prefix_sum.copy()
                reversed_first_contig_atus_prefix_sum[:-1] = reversed_first_contig_atus_prefix_sum[-1] - np.flip(
                    reversed_first_contig_atus_prefix_sum)[1:]

            index_of_atu_containing_start: np.int64 = np.searchsorted(
                reversed_first_contig_atus_prefix_sum,
                delta_px_between_segment_first_contig_start_and_query_start,
                side='right'
            )

            assert (
                index_of_atu_containing_start < len(
                    reversed_first_contig_atus_prefix_sum)
            ), "Start of query does not fall into exposed leftmost contig??"

            length_of_atus_before_one_containing_start_px: np.int64 = (
                reversed_first_contig_atus_prefix_sum[
                    index_of_atu_containing_start-1
                ] if index_of_atu_containing_start > 0 else np.int64(0)
            )

            old_first_atu = atus[index_of_atu_containing_start]

            assert (
                old_first_atu.start_index_in_stripe_incl < old_first_atu.end_index_in_stripe_excl
            ), "Incorrect old first ATU??"

            new_first_atu: ATUDescriptor = old_first_atu.clone()

            if old_first_atu.direction == ATUDirection.FORWARD:
                new_first_atu.start_index_in_stripe_incl += (
                    delta_px_between_segment_first_contig_start_and_query_start -
                    length_of_atus_before_one_containing_start_px
                )

                assert (
                    0 <= new_first_atu.start_index_in_stripe_incl < new_first_atu.stripe_descriptor.stripe_length_bins
                ), "Incorrect first ATU left border??"

                assert (
                    new_first_atu.start_index_in_stripe_incl < new_first_atu.end_index_in_stripe_excl
                ), "Incorrect new first ATU??"
            else:
                new_first_atu.end_index_in_stripe_excl -= (
                    delta_px_between_segment_first_contig_start_and_query_start -
                    length_of_atus_before_one_containing_start_px
                )

                assert (
                    new_first_atu.end_index_in_stripe_excl >= 0
                ), "Negative right border of new reversed ATU??"

                assert (
                    0 <= new_first_atu.start_index_in_stripe_incl < new_first_atu.stripe_descriptor.stripe_length_bins
                ), "Incorrect first ATU left border??"

                assert (
                    new_first_atu.start_index_in_stripe_incl < new_first_atu.end_index_in_stripe_excl
                ), "Incorrect new first ATU??"

            atus[index_of_atu_containing_start] = new_first_atu
            atus = atus[index_of_atu_containing_start:]

            delta_between_right_px_and_exposed_segment: np.int64 = end_px_excl - \
                (less_size + segment_size)
            last_contig_node = es.segment.rightmost()
            reversed_last_contig_atus_prefix_sum = last_contig_node.contig_descriptor.atu_prefix_sum_length_bins[
                resolution]
            if last_contig_node.direction == ContigDirection.FORWARD:
                reversed_last_contig_atus_prefix_sum = reversed_last_contig_atus_prefix_sum.copy()
                reversed_last_contig_atus_prefix_sum[:-1] = reversed_last_contig_atus_prefix_sum[-1] - np.flip(
                    reversed_last_contig_atus_prefix_sum)[1:]

            right_offset_atus: np.int64 = np.searchsorted(
                reversed_last_contig_atus_prefix_sum,
                -delta_between_right_px_and_exposed_segment,
                side='right'
            )

            deleted_atus_length: np.int64 = np.int64(0)
            if right_offset_atus > 0:
                atus = atus[:-right_offset_atus]
                deleted_atus_length = reversed_last_contig_atus_prefix_sum[right_offset_atus-1]

            old_last_atu = atus[-1]
            assert (
                old_last_atu.start_index_in_stripe_incl < old_last_atu.end_index_in_stripe_excl
            ), "Incorrect old last ATU??"
            new_last_atu = old_last_atu.clone()

            if old_last_atu.direction == ATUDirection.FORWARD:
                new_last_atu.end_index_in_stripe_excl += (
                    deleted_atus_length + delta_between_right_px_and_exposed_segment)
                assert (
                    new_last_atu.stripe_descriptor.stripe_length_bins >= new_last_atu.end_index_in_stripe_excl > new_last_atu.start_index_in_stripe_incl
                ), "Incorrect ATU right border??"
                atus[-1] = new_last_atu

                assert (
                    new_last_atu.start_index_in_stripe_incl < new_last_atu.end_index_in_stripe_excl
                ), "Incorrect new last ATU??"
            else:
                new_last_atu.start_index_in_stripe_incl -= (
                    deleted_atus_length + delta_between_right_px_and_exposed_segment)

                assert (
                    new_last_atu.start_index_in_stripe_incl >= 0
                ), "Negative left border of new reversed last ATU??"

                assert (
                    new_last_atu.stripe_descriptor.stripe_length_bins >= new_last_atu.end_index_in_stripe_excl > new_last_atu.start_index_in_stripe_incl
                ), "Incorrect reversed ATU borders??"
                atus[-1] = new_last_atu

                assert (
                    new_last_atu.start_index_in_stripe_incl < new_last_atu.end_index_in_stripe_excl
                ), "Incorrect new reversed last ATU??"

            assert all(map(
                lambda atu: atu.start_index_in_stripe_incl < atu.end_index_in_stripe_excl,
                atus
            )), "Incorrect ATUs before reduce??"

            total_atu_length = sum(
                map(
                    lambda atu: atu.end_index_in_stripe_excl -
                    atu.start_index_in_stripe_incl, atus
                )
            )

            expected_total_length = (
                min(end_px_excl, total_assembly_length) -
                max(np.int64(0), start_px_incl)
            )

            assert (
                total_atu_length
                == expected_total_length
            ), f"ATUs total length {total_atu_length} is not equal to the requested query's {expected_total_length}??"

            result_atus = ATUDescriptor.reduce(atus)

            assert all(map(
                lambda atu: atu.start_index_in_stripe_incl < atu.end_index_in_stripe_excl,
                result_atus
            )), "Incorrect ATUs after reduce??"

        assert (
            (len(result_atus) <= 0) == (start_px_incl >= end_px_excl)
        ), "No row ATUs were fetched but query is correct??"

        total_result_atu_length = sum(
            map(
                lambda atu: atu.end_index_in_stripe_excl -
                atu.start_index_in_stripe_incl, result_atus
            )
        )

        assert (
            total_result_atu_length
            ==
            expected_total_length

        ), "Resulting ATUs total length is not equal to the requested query??"

        return result_atus

    def reverse_selection_range_bp(self, queried_start_bp: np.int64, queried_end_bp: np.int64) -> None:
        assert (
            self.state == ChunkedFile.FileState.OPENED and self.contig_tree is not None and self.scaffold_tree is not None
        ), "Operation requires file to be opened"

        assert (
            queried_start_bp < queried_end_bp
        ), "Left contig border should be less than right"

        with self.contig_tree.root_lock.gen_wlock(), self.scaffold_tree.root_lock.gen_wlock():
            left_bp, _, right_bp, _ = self.scaffold_tree.extend_borders_to_scaffolds(
                queried_start_bp,
                queried_end_bp
            )

            es = self.contig_tree.expose_segment(
                resolution=np.int64(0),
                start_incl=left_bp,
                end_excl=right_bp,
                units=QueryLengthUnit.BASE_PAIRS
            )

            if es.segment is not None:
                segm = es.segment.clone()
                segm.needs_changing_direction = not segm.needs_changing_direction

                self.contig_tree.commit_exposed_segment(
                    ContigTree.ExposedSegment(
                        es.less,
                        segm.push(),
                        es.greater
                    )
                )

                # self.scaffold_tree.rescaffold(left_bp, right_bp)

    def move_selection_range_bp(
        self,
        queried_start_bp: np.int64,
        queried_end_bp: np.int64,
        target_start_bp: np.int64
    ) -> None:
        assert (
            self.state == ChunkedFile.FileState.OPENED and self.contig_tree is not None and self.scaffold_tree is not None
        ), "Operation requires file to be opened"

        assert (
            queried_start_bp < queried_end_bp
        ), "Left contig border should be less than right"

        with self.contig_tree.root_lock.gen_wlock(), self.scaffold_tree.root_lock.gen_wlock():
            left_bp, _, right_bp, _ = self.scaffold_tree.extend_borders_to_scaffolds(
                queried_start_bp,
                queried_end_bp
            )

            es = self.contig_tree.expose_segment(
                resolution=np.int64(0),
                start_incl=left_bp,
                end_excl=right_bp,
                units=QueryLengthUnit.BASE_PAIRS
            )

            if es.segment is not None:
                tmp = self.contig_tree.merge_nodes(es.less, es.greater)
                nl, nr = self.contig_tree.split_node_by_length(
                    resolution=np.int64(0),
                    t=tmp,
                    k=target_start_bp,
                    include_equal_to_the_left=False,
                    units=QueryLengthUnit.BASE_PAIRS
                )

                self.contig_tree.commit_exposed_segment(
                    ContigTree.ExposedSegment(
                        nl,
                        es.segment,
                        nr
                    )
                )

                self.scaffold_tree.move_selection_range(
                    left_bp, right_bp, target_start_bp)


    def extend_bp_borders_to_contigs(
        self,
        query_start_bp: np.int64,
        query_end_bp: np.int64
    ) -> Tuple[np.int64, np.int64]:
        assert (
            self.contig_tree is not None
        ), "Contig tree is not present?"
        with self.contig_tree.root_lock.gen_rlock():
            es = self.contig_tree.expose_segment(
                resolution=np.int64(0),
                start_incl=query_start_bp,
                end_excl=query_end_bp,
                units=QueryLengthUnit.BASE_PAIRS
            )
            less_size_bp = es.less.get_sizes(
            )[0][0] if es.less is not None else np.int64(0)
            segm_size_bp = es.segment.get_sizes(
            )[0][0] if es.segment is not None else np.int64(0)
            return (less_size_bp, less_size_bp+segm_size_bp)

    def scaffold_segment(
        self,
        query_start_bp: np.int64,
        query_end_bp: np.int64,
        name: Optional[str] = None,
        spacer_length: int = 1000
    ) -> ScaffoldDescriptor:
        assert (
            self.state == ChunkedFile.FileState.OPENED and self.contig_tree is not None and self.scaffold_tree is not None
        ), "Operation requires file to be opened"
        ctg_l_bp, ctg_r_bp = self.extend_bp_borders_to_contigs(
            query_start_bp,
            query_end_bp
        )
        return self.scaffold_tree.rescaffold(ctg_l_bp, ctg_r_bp, spacer_length)

    def unscaffold_segment(
        self,
        query_start_bp: np.int64,
        query_end_bp: np.int64,
    ) -> None:
        assert (
            self.state == ChunkedFile.FileState.OPENED and self.contig_tree is not None and self.scaffold_tree is not None
        ), "Operation requires file to be opened"
        ctg_l_bp, ctg_r_bp = self.extend_bp_borders_to_contigs(
            query_start_bp,
            query_end_bp
        )
        self.scaffold_tree.unscaffold(ctg_l_bp, ctg_r_bp)

    def close(self, need_save: bool = True) -> None:
        self.state = ChunkedFile.FileState.CLOSED

    def link_fasta(self, fasta_filename: str) -> None:
        with self.fasta_file_lock.gen_wlock():
            if self.fasta_processor is not None:
                print("Warning: re-linking FASTA file")
                del self.fasta_processor
            self.fasta_processor = FASTAProcessor(fasta_filename)

    def get_fasta_for_assembly(self, writable_stream) -> None:
        assert (
            self.state == ChunkedFile.FileState.OPENED and self.contig_tree is not None and self.scaffold_tree is not None
        ), "Operation requires file to be opened"
        with self.fasta_file_lock.gen_rlock():
            if self.fasta_processor is None:
                raise Exception("FASTA File is not linked")
            
            contigs_and_dirs, scaffolds_and_lengths = self.get_assembly_info()

            ordered_finalization_records: List[Tuple[Optional[ScaffoldDescriptor], List[Tuple[ContigDescriptor, ContigDirection]]]] = [
            ]
            
            scaffold_index: int = 0
            bp_position: int = 0
            scaffold_left_bp: int = 0
            
            for ctg, ctg_dir in contigs_and_dirs:
                while (scaffold_index < len(scaffolds_and_lengths)) and (bp_position >= scaffold_left_bp + scaffolds_and_lengths[scaffold_index][1]):
                    scaffold_left_bp += scaffolds_and_lengths[scaffold_index][1]
                    scaffold_index += 1
                
                if scaffold_index < len(scaffolds_and_lengths):
                    opt_sd = scaffolds_and_lengths[scaffold_index][0]
                else:
                    opt_sd = None
                
                if opt_sd is None:
                    ordered_finalization_records.append((
                        None,
                        [(ctg, ctg_dir)]
                    ))
                else:
                    last_scaffold_desc: Optional[ScaffoldDescriptor] = None
                    if len(ordered_finalization_records) > 0:
                        last_scaffold_desc = ordered_finalization_records[-1][0]
                    if last_scaffold_desc is not None and last_scaffold_desc.scaffold_id == opt_sd.scaffold_id and last_scaffold_desc.scaffold_name == opt_sd.scaffold_name:
                        ordered_finalization_records[-1][1].append((ctg, ctg_dir))
                    else:
                        ordered_finalization_records.append((opt_sd, [(ctg, ctg_dir)]))
                 
                bp_position += ctg.contig_length_at_resolution[0]


            self.fasta_processor.finalize_fasta_for_assembly(
                writable_stream,
                ordered_finalization_records,
            )

    def load_assembly_from_agp(self, agp_filepath: Path) -> None:
        assert (
            self.state == ChunkedFile.FileState.OPENED and self.contig_tree is not None and self.scaffold_tree is not None
        ), "Operation requires file to be opened"

        agpParser: AGPparser = AGPparser(agp_filepath.absolute())
        contig_records = agpParser.getAGPContigRecords()
        scaffold_records = agpParser.getAGPScaffoldRecords()

        contig_id_to_borders_bp: Dict[np.int64,
                                      Tuple[np.int64, np.int64]] = dict()
        position_bp: np.int64 = np.int64(0)

        with self.contig_tree.root_lock.gen_wlock():
            self.contig_tree.root = None

            for i, contig_record in enumerate(contig_records):
                contig_id = self.contig_name_to_contig_id[contig_record.name]
                contig_descriptor = self.contig_id_to_contig_descriptor[contig_id]
                self.contig_tree.insert_at_position(
                    contig_descriptor,
                    i,
                    direction=contig_record.direction,
                    # update_tree=False
                )
                contig_id_to_borders_bp[contig_id] = (
                    position_bp, position_bp+contig_descriptor.contig_length_at_resolution[0])
                position_bp += contig_descriptor.contig_length_at_resolution[0]

        old_scaffold_tree = self.scaffold_tree
        with old_scaffold_tree.root_lock.gen_rlock():
            new_scaffold_tree = ScaffoldTree(
                old_scaffold_tree.root.update_sizes().subtree_length_bp, self.mp_manager)
            with new_scaffold_tree.root_lock.gen_wlock():
                for scaffold_ord, scaffold_record in enumerate(scaffold_records):
                    start_contig_id: np.int64 = self.contig_name_to_contig_id[
                        scaffold_record.start_ctg]
                    end_contig_id: np.int64 = self.contig_name_to_contig_id[scaffold_record.end_ctg]
                    scaffold_start_bp = contig_id_to_borders_bp[start_contig_id][0]
                    scaffold_end_bp = contig_id_to_borders_bp[end_contig_id][1]
                    sd = ScaffoldDescriptor.make_scaffold_descriptor(
                        scaffold_ord,
                        scaffold_record.name
                    )
                    new_scaffold_tree.add_scaffold(
                        scaffold_start_bp, scaffold_end_bp, sd)
            self.scaffold_tree = new_scaffold_tree
        gc.collect()

    def get_agp_for_assembly(self, writable_stream) -> None:
        assert (
            self.state == ChunkedFile.FileState.OPENED and self.contig_tree is not None and self.scaffold_tree is not None
        ), "Operation requires file to be opened"

        agp_export_processor: AGPExporter = AGPExporter()

        ordered_contig_descriptors: List[
            Tuple[
                ContigDescriptor,
                ContigDirection,
                # Dict[np.int64, Tuple[np.int64, np.int64]]
            ]
        ] = self.contig_tree.get_contig_list()

        scaffold_list: List[Tuple[ScaffoldDescriptor, ScaffoldBordersBP]
                            ] = self.scaffold_tree.get_scaffold_list()

        agp_export_processor.exportAGP(
            writable_stream,
            ordered_contig_descriptors,
            scaffold_list,
        )

        
    def get_fasta_for_range(
            self, from_bp_incl: np.int64, to_bp_excl: np.int64,
            buf: BytesIO,
            intercontig_spacer: str = 500 * 'N'
    ) -> None:
        assert (
            (self.state == ChunkedFile.FileState.OPENED) 
            and 
            (self.contig_tree is not None) 
            and 
            (self.scaffold_tree is not None)
            and 
            (self.fasta_processor is not None)
        ), "Operation requires file to be opened"
        
        with self.contig_tree.root_lock.gen_rlock():
            es = self.contig_tree.expose_segment(np.int64(0), from_bp_incl, to_bp_excl, QueryLengthUnit.BASE_PAIRS)
            if es.segment is not None:
                left_size = es.less.get_sizes()[0][0] if es.less is not None else np.int64(0)
                segment_size = es.segment.get_sizes()[0][0]
                delta_in_first_contig = from_bp_incl - left_size
                delta_in_last_contig = (left_size + segment_size) - to_bp_excl
                
                descriptors: List[Tuple[ContigDescriptor, ContigDirection]] = []
                self.contig_tree.traverse_node(es.segment, lambda n: descriptors.append((n.contig_descriptor, n.true_direction())))
                
                self.fasta_processor.get_fasta_for_range(
                    buf,
                    descriptors,
                    f"{from_bp_incl}bp-{to_bp_excl}bp",
                    delta_in_first_contig,
                    delta_in_last_contig,
                    intercontig_spacer=intercontig_spacer
                )
                
    def convert_units(
        self,
        position: np.int64,
        from_resolution: np.int64,
        from_units: QueryLengthUnit,
        to_resolution: np.int64,
        to_units: QueryLengthUnit
    ) -> np.int64:
        assert (
            (self.state == ChunkedFile.FileState.OPENED) 
            and 
            (self.contig_tree is not None) 
        ), "Operation requires file to be opened"
                
        assert (
            (from_units == QueryLengthUnit.BASE_PAIRS) == (from_resolution == 0)
        ), "If converting from base pairs, set from_resolution=0"
        assert (
            (to_units == QueryLengthUnit.BASE_PAIRS) == (to_resolution == 0)
        ), "If converting to base pairs, set to_resolution=0"
        
        es = self.contig_tree.expose_segment(
            from_resolution,
            position,
            position+1,
            from_units
        )
        left_from_units = 0
        left_to_units = 0
        if es.less is not None:
            left_from_units = es.less.get_sizes()[{QueryLengthUnit.BASE_PAIRS: 0, QueryLengthUnit.BINS: 1, QueryLengthUnit.PIXELS: 2}[from_units]][from_resolution]
            left_to_units = es.less.get_sizes()[{QueryLengthUnit.BASE_PAIRS: 0, QueryLengthUnit.BINS: 1, QueryLengthUnit.PIXELS: 2}[to_units]][to_resolution]
            
        delta_from_units = position - left_from_units
        delta_bp = delta_from_units if from_units == QueryLengthUnit.BASE_PAIRS else (delta_from_units*from_resolution)
        
        delta_to_units = delta_bp if to_units == QueryLengthUnit.BASE_PAIRS else (delta_bp//to_resolution)
        
        return left_to_units + delta_to_units           
            
                
                
    def split_contig_at_bin(
        self,
        split_position: np.int64,
        split_resolution: np.int64,        
        split_units: QueryLengthUnit
    ) -> None:
        assert (
            (self.state == ChunkedFile.FileState.OPENED) 
            and 
            (self.contig_tree is not None) 
            and
            (self.scaffold_tree is not None) 
        ), "Operation requires file to be opened"
        
        if split_units == QueryLengthUnit.BASE_PAIRS:
            assert (
                split_resolution == 0
            ), "In bp query resolution should be set to 0"
            
        min_resolution = min(self.resolutions)
        
        with self.contig_tree.root_lock.gen_wlock(), self.scaffold_tree.root_lock.gen_wlock():
            split_position_bins = self.convert_units(
                position=split_position,
                from_resolution=split_resolution,
                from_units=split_units,
                to_resolution=min_resolution,
                to_units=QueryLengthUnit.BINS                
            )
            
            es = self.contig_tree.expose_segment(
                min_resolution,
                split_position_bins, 
                split_position_bins+1,
                units=QueryLengthUnit.BINS
            )
            
            left_bins = 0
            if es.less is not None:
                left_bins = es.less.get_sizes()[0][min_resolution]
                
            split_position_bp = self.convert_units(split_position, split_resolution, split_units, np.int64(0), QueryLengthUnit.BASE_PAIRS)
            
            assert (
                es.segment is not None
            ), "Split position does not fall into any contig?"
            
            segment_sizes = es.segment.get_sizes()
            
            assert (
                segment_sizes[1] == 1
            ), f"Split position should fall into exactly one contig (currently segment holds {segment_sizes[1]} nodes)"
            
            node = es.segment
            old_contig = node.contig_descriptor
            
            delta_from_contig_start = split_position_bins - left_bins
            
            max_contig_id = max(map(lambda cd: cd.contig_id, self.contig_id_to_contig_descriptor.values()))
            new_contig_ids: Tuple[np.int64, np.int64] = (1+max_contig_id, 2+max_contig_id)
            new_contig_names: Tuple[str, str] = (f"{old_contig.contig_name}_hictsplit_1", f"{old_contig.contig_name}_hictsplit_2")
            new_contig_length_bps: Tuple[np.int64, np.int64] = (delta_from_contig_start*min_resolution, old_contig.contig_length_at_resolution[0] - (1+delta_from_contig_start)*min_resolution)
            
            new_contig_length_at_resolution: Tuple[Dict[np.int64, np.int64], Dict[np.int64, np.int64]] = (dict(), dict())
            new_contig_presence_in_resolution: Tuple[Dict[np.int64, ContigHideType], Dict[np.int64, ContigHideType]] = (dict(), dict())
            new_atus: Tuple[Dict[np.int64, List[ATUDescriptor]], Dict[np.int64, List[ATUDescriptor]]] = (dict(), dict())
            
            new_contig_names_in_source_fasta: Tuple[str, str] = (old_contig.contig_name_in_source_fasta, old_contig.contig_name_in_source_fasta)
            new_offsets_inside_fasta_contig: Tuple[np.int64, np.int64]
            if node.direction == ContigDirection.FORWARD:
                new_offsets_inside_fasta_contig = (old_contig.offset_inside_fasta_contig, old_contig.offset_inside_fasta_contig + (1+delta_from_contig_start)*min_resolution)
            else:
                new_offsets_inside_fasta_contig = (old_contig.offset_inside_fasta_contig + (1+delta_from_contig_start)*min_resolution, old_contig.offset_inside_fasta_contig)

            
            
            for resolution in self.resolutions:
                
                delta_from_start_at_resolution = self.convert_units(
                    delta_from_contig_start,
                    min_resolution, 
                    QueryLengthUnit.BINS,
                    resolution,
                    QueryLengthUnit.BINS
                )
                
                if resolution == min_resolution:
                    new_contig_length_at_resolution[0][resolution] = delta_from_contig_start
                    new_contig_length_at_resolution[1][resolution] = old_contig.contig_length_at_resolution[resolution] - delta_from_contig_start - 1
                else:
                    new_contig_length_at_resolution[0][resolution] = delta_from_start_at_resolution
                    new_contig_length_at_resolution[1][resolution] = old_contig.contig_length_at_resolution[resolution] - delta_from_start_at_resolution
                    
                def copy_true_atu(atu: ATUDescriptor, contig_direction: ContigDirection) -> ATUDescriptor:
                    new_atu = atu.clone()
                    if contig_direction == ContigDirection.REVERSED:
                        new_atu.direction = ATUDirection(1-new_atu.direction.value)
                    return new_atu
                    
                source_atus = tuple(map(lambda old_atu: copy_true_atu(old_atu, node.direction), old_contig.atus[resolution]))
                source_atus_prefix_sum = old_contig.atu_prefix_sum_length_bins[resolution]
                if node.direction == ContigDirection.REVERSED:
                    source_atus_prefix_sum = source_atus_prefix_sum.copy()
                    source_atus_prefix_sum[:-1] = source_atus_prefix_sum[-1] - np.flip(source_atus_prefix_sum)[1:]
                    source_atus = tuple(reversed(source_atus))
                    
                index_of_atu_where_split_occurs = np.searchsorted(source_atus_prefix_sum, delta_from_start_at_resolution, side='left')                    
                old_join_atu = source_atus[index_of_atu_where_split_occurs]
                atus_l = list(source_atus[:index_of_atu_where_split_occurs])
                atus_r = list(source_atus[-1+1+index_of_atu_where_split_occurs:])
                atus_l_length_bins = source_atus_prefix_sum[index_of_atu_where_split_occurs-1] if index_of_atu_where_split_occurs > 0 else 0
                
                #atus_r_length_bins = source_atus_prefix_sum[-1] - atus_l_length_bins
                
                delta_l = delta_from_start_at_resolution - atus_l_length_bins
                if delta_l > 0:
                    atus_l.append(
                        ATUDescriptor.make_atu_descriptor(
                            old_join_atu.stripe_descriptor,
                            old_join_atu.start_index_in_stripe_incl if old_join_atu.direction == ATUDirection.FORWARD else (old_join_atu.end_index_in_stripe_excl - delta_l),
                            (old_join_atu.start_index_in_stripe_incl + delta_l) if old_join_atu.direction == ATUDirection.FORWARD else old_join_atu.end_index_in_stripe_excl,
                            old_join_atu.direction                                
                        )
                    )
                    
                assert (
                    atus_l_length_bins + delta_l == new_contig_length_at_resolution[0][resolution]
                ), "Unexpected length of left part"
                
                #delta_r_positive = new_contig_length_at_resolution[1][resolution] - atus_r_length_bins
                
                new_r_atu_start = (old_join_atu.start_index_in_stripe_incl + delta_l + (0 if resolution != min_resolution else 1)) if old_join_atu.direction == ATUDirection.FORWARD else (old_join_atu.start_index_in_stripe_incl)
                new_r_atu_end = (old_join_atu.end_index_in_stripe_excl) if old_join_atu.direction == ATUDirection.FORWARD else (old_join_atu.end_index_in_stripe_excl - delta_l + (0 if resolution != min_resolution else 1))
                
                if new_r_atu_end - new_r_atu_start > 0:
                    # atus_r[0] = ATUDescriptor.make_atu_descriptor(
                    #         old_join_atu.stripe_descriptor,
                    #         (old_join_atu.start_index_in_stripe_incl + delta_r_positive) if old_join_atu.direction == ATUDirection.FORWARD else old_join_atu.start_index_in_stripe_incl,
                    #         old_join_atu.end_index_in_stripe_excl if old_join_atu.direction == ATUDirection.FORWARD else (old_join_atu.end_index_in_stripe_excl - delta_r_positive),
                    #         old_join_atu.direction     
                    #     )
                    atus_r[0] = ATUDescriptor.make_atu_descriptor(
                            old_join_atu.stripe_descriptor,
                            new_r_atu_start,
                            new_r_atu_end,
                            old_join_atu.direction        
                        )
                else:
                    atus_r = atus_r[1:]
                    
                    
                assert (
                    sum(map(lambda atu: atu.end_index_in_stripe_excl - atu.start_index_in_stripe_incl, atus_r)) == new_contig_length_at_resolution[1][resolution]
                ), "Unexpected length of right part"

                
                if old_contig.presence_in_resolution[resolution] in (
                    ContigHideType.FORCED_HIDDEN, ContigHideType.FORCED_SHOWN
                ):
                    new_contig_length_at_resolution[0][resolution] = old_contig.presence_in_resolution[resolution] 
                    new_contig_length_at_resolution[1][resolution] = old_contig.presence_in_resolution[resolution] 
                else:
                    new_contig_presence_in_resolution[0][resolution] = ContigHideType.AUTO_SHOWN if new_contig_length_bps[0] >= resolution else ContigHideType.AUTO_HIDDEN
                    new_contig_presence_in_resolution[1][resolution] = ContigHideType.AUTO_SHOWN if new_contig_length_bps[1] >= resolution else ContigHideType.AUTO_HIDDEN
                    
                new_atus[0][resolution] = atus_l
                new_atus[1][resolution] = atus_r
                
            new_contigs = tuple(
                map(
                    lambda t: ContigDescriptor.make_contig_descriptor(*t), 
                    zip(
                        new_contig_ids,
                        new_contig_names,
                        new_contig_length_bps,
                        new_contig_length_at_resolution,
                        new_contig_presence_in_resolution,
                        new_atus,
                        new_contig_names_in_source_fasta,
                        new_offsets_inside_fasta_contig
                    )
                )
            )
            
            new_nodes = tuple(map(lambda cd: ContigTree.Node.make_new_node_from_descriptor(cd, node.direction), new_contigs))
            
            new_segment_part = ContigTree.Node.merge_nodes(*new_nodes)
            
            new_exposed_segment = ContigTree.ExposedSegment(es.less, new_segment_part, es.greater)
            
            self.contig_tree.commit_exposed_segment(new_exposed_segment)
            
            self.scaffold_tree.remove_segment_from_assembly(
                start_bp_incl=split_position_bp,
                end_bp_excl=split_position_bp + min_resolution
            )
        
        
    def get_ordered_contigs(self) -> List[Tuple[ContigDescriptor, ContigDirection]]:
        tree = self.contig_tree
        result: List[Tuple[ContigDescriptor, ContigDirection]] = []

        assert (
            tree is not None
        ), "No contig tree is present?"

        def traverse_fn(n: ContigTree.Node) -> None:
            nonlocal result
            result.append((
                n.contig_descriptor,
                n.true_direction()
            ))

        tree.traverse(traverse_fn)

        return result

    def get_ordered_scaffolds(self) -> List[Tuple[Optional[ScaffoldDescriptor], int]]:
        tree = self.scaffold_tree
        result: List[Tuple[Optional[ScaffoldDescriptor], int]] = []

        assert (
            tree is not None
        ), "No scaffold tree is present?"

        def traverse_fn(n: ScaffoldTree.Node) -> None:
            nonlocal result
            result.append((
                n.scaffold_descriptor,
                int(n.length_bp)
            ))

        tree.traverse(traverse_fn)

        return result

    def get_assembly_info(self) -> Tuple[List[Tuple[ContigDescriptor, ContigDirection]], List[Tuple[Optional[ScaffoldDescriptor], int]]]:
        contig_tree = self.contig_tree
        scaffold_tree = self.scaffold_tree

        assert (
            contig_tree is not None
        ), "Contig tree is None?"
        assert (
            scaffold_tree is not None
        ), "Scaffold tree is None?"

        with contig_tree.root_lock.gen_rlock(), scaffold_tree.root_lock.gen_rlock():
            return self.get_ordered_contigs(), self.get_ordered_scaffolds()
                
            
            
        
        