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

import gc
import time
import random
from typing import Dict
from hict.api.ContactMatrixFacet import ContactMatrixFacet
from hict.core.common import QueryLengthUnit
import numpy as np
import cooler
from readerwriterlock import rwlock
from pathlib import Path
import pytest
from pytest import fail
from hypothesis import given, example, event, settings, strategies as st, assume, HealthCheck
from hypothesis.extra import numpy as nps
import multiprocessing
import multiprocessing.managers
import dill

mp_manager: multiprocessing.managers.SyncManager = multiprocessing.Manager()

mp_rlock = mp_manager.RLock()


def get_lock():
    return mp_rlock


random.seed(int(time.time()))

file_name: str = "zanu_male_4DN.mcool" # "mat18_100k.cool"  # "zanu_male_4DN.mcool"


mcool_file_path: Path = Path(
    ".", "..", "HiCT_Server", "data", file_name).resolve()
hict_file_path: Path = Path(
    ".", "..", "HiCT_Server", "data", f"{file_name}.hict.hdf5").resolve()

if not hict_file_path.is_file():
    pytest.exit(
        msg=f"Test hict file must be present for this test at {hict_file_path}")

if not mcool_file_path.is_file():
    pytest.exit(
        msg=f"Test mcool file must be present for this test at {mcool_file_path}")
# pytestmark = pytest.mark.skipif(
#     not hict_file_path.is_file(),
#     reason=f"Test hict file must be present for this test at {hict_file_path}"
# )
# pytestmark()
# pytestmark = pytest.mark.skipif(
#     not mcool_file_path.is_file(),
#     reason=f"Test mcool file must be present for this test at {mcool_file_path}"
# )

resolutions_mcool = list(map(lambda s: int(s.replace(
    '/resolutions/', '')), cooler.fileops.list_coolers(str(mcool_file_path))))
hict_file = ContactMatrixFacet.get_file_descriptor(
    str(hict_file_path), 4, mp_manager=mp_manager)
ContactMatrixFacet.open_file(hict_file)
resolutions_hict = ContactMatrixFacet.get_resolutions_list(hict_file)
resolution_to_size_bins: Dict[np.int64, np.int64] = dict()
assert hict_file.contig_tree.root is not None, "HiCT file has no matrix inside?"
total_bp_length = hict_file.contig_tree.root.get_sizes()[0][0]
hict_file_lock: rwlock.RWLockWrite = rwlock.RWLockWrite(lock_factory=get_lock)




def test_resolutions_match():
    assert (
        sorted(resolutions_mcool) == sorted(resolutions_hict)
    ), "Resolutions in mCool and HiCT files should match"

# NOTE: Query size is not limited so this method may fail due to the OoM


@settings(
    max_examples=5000,
    deadline=30000,
    derandomize=True,
    report_multiple_bugs=True,
    suppress_health_check=(
        HealthCheck.filter_too_much,
        HealthCheck.data_too_large
    )
)
@given(
    resolution=st.sampled_from(resolutions_mcool),
    start_row_incl_bp=st.integers(min_value=0, max_value=total_bp_length),
    start_col_incl_bp=st.integers(min_value=0, max_value=total_bp_length),
    end_row_excl_bp=st.integers(min_value=0, max_value=total_bp_length),
    end_col_excl_bp=st.integers(min_value=0, max_value=total_bp_length)
)
def test_compare_with_cooler(
    resolution,
    start_row_incl_bp,
    start_col_incl_bp,
    end_row_excl_bp,
    end_col_excl_bp,
):
    matrix_size_bins = ContactMatrixFacet.get_matrix_size_bins(
        hict_file, resolution)
    start_row_incl = (start_row_incl_bp // resolution) % matrix_size_bins
    start_col_incl = (start_col_incl_bp // resolution) % matrix_size_bins
    end_row_excl = (end_row_excl_bp // resolution) % matrix_size_bins
    end_col_excl = (end_col_excl_bp // resolution) % matrix_size_bins
    if start_row_incl > end_row_excl:
        start_row_incl, end_row_excl = end_row_excl, start_row_incl
    if start_col_incl > end_col_excl:
        start_col_incl, end_col_excl = end_col_excl, start_col_incl
    if end_row_excl - start_row_incl > 2048:
        end_row_excl = start_row_incl + \
            ((end_row_excl - start_row_incl) % 2048)
    if end_col_excl - start_col_incl > 2048:
        end_col_excl = start_col_incl + \
            ((end_col_excl - start_col_incl) % 2048)
    if start_row_incl > end_row_excl:
        start_row_incl, end_row_excl = end_row_excl, start_row_incl
    if start_col_incl > end_col_excl:
        start_col_incl, end_col_excl = end_col_excl, start_col_incl
    cooler_file: cooler.Cooler = cooler.Cooler(
        "{}::/resolutions/{}".format(str(mcool_file_path), resolution))
    cooler_matrix_selector: cooler.api.RangeSelector2D = cooler_file.matrix(
        field='count', balance=False)
    cooler_dense: np.ndarray = cooler_matrix_selector[start_row_incl:end_row_excl,
                                                      start_col_incl:end_col_excl]
    with hict_file_lock.gen_rlock() as hfl:
        my_dense = ContactMatrixFacet.get_dense_submatrix(
            hict_file,
            resolution,
            start_row_incl,
            start_col_incl,
            end_row_excl,
            end_col_excl,
            units=QueryLengthUnit.BINS,
            exclude_hidden_contigs=False
        )[0]
    my_dense = np.pad(my_dense, [(0, end_row_excl-start_row_incl-my_dense.shape[0]), (0,
                      end_col_excl-start_col_incl-my_dense.shape[1])], mode='constant', constant_values=0)
    assert (
        my_dense.shape == (end_row_excl-start_row_incl,
                           end_col_excl-start_col_incl)
    ), f"Matrix shape {my_dense.shape} should be equal to that of query: {(end_row_excl-start_row_incl, end_col_excl-start_col_incl)}, whereas cooler returned {cooler_dense.shape}"
    assert (
        np.array_equal(cooler_dense, my_dense)
    ), "Dense random submatrices returned by Cooler and HiCT should be equal"
    del cooler_dense
    del my_dense
    del cooler_file
    gc.collect()


@settings(
    max_examples=500,
    deadline=30000,
    derandomize=True,
    report_multiple_bugs=True,
    suppress_health_check=(
        HealthCheck.filter_too_much,
        HealthCheck.data_too_large
    )
)
@given(
    resolution=st.sampled_from(resolutions_mcool),
    start_row_incl=st.integers(min_value=0, max_value=total_bp_length),
    start_col_incl=st.integers(min_value=0, max_value=total_bp_length),
    end_row_excl=st.integers(min_value=0, max_value=total_bp_length),
    end_col_excl=st.integers(min_value=0, max_value=total_bp_length)
)
def test_compare_with_cooler_by_bins(
    resolution,
    start_row_incl,
    start_col_incl,
    end_row_excl,
    end_col_excl,
):
    matrix_size_bins = ContactMatrixFacet.get_matrix_size_bins(
        hict_file, resolution)
    start_row_incl %= matrix_size_bins
    start_col_incl %= matrix_size_bins
    end_row_excl %= matrix_size_bins
    end_col_excl %= matrix_size_bins
    if start_row_incl > end_row_excl:
        start_row_incl, end_row_excl = end_row_excl, start_row_incl
    if start_col_incl > end_col_excl:
        start_col_incl, end_col_excl = end_col_excl, start_col_incl
    if end_row_excl - start_row_incl > 2048:
        end_row_excl = start_row_incl + \
            ((end_row_excl - start_row_incl) % 2048)
    if end_col_excl - start_col_incl > 2048:
        end_col_excl = start_col_incl + \
            ((end_col_excl - start_col_incl) % 2048)
    cooler_file: cooler.Cooler = cooler.Cooler(
        "{}::/resolutions/{}".format(str(mcool_file_path), resolution))
    cooler_matrix_selector: cooler.api.RangeSelector2D = cooler_file.matrix(
        field='count', balance=False)
    cooler_dense: np.ndarray = cooler_matrix_selector[start_row_incl:end_row_excl,
                                                      start_col_incl:end_col_excl]
    with hict_file_lock.gen_rlock():
        my_dense = ContactMatrixFacet.get_dense_submatrix(
            hict_file,
            resolution,
            start_row_incl,
            start_col_incl,
            end_row_excl,
            end_col_excl,
            units=QueryLengthUnit.BINS,
            exclude_hidden_contigs=False
        )[0]
    my_dense = np.pad(my_dense, [(0, end_row_excl-start_row_incl-my_dense.shape[0]), (0,
                      end_col_excl-start_col_incl-my_dense.shape[1])], mode='constant', constant_values=0)
    assert (
        my_dense.shape == (end_row_excl-start_row_incl,
                           end_col_excl-start_col_incl)
    ), f"Matrix shape {my_dense.shape} should be equal to that of query: {(end_row_excl-start_row_incl, end_col_excl-start_col_incl)}, whereas cooler returned {cooler_dense.shape}"
    assert (
        np.array_equal(cooler_dense, my_dense)
    ), "Dense random submatrices returned by Cooler and HiCT should be equal"
    del cooler_dense
    del my_dense
    del cooler_file
    gc.collect()


@settings(
    max_examples=500,
    deadline=30000,
    derandomize=True,
    report_multiple_bugs=True,
    suppress_health_check=(
        HealthCheck.filter_too_much,
        HealthCheck.data_too_large
    )
)
@given(
    resolution=st.sampled_from(resolutions_mcool),
    start_row_incl_bp=st.integers(min_value=0, max_value=total_bp_length),
    start_col_incl_bp=st.integers(min_value=0, max_value=total_bp_length),
    # , 5, 10, 64, 100, 127, 512, 1000, 2560])
    query_size=st.sampled_from([1, 2, 3]),
)
def test_compare_small_square_queries_with_cooler(
    resolution,
    start_row_incl_bp,
    start_col_incl_bp,
    query_size
):
    return compare_square_queries_with_cooler(
        resolution,
        start_row_incl_bp,
        start_col_incl_bp,
        query_size
    )


@settings(
    max_examples=500,
    deadline=30000,
    derandomize=True,
    report_multiple_bugs=True,
    suppress_health_check=(
        HealthCheck.filter_too_much,
        HealthCheck.data_too_large
    )
)
@given(
    resolution=st.sampled_from(resolutions_mcool),
    start_row_incl_bp=st.integers(min_value=0, max_value=total_bp_length),
    start_col_incl_bp=st.integers(min_value=0, max_value=total_bp_length),
    query_size=st.sampled_from([1, 2, 5, 10, 64, 100, 127, 512, 1000, 2560])
)
def test_compare_square_queries_with_cooler(
    resolution,
    start_row_incl_bp,
    start_col_incl_bp,
    query_size
):
    return compare_square_queries_with_cooler(
        resolution,
        start_row_incl_bp,
        start_col_incl_bp,
        query_size
    )


def compare_square_queries_with_cooler(
    resolution,
    start_row_incl_bp,
    start_col_incl_bp,
    query_size
):
    matrix_size_bins = ContactMatrixFacet.get_matrix_size_bins(
        hict_file, resolution)
    start_row_incl = (start_row_incl_bp // resolution) % matrix_size_bins
    start_col_incl = (start_col_incl_bp // resolution) % matrix_size_bins
    end_row_excl = start_row_incl + query_size
    end_col_excl = start_col_incl + query_size
    if start_row_incl > end_row_excl:
        start_row_incl, end_row_excl = end_row_excl, start_row_incl
    if start_col_incl > end_col_excl:
        start_col_incl, end_col_excl = end_col_excl, start_col_incl
    cooler_file: cooler.Cooler = cooler.Cooler(
        "{}::/resolutions/{}".format(str(mcool_file_path), resolution))
    cooler_matrix_selector: cooler.api.RangeSelector2D = cooler_file.matrix(
        field='count', balance=False)
    cooler_dense: np.ndarray = cooler_matrix_selector[start_row_incl:end_row_excl,
                                                      start_col_incl:end_col_excl]
    with hict_file_lock.gen_rlock() as hfl:
        my_dense = ContactMatrixFacet.get_dense_submatrix(
            hict_file,
            resolution,
            start_row_incl,
            start_col_incl,
            end_row_excl,
            end_col_excl,
            units=QueryLengthUnit.BINS,
            exclude_hidden_contigs=False
        )[0]
    my_dense = np.pad(my_dense, [(0, query_size-my_dense.shape[0]), (0,
                      query_size-my_dense.shape[1])], mode='constant', constant_values=0)
    assert (
        my_dense.shape == (query_size, query_size)
    ), f"Matrix shape {my_dense.shape} should be equal to that of query: {(query_size, query_size)}, whereas cooler returned {cooler_dense.shape}"
    assert (
        np.array_equal(cooler_dense, my_dense)
    ), "Dense square submatrices returned by Cooler and HiCT should be equal"
    del cooler_dense
    del my_dense
    del cooler_file
    gc.collect()


@settings(
    max_examples=500,
    deadline=30000,
    derandomize=True,
    report_multiple_bugs=True,
    suppress_health_check=(
        HealthCheck.filter_too_much,
        HealthCheck.data_too_large
    )
)
@given(
    resolution=st.sampled_from(resolutions_mcool),
    start_row_incl=st.integers(min_value=0, max_value=total_bp_length),
    start_col_incl=st.integers(min_value=0, max_value=total_bp_length),
    query_size=st.sampled_from([1, 2, 3, 5, 10])
)
def test_compare_small_square_queries_with_cooler_by_bins(
    resolution,
    start_row_incl,
    start_col_incl,
    query_size
):
    return compare_square_queries_with_cooler_by_bins(
        resolution,
        start_row_incl,
        start_col_incl,
        query_size
    )


@settings(
    max_examples=500,
    deadline=30000,
    derandomize=True,
    report_multiple_bugs=True,
    suppress_health_check=(
        HealthCheck.filter_too_much,
        HealthCheck.data_too_large
    )
)
@given(
    resolution=st.sampled_from(resolutions_mcool),
    start_row_incl=st.integers(min_value=0, max_value=total_bp_length),
    start_col_incl=st.integers(min_value=0, max_value=total_bp_length),
    query_size=st.sampled_from([1, 2, 5, 10, 64, 100, 127, 512, 1000, 2560])
)
def test_compare_square_queries_with_cooler_by_bins(
    resolution,
    start_row_incl,
    start_col_incl,
    query_size
):
    return compare_square_queries_with_cooler_by_bins(
        resolution,
        start_row_incl,
        start_col_incl,
        query_size
    )


def compare_square_queries_with_cooler_by_bins(
    resolution,
    start_row_incl,
    start_col_incl,
    query_size
):
    matrix_size_bins = ContactMatrixFacet.get_matrix_size_bins(
        hict_file, resolution)
    start_row_incl %= matrix_size_bins
    start_col_incl %= matrix_size_bins
    end_row_excl = start_row_incl + query_size
    end_col_excl = start_col_incl + query_size
    if start_row_incl > end_row_excl:
        start_row_incl, end_row_excl = end_row_excl, start_row_incl
    if start_col_incl > end_col_excl:
        start_col_incl, end_col_excl = end_col_excl, start_col_incl
    cooler_file: cooler.Cooler = cooler.Cooler(
        "{}::/resolutions/{}".format(str(mcool_file_path), resolution))
    cooler_matrix_selector: cooler.api.RangeSelector2D = cooler_file.matrix(
        field='count', balance=False)
    cooler_dense: np.ndarray = cooler_matrix_selector[start_row_incl:end_row_excl,
                                                      start_col_incl:end_col_excl]
    with hict_file_lock.gen_rlock():
        my_dense = ContactMatrixFacet.get_dense_submatrix(
            hict_file,
            resolution,
            start_row_incl,
            start_col_incl,
            end_row_excl,
            end_col_excl,
            units=QueryLengthUnit.BINS,
            exclude_hidden_contigs=False
        )[0]
    my_dense = np.pad(my_dense, [(0, query_size-my_dense.shape[0]), (0,
                      query_size-my_dense.shape[1])], mode='constant', constant_values=0)
    assert (
        my_dense.shape == (query_size, query_size)
    ), f"Matrix shape {my_dense.shape} should be equal to that of query: {(query_size, query_size)}, whereas cooler returned {cooler_dense.shape}"
    if not np.array_equal(cooler_dense, my_dense):
        again_my_dense = ContactMatrixFacet.get_dense_submatrix(
            hict_file,
            resolution,
            start_row_incl,
            start_col_incl,
            end_row_excl,
            end_col_excl,
            units=QueryLengthUnit.BINS,
            exclude_hidden_contigs=False
        )[0]
        assert np.array_equal(
            my_dense, again_my_dense), "Non-determinism in query results detected??"
    assert (
        np.array_equal(cooler_dense, my_dense)
    ), "Dense square submatrices returned by Cooler and HiCT should be equal"
    del cooler_dense
    del my_dense
    del cooler_file
    gc.collect()


@settings(
    max_examples=500,
    deadline=30000,
    derandomize=True,
    report_multiple_bugs=True,
    suppress_health_check=(
        HealthCheck.filter_too_much,
        HealthCheck.data_too_large
    )
)
@given(
    resolution=st.sampled_from(resolutions_mcool),
    start_row_incl_bp=st.integers(min_value=0, max_value=total_bp_length),
    start_col_incl_bp=st.integers(min_value=0, max_value=total_bp_length),
    query_size_row=st.sampled_from([1, 2, 3, 10, 100, 1000]),
    query_size_col=st.sampled_from([1, 2, 3, 5, 64, 127, 512])
)
def test_compare_rectangular_queries_with_cooler(
    resolution,
    start_row_incl_bp,
    start_col_incl_bp,
    query_size_row,
    query_size_col
):
    return compare_rectangular_queries_with_cooler(
        resolution,
        start_row_incl_bp,
        start_col_incl_bp,
        query_size_row,
        query_size_col
    )


@settings(
    max_examples=500,
    deadline=30000,
    derandomize=True,
    report_multiple_bugs=True,
    suppress_health_check=(
        HealthCheck.filter_too_much,
        HealthCheck.data_too_large
    )
)
@given(
    resolution=st.sampled_from(resolutions_mcool),
    start_row_incl_bp=st.integers(min_value=0, max_value=total_bp_length),
    start_col_incl_bp=st.integers(min_value=0, max_value=total_bp_length),
    query_size_row=st.sampled_from([1, 2, 3, 5]),
    query_size_col=st.sampled_from([1, 2, 3, 5])
)
def test_compare_small_rectangular_queries_with_cooler(
    resolution,
    start_row_incl_bp,
    start_col_incl_bp,
    query_size_row,
    query_size_col
):
    return compare_rectangular_queries_with_cooler(
        resolution,
        start_row_incl_bp,
        start_col_incl_bp,
        query_size_row,
        query_size_col
    )


def compare_rectangular_queries_with_cooler(
    resolution,
    start_row_incl_bp,
    start_col_incl_bp,
    query_size_row,
    query_size_col
):
    matrix_size_bins = ContactMatrixFacet.get_matrix_size_bins(
        hict_file, resolution)
    start_row_incl = (start_row_incl_bp // resolution) % matrix_size_bins
    start_col_incl = (start_col_incl_bp // resolution) % matrix_size_bins
    end_row_excl = start_row_incl + query_size_row
    end_col_excl = start_col_incl + query_size_col
    if start_row_incl > end_row_excl:
        start_row_incl, end_row_excl = end_row_excl, start_row_incl
    if start_col_incl > end_col_excl:
        start_col_incl, end_col_excl = end_col_excl, start_col_incl
    cooler_file: cooler.Cooler = cooler.Cooler(
        "{}::/resolutions/{}".format(str(mcool_file_path), resolution))
    cooler_matrix_selector: cooler.api.RangeSelector2D = cooler_file.matrix(
        field='count', balance=False)
    cooler_dense: np.ndarray = cooler_matrix_selector[start_row_incl:end_row_excl,
                                                      start_col_incl:end_col_excl]
    with hict_file_lock.gen_rlock():
        my_dense = ContactMatrixFacet.get_dense_submatrix(
            hict_file,
            resolution,
            start_row_incl,
            start_col_incl,
            end_row_excl,
            end_col_excl,
            units=QueryLengthUnit.BINS,
            exclude_hidden_contigs=False
        )[0]
    my_dense = np.pad(my_dense, [(0, query_size_row-my_dense.shape[0]), (0,
                      query_size_col-my_dense.shape[1])], mode='constant', constant_values=0)
    assert (
        my_dense.shape == (query_size_row, query_size_col)
    ), f"Matrix shape {my_dense.shape} should be equal to that of query: {(query_size_row, query_size_col)}, whereas cooler returned {cooler_dense.shape}"
    assert (
        np.array_equal(cooler_dense, my_dense)
    ), "Dense rectangular submatrices returned by Cooler and HiCT should be equal"
    del cooler_dense
    del my_dense
    del cooler_file
    gc.collect()


@settings(
    max_examples=500,
    deadline=30000,
    derandomize=True,
    report_multiple_bugs=True,
    suppress_health_check=(
        HealthCheck.filter_too_much,
        HealthCheck.data_too_large
    )
)
@given(
    resolution=st.sampled_from(resolutions_mcool),
    start_row_incl_bp=st.integers(min_value=0, max_value=total_bp_length),
    start_col_incl_bp=st.integers(min_value=0, max_value=total_bp_length),
    end_row_excl_bp=st.integers(min_value=0, max_value=total_bp_length),
    end_col_excl_bp=st.integers(min_value=0, max_value=total_bp_length)
)
def test_hict_file_should_be_symmetric(
    resolution,
    start_row_incl_bp,
    start_col_incl_bp,
    end_row_excl_bp,
    end_col_excl_bp,
):
    matrix_size_bins = ContactMatrixFacet.get_matrix_size_bins(
        hict_file,
        resolution
    )
    start_row_incl = (start_row_incl_bp // resolution) % matrix_size_bins
    start_col_incl = (start_col_incl_bp // resolution) % matrix_size_bins
    end_row_excl = (end_row_excl_bp // resolution) % matrix_size_bins
    end_col_excl = (end_col_excl_bp // resolution) % matrix_size_bins
    if start_row_incl > end_row_excl:
        start_row_incl, end_row_excl = end_row_excl, start_row_incl
    if start_col_incl > end_col_excl:
        start_col_incl, end_col_excl = end_col_excl, start_col_incl
    if end_row_excl - start_row_incl > 2048:
        end_row_excl = start_row_incl + \
            ((end_row_excl - start_row_incl) % 2048)
    if end_col_excl - start_col_incl > 2048:
        end_col_excl = start_col_incl + \
            ((end_col_excl - start_col_incl) % 2048)
    with hict_file_lock.gen_rlock():
        plain_dense = ContactMatrixFacet.get_dense_submatrix(
            hict_file,
            resolution,
            start_row_incl,
            start_col_incl,
            end_row_excl,
            end_col_excl,
            units=QueryLengthUnit.BINS,
            exclude_hidden_contigs=False
        )[0]
        transposed_dense = ContactMatrixFacet.get_dense_submatrix(
            hict_file,
            resolution,
            start_col_incl,
            start_row_incl,
            end_col_excl,
            end_row_excl,
            units=QueryLengthUnit.BINS,
            exclude_hidden_contigs=False
        )[0]
    assert (
        np.array_equal(plain_dense, transposed_dense.T)
    ), "HiC contact matrix returned by HiCT should be symmetric"
    del plain_dense
    del transposed_dense
    gc.collect()
