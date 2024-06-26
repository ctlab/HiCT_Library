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

from typing import List, Optional
from hict.core.common import ScaffoldDescriptor
from hict.core.scaffold_tree import ScaffoldTree
import numpy as np
from hypothesis import given, settings, strategies as st, HealthCheck
import multiprocessing
import multiprocessing.managers

mp_manager: multiprocessing.managers.SyncManager = multiprocessing.Manager()

mp_rlock = mp_manager.RLock()


def get_lock():
    return mp_rlock


# random.seed(int(time.time()))

def build_tree(
    scaffold_descriptors: List[ScaffoldDescriptor],
    scaffold_lengths: List[int],
    empty_space_lengths: List[int],
    mp_manager: Optional[multiprocessing.managers.SyncManager]
) -> ScaffoldTree:
    tree = ScaffoldTree(
        assembly_length_bp=np.int64(
            sum(scaffold_lengths)+sum(empty_space_lengths)),
        mp_manager=mp_manager
    )
    last_pos: np.int64 = np.int64(empty_space_lengths[0])
    for i, sd in enumerate(scaffold_descriptors):
        tree.add_scaffold(
            last_pos,
            last_pos+scaffold_lengths[i],
            sd
        )
        if last_pos > 0:
            sd_pre = tree.get_scaffold_at_bp(last_pos-1)
            sd_at = tree.get_scaffold_at_bp(last_pos)
            sd_next = tree.get_scaffold_at_bp(last_pos+1)
            sd_preend = tree.get_scaffold_at_bp(last_pos+scaffold_lengths[i]-1)
            sd_end = tree.get_scaffold_at_bp(last_pos+scaffold_lengths[i])
            sd_post = tree.get_scaffold_at_bp(last_pos+scaffold_lengths[i]+1)

            # PRE
            if not (sd_pre is None or sd_pre != sd):
                print("Going to fetch that one again")
                tree.get_scaffold_at_bp(last_pos-1)

            assert (
                sd_pre is None or sd_pre != sd
            ), "Before required scaffold's position there should be no that scaffold"

            # AT
            if not ((sd_at is not None) and (sd_at == sd)):
                print("Going to fetch that one again")
                tree.get_scaffold_at_bp(last_pos)

            assert (
                sd_at is not None
            ), "At starting scaffold position scaffold should be present"

            assert (
                sd_at == sd
            ), "At starting scaffold position should be the added scaffold itself"

            # NEXT
            if scaffold_lengths[i] > 1:
                if not ((sd_next is not None) and (sd_next == sd)):
                    print("Going to fetch that one again")
                    tree.get_scaffold_at_bp(last_pos+1)

                assert (
                    sd_next is not None
                ), "Scaffold with length>1 should span over the next position from its start"

                assert (
                    sd_next == sd
                ), "Scaffold with length>1 should span itself over the next position from its start"
            else:
                if not (sd_next is None or sd_next != sd):
                    print("Going to fetch that one again")
                    tree.get_scaffold_at_bp(last_pos+1)

                assert (
                    sd_next is None or sd_next != sd
                ), "Scaffold with length<=1 should not span over the next position from its start"

            # PRE-END
            if not ((sd_preend is not None) and (sd_preend == sd)):
                print("Going to fetch that one again")
                tree.get_scaffold_at_bp(
                    last_pos+last_pos+scaffold_lengths[i]-1)

            assert (
                sd_preend is not None
            ), "Scaffold should cover the position previous to its end"

            assert (
                sd_preend == sd
            ), "Scaffold itself should cover the position previous to its end"

            # END
            if not (sd_end is None or sd_pre != sd):
                print("Going to fetch that one again")
                tree.get_scaffold_at_bp(
                    last_pos+last_pos+scaffold_lengths[i]-1)

            assert (
                sd_end is None or sd_end != sd
            ), "Since scaffold covers positions [start, start+length), it should not be present at its end bp"

            # POST
            if not (sd_post is None or sd_post != sd):
                print("Going to fetch that one again")
                tree.get_scaffold_at_bp(
                    last_pos+last_pos+scaffold_lengths[i]+1)

            assert (
                sd_post is None or sd_post != sd
            ), "Since scaffold covers positions [start, start+length), to the right from its end border"

        if not (tree.get_scaffold_at_bp(last_pos+scaffold_lengths[i]) == None):
            print("Going to fetch that one again")
            tree.get_scaffold_at_bp(last_pos+scaffold_lengths[i])

        assert (
            tree.get_scaffold_at_bp(last_pos+scaffold_lengths[i]) == None
        ), "Accoring to the tree insertion algorithm, no scaffolds should be present at the end of the newly added"

        if not (tree.get_scaffold_at_bp(last_pos+scaffold_lengths[i]+1) == None):
            print("Going to fetch that one again")
            tree.get_scaffold_at_bp(last_pos+scaffold_lengths[i]+1)

        assert (
            tree.get_scaffold_at_bp(last_pos+scaffold_lengths[i]+1) == None
        ), "Accoring to the tree insertion algorithm, no scaffolds should be present after the newly added"

        last_pos += scaffold_lengths[i]+empty_space_lengths[1+i]
    return tree


@settings(
    max_examples=10000,
    deadline=30000,
    derandomize=True,
    report_multiple_bugs=False,
    suppress_health_check=(
        HealthCheck.filter_too_much,
        HealthCheck.data_too_large
    )
)
@given(dummy_param=st.integers())
def test_unit_1(dummy_param: int):
    tree = ScaffoldTree(2)
    tree.rescaffold(0, 1)

    assert (
        tree.get_scaffold_at_bp(0) is not None
    )
    assert (
        tree.get_scaffold_at_bp(1) is None
    )


@settings(
    max_examples=10000,
    deadline=30000,
    derandomize=True,
    report_multiple_bugs=False,
    suppress_health_check=(
        HealthCheck.filter_too_much,
        HealthCheck.data_too_large
    )
)
@given(dummy_param=st.integers())
def test_unit_2(dummy_param: int):
    tree = ScaffoldTree(4)
    tree.rescaffold(0, 1)
    tree.rescaffold(2, 3)

    pos0 = tree.get_scaffold_at_bp(0)
    pos1 = tree.get_scaffold_at_bp(1)
    pos2 = tree.get_scaffold_at_bp(2)
    pos3 = tree.get_scaffold_at_bp(3)

    assert (pos0 is not None)
    assert (pos1 is None)
    assert (pos2 is not None)
    assert (pos3 is None)


@settings(
    max_examples=10000,
    deadline=30000,
    derandomize=True,
    report_multiple_bugs=False,
    suppress_health_check=(
        HealthCheck.filter_too_much,
        HealthCheck.data_too_large
    )
)
@given(dummy_param=st.integers())
def test_unit_3(dummy_param: int):
    tree = build_tree(
        [
            ScaffoldDescriptor.make_scaffold_descriptor(1, "s1"),
            ScaffoldDescriptor.make_scaffold_descriptor(2, "s2"),
        ],
        [1, 1],
        [0, 1, 1],
        mp_manager=None
    )


@settings(
    max_examples=10000,
    deadline=30000,
    derandomize=True,
    report_multiple_bugs=False,
    suppress_health_check=(
        HealthCheck.filter_too_much,
        HealthCheck.data_too_large
    )
)
@given(
    scaffold_descriptors=st.lists(
        st.builds(
            ScaffoldDescriptor.make_scaffold_descriptor,
            scaffold_id=st.integers(0, 10000),
            scaffold_name=st.text(max_size=10),
            spacer_length=st.integers(min_value=500, max_value=501),
        ),
        max_size=2,
        unique_by=(lambda sd: sd.scaffold_id)
    ),
    scaffold_size_bound=st.integers(min_value=2, max_value=3),
    empty_size_bound=st.integers(min_value=2, max_value=3)
)
def test_build_tree_small(
    scaffold_descriptors: List[ScaffoldDescriptor],
    scaffold_size_bound: int,
    empty_size_bound: int
):
    generic_test_build_tree(
        scaffold_descriptors,
        scaffold_size_bound,
        empty_size_bound
    )


@settings(
    max_examples=10000,
    deadline=30000,
    derandomize=True,
    report_multiple_bugs=True,
    suppress_health_check=(
        HealthCheck.filter_too_much,
        HealthCheck.data_too_large
    )
)
@given(
    scaffold_descriptors=st.lists(
        st.builds(
            ScaffoldDescriptor.make_scaffold_descriptor,
            scaffold_id=st.integers(0, 10000),
            scaffold_name=st.text(max_size=10),
            spacer_length=st.integers(min_value=500, max_value=501),
        ),
        unique_by=(lambda sd: sd.scaffold_id)
    ),
    scaffold_size_bound=st.integers(min_value=2, max_value=100000),
    empty_size_bound=st.integers(min_value=2, max_value=100000)
)
def test_build_tree(
    scaffold_descriptors: List[ScaffoldDescriptor],
    scaffold_size_bound: int,
    empty_size_bound: int
):
    generic_test_build_tree(
        scaffold_descriptors,
        scaffold_size_bound,
        empty_size_bound
    )


def generic_test_build_tree(
    scaffold_descriptors: List[ScaffoldDescriptor],
    scaffold_size_bound: int,
    empty_size_bound: int
):
    scaffold_lengths = list(np.random.randint(
        1,
        scaffold_size_bound,
        size=len(scaffold_descriptors),
        dtype=np.int64)
    )
    empty_space_lengths = list(np.random.randint(
        0,
        empty_size_bound,
        size=1+len(scaffold_descriptors),
        dtype=np.int64
    ))

    tree = build_tree(
        scaffold_descriptors=scaffold_descriptors,
        scaffold_lengths=scaffold_lengths,
        empty_space_lengths=empty_space_lengths,
        mp_manager=None
    )

    total_assembly_length = sum(scaffold_lengths)+sum(empty_space_lengths)

    assert (
        tree.root.subtree_length_bp == total_assembly_length
    ), "Tree does not cover all assembly?"

    nodes: List[ScaffoldTree.Node] = []

    def traverse_fn(node: ScaffoldTree.Node):
        nodes.append(node)

    tree.traverse(traverse_fn)

    expected_descriptors = sorted(
        scaffold_descriptors,
        key=lambda d: d.scaffold_id
    )
    actual_descriptors = sorted(
        map(
            lambda n: n.scaffold_descriptor,
            filter(
                lambda n: n.scaffold_descriptor is not None,
                nodes
            )
        ),
        key=lambda d: d.scaffold_id
    )

    assert (
        expected_descriptors == actual_descriptors
    ), "Not all descriptors are present after building tree??"

    nodes_index: int = 0

    for (esl, sdl) in zip(empty_space_lengths[:-1], scaffold_lengths):
        if esl > 0:
            assert (
                nodes[nodes_index].scaffold_descriptor is None
            ), "Empty space and scaffolds should interleave: empty space case"
            assert (
                nodes[nodes_index].length_bp == esl
            ), "Empty space length should be as requested"
            nodes_index += 1
        assert (sdl > 0), "Scaffolds must have non-negative lengths"
        assert (
            nodes[nodes_index].scaffold_descriptor is not None
        ), "Empty space and scaffolds should interleave: scaffold case"
        assert (
            nodes[nodes_index].length_bp == sdl
        ), "Scaffold length should be as requested"
        nodes_index += 1

    esl = empty_space_lengths[-1]

    if esl > 0:
        assert (
            nodes[nodes_index].scaffold_descriptor is None
        ), "Empty space and scaffolds should interleave: empty space case"
        assert (
            nodes[nodes_index].length_bp == esl
        ), "Empty space length should be as requested"
        nodes_index += 1


@settings(
    max_examples=10000,
    deadline=30000,
    derandomize=True,
    report_multiple_bugs=True,
    suppress_health_check=(
        HealthCheck.filter_too_much,
        HealthCheck.data_too_large
    )
)
@given(
    scaffold_descriptors=st.lists(
        st.builds(
            ScaffoldDescriptor.make_scaffold_descriptor,
            scaffold_id=st.integers(0, 10000),
            scaffold_name=st.text(max_size=10),
            spacer_length=st.integers(min_value=500, max_value=501),
        ),
        unique_by=(lambda sd: sd.scaffold_id)
    ),
    scaffold_size_bound=st.integers(min_value=2, max_value=100000),
    empty_size_bound=st.integers(min_value=2, max_value=100000),
    left_size=st.integers(0, 100000000)
)
def test_split_tree(
    scaffold_descriptors: List[ScaffoldDescriptor],
    scaffold_size_bound: int,
    empty_size_bound: int,
    left_size: int
):
    scaffold_lengths = list(np.random.randint(
        1,
        scaffold_size_bound,
        size=len(scaffold_descriptors),
        dtype=np.int64)
    )
    empty_space_lengths = list(np.random.randint(
        0,
        empty_size_bound,
        size=1+len(scaffold_descriptors),
        dtype=np.int64
    ))

    tree = build_tree(
        scaffold_descriptors=scaffold_descriptors,
        scaffold_lengths=scaffold_lengths,
        empty_space_lengths=empty_space_lengths,
        mp_manager=None
    )

    total_assembly_length = sum(scaffold_lengths)+sum(empty_space_lengths)

    left_size = min(left_size, total_assembly_length)

    with tree.root_lock.gen_rlock():
        (l, r) = ScaffoldTree.Node.split_bp(tree.root,
                                            np.int64(left_size), include_equal_to_the_left=True)

    ls = (l.subtree_length_bp if l is not None else np.int64(0))

    rightmost = ScaffoldTree.Node.rightmost(l)

    if rightmost is not None and rightmost.scaffold_descriptor is not None:
        assert (
            ls >= left_size
        ), "Split size is greater than expected??"
    else:
        assert (
            ls == left_size
        ), "Split size is not equal to requested when it ends not with scaffold??"


@settings(
    max_examples=10000,
    deadline=30000,
    derandomize=True,
    report_multiple_bugs=True,
    suppress_health_check=(
        HealthCheck.filter_too_much,
        HealthCheck.data_too_large
    )
)
@given(
    scaffold_descriptors=st.lists(
        st.builds(
            ScaffoldDescriptor.make_scaffold_descriptor,
            scaffold_id=st.integers(0, 10000),
            scaffold_name=st.text(max_size=10),
            spacer_length=st.integers(min_value=500, max_value=501),
        ),
        unique_by=(lambda sd: sd.scaffold_id),
        max_size=10,
    ),
    scaffold_size_bound=st.integers(min_value=2, max_value=100),
    empty_size_bound=st.integers(min_value=2, max_value=100),
)
def test_get_scaffold_at_bp(
    scaffold_descriptors: List[ScaffoldDescriptor],
    scaffold_size_bound: int,
    empty_size_bound: int,
):
    scaffold_lengths = list(np.random.randint(
        1,
        scaffold_size_bound,
        size=len(scaffold_descriptors),
        dtype=np.int64)
    )
    empty_space_lengths = list(np.random.randint(
        0,
        empty_size_bound,
        size=1+len(scaffold_descriptors),
        dtype=np.int64
    ))

    tree = build_tree(
        scaffold_descriptors=scaffold_descriptors,
        scaffold_lengths=scaffold_lengths,
        empty_space_lengths=empty_space_lengths,
        mp_manager=None
    )

    total_assembly_length = sum(scaffold_lengths)+sum(empty_space_lengths)

    position_bp: np.int64 = np.int64(0)

    for i, (sd, sl, el) in enumerate(
        zip(
            scaffold_descriptors,
            scaffold_lengths,
            empty_space_lengths[:-1],
        )
    ):
        if el > 0:
            ns = tree.get_scaffold_at_bp(position_bp)
            assert (
                ns is None
            ), "Empty position should not contain a scaffold"
        position_bp += el
        if sl > 0:
            sn = tree.get_scaffold_at_bp(position_bp)
            assert (
                sn is not None
            ), "Scaffold descriptor should be present"
            assert (
                sn.scaffold_id == sd.scaffold_id
            ), "The same scaffold should be there"
        position_bp += sl
        rn = tree.get_scaffold_at_bp(position_bp)
        assert (
            (rn is None)
            or
            (rn.scaffold_id != sd.scaffold_id)
        ), "Right border should not be included to the scaffold"
