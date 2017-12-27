from builtins import next
from functions import resample
from hypothesis import given, strategies as st
from hypothesis.extra import numpy
import pytest as pt


# DATA AS ITERABLE TESTS #
@given(st.iterables(max_size=0), st.integers())
def test_resample_empty_data_raises_value_error(data, B):
    with pt.raises(ValueError):
        resample(data, B)


@given(data=st.iterables(min_size=1, elements=st.floats()),
       B=st.integers(min_value=0, max_value=1))
def test_resample_nonempty_data_passed_B_leq_1(data, B):
    g = resample(data, B)
    next(g)
    # A second next() call should raise StopIteration as it should have length 1
    with pt.raises(StopIteration):
        next(g)


@given(data=st.iterables(elements=st.floats(), min_size=1, max_size=1e2),
       B=st.integers(min_value=1, max_value=10))
def test_resample_nonempty_data_passed_B_geq_1(data, B):
    # If B >= 1, should return a generator with `B` elements
    assert B == sum(1 for _ in resample(data, B))


@given(data=st.iterables(elements=st.floats(), min_size=1, max_size=1e2),
       B=st.integers(min_value=1, max_value=10))
def test_resample_nonempty_data_passed_outputs_gen_elements_right_size(data, B):
    length_of_data = sum(1 for _ in data)
    assert all(len(x) == length_of_data for x in resample(data, B))


# NUMPY TESTS #
@given(data=numpy.arrays(dtype=numpy.floating_dtypes(),
                         shape=st.tuples(st.integers(min_value=1,
                                                     max_value=10),
                                         st.integers(min_value=1,
                                                     max_value=10)
                                         )),
       B=st.integers(min_value=1, max_value=10))
def test_resample_numpy_data_outputs_correct_generator_size(data, B):
    assert B == sum(1 for _ in resample(data, B))


@given(data=numpy.arrays(dtype=numpy.floating_dtypes(),
                         shape=st.tuples(st.integers(min_value=1,
                                                     max_value=10),
                                         st.integers(min_value=1,
                                                     max_value=10)
                                         )),
       B=st.integers(min_value=1, max_value=10))
def test_resample_numpy_data_outputs_same_sized_arrays(data, B):
    pass