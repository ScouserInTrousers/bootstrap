#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from builtins import dict, int, map, range, zip
from itertools import tee
import numpy as np


def antithetic(data):
    """
    First pass, estimator is np.mean
    """
    x = np.ascontiguousarray(data).ravel().sort()
    return x


def estimate(data_gen, estimators):
    """
    For every statistic in `estimators`, that statistic will be calculated
    for every sample in `data_gen`. In order to support variable number of
    estimators, for each estimator an itertools.tee object is created.
    Returns a dictionary of the form
        statistic: sampling distribution of the statistic
    Args:
        data_gen (PyGenObject): generator of samples (usually np.array)
        estimators (iterable of callables): the functions of the data to calc
    Returns:
        (dict): keys = names of estimators; values = resulting arrays of calc
    """
    tees = tee(data_gen, len(estimators) + 1)
    # 1 more than specified is created to get the length of `data_gen`; i.e. #
    # of samples in `data_gen`
    len_gen, sample_gens = tees[0], tees[1:]
    del tees
    num_samples = sum(1 for _ in len_gen)
    return {
        estimator.__name__: np.fromiter(map(estimator, sample_gen),
                                        dtype=np.float, count=num_samples)
        for estimator, sample_gen in zip(estimators, sample_gens)
    }


def resample(data, B):
    """
    Return `B` bootstrap samples of `data` as a generator. The motivation for
    this is that it may be desirable to calculate several statistics of the
    data; returning as a generator lends itself to itertools.tee.
    Args:
        data (array-like): the numerical values to be resampled. N.b. will be
            cast to Numpy array
        B (int): the amount of bootstrap pseudo-datasets to create. If B <= 1,
            returns generator of length 1 with original data as Numpy array
    Returns:
        (PyGenObject): generator of Numpy arrays the same size as `data`
    """
    try:
        to_be_sampled = np.ascontiguousarray([x for x in data]).ravel()
    except TypeError:
        raise ValueError("Dataset must support iteration")
    else:
        if not to_be_sampled.size:
            raise ValueError("Dataset to be resampled must have positive "
                             "cardinality")
    if B <= 1:
        return (original_data for original_data in [to_be_sampled])

    return (np.random.choice(a=to_be_sampled,
                             size=to_be_sampled.size,
                             replace=True) for _ in range(0, B))
