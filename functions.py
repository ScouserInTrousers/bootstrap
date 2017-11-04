#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools as it
import numpy as np


def antithetic(data):
    return np.ascontiguousarray(data)


def binomial(n, r):
    """ Binomial coefficient, nCr, aka the "choose" function
        n! / (r! * (n - r)!)
    """
    p = 1
    for i in xrange(1, min(r, n - r) + 1):
        p *= n
        p //= i
        n -= 1
    return p


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
    # no keywords due to low-level C API
    # https://stackoverflow.com/questions/24463202/typeerror-get-takes-no-keyword-arguments
    tees = it.tee(data_gen, len(estimators) + 1)
    # 1 more than specified is created to get the length of `data_gen`; i.e. #
    # of samples in `data_gen`
    len_gen, sample_gens = tees[0], tees[1:]
    del tees
    num_samples = sum(1 for _ in len_gen)  # or num_samples = len(list(len_gen))
    return {
        estimator.__name__: np.fromiter(it.imap(estimator, sample_gen),
                                        dtype=np.float, count=num_samples)
        for estimator, sample_gen in it.izip(estimators, sample_gens)
    }


def resample(data, B):
    """
    Return `B` bootstrap samples of `data` as a generator. The motivation for
    this is that it may be desirable to calculate several statistics of the
    data; returning as a generator lends itself to itertools.tee.
    Args:
        data (array-like): the numerical values to be resampled. N.b. will be
            cast to Numpy array
        B (int): the amount of bootstrap samples to produce
    Returns:
        (PyGenObject): generator of Numpy arrays the same size as `data`
    """
    to_be_sampled = data.ravel()
    for _ in xrange(0, B):
        yield np.random.choice(a=to_be_sampled,
                               size=data.shape[0],
                               replace=True)
