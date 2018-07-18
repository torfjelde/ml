#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `ml.sampling` package."""

import pytest

from scipy import stats

from ml import np
from ml import sampling


@pytest.fixture
def samples():
    # data
    samples_num = 1000
    xs = np.random.normal(loc=0.0, size=samples_num)

    return xs


@pytest.fixture
def proba():
    def proba_(xs, loc=0.0):
        return stats.norm.pdf(xs, loc)

    return proba_


@pytest.fixture
def proposal_proba():
    def proposal_proba_(x, y):
        return stats.norm.pdf(x, loc=y)
    return proposal_proba_


@pytest.fixture
def proposal_sample():
    def proposal_sample_(x):
        return stats.norm.rvs(loc=x)
    return proposal_sample_


def test_metropolis_hastings(samples,
                             proba,
                             proposal_proba,
                             proposal_sample):
    # initialize kernel
    kernel = sampling.MetropolisHastingsKernel(
        proba,
        proposal_sample,
        proposal_proba
    )

    # test kernel
    state = 1.0
    state = kernel.sample(state)
    print(state)

    # test sampler
    sampler = sampling.Sampler(kernel, show_progress=True)
    trace = sampler.run(initial=state)

    # verify the sampler produces reasonable results
    target_mean = np.mean(samples)
    target_std = np.std(samples)
    assert np.abs(np.mean(trace) - target_mean) < 0.1
    assert np.abs(np.std(trace) - target_std) < 0.1
