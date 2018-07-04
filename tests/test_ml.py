#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `ml` package."""

import pytest

from click.testing import CliRunner

from ml import np
from ml import cli
from ml.rbms import BernoulliRBM
from ml.rbms.core import RBM
from ml.rbms.rbm import SimpleBernoulliRBM
from ml.rbms.gaussian import GaussianRBM
from ml.datasets import mnist


RANDOM_SEED = 42


@pytest.fixture
def mnist_data():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    return mnist.load("tests/data/mnist-original.mat")


def rbm_verify_shapes(model, v, batch_size, visible_size, hidden_size):
    # verify shapes of different computations
    p_h = model.proba_hidden(v)
    h = model.sample_hidden(v)
    proba_v = model.proba_visible(h, v=v)

    assert p_h.shape == (batch_size, hidden_size)
    assert h.shape == (batch_size, hidden_size)
    assert proba_v.shape == (batch_size, visible_size)

    v_new = model.sample_visible(h)
    p_v_new = model.proba_visible(h, v=v_new)
    assert v_new.shape == (batch_size, visible_size)
    assert p_v_new.shape == (batch_size, visible_size)

    # energy
    energy = model.energy(v, h)
    assert energy.shape == (batch_size, )

    # step
    model.step(v, k=1)


def rbm_train_single_sample(model, v, steps=100):
    nll = model.free_energy(v)

    model.fit(v, batch_size=1, num_epochs=25, show_progress=False)
    # for i in range(steps):
    #     model.step(v)

    nll_new = model.free_energy(v)

    # wan log-likelihood to increase
    # => negative log-likelihood should decrease
    assert nll_new < nll
    print(f"Initial nll: {nll}")
    print(f"Final nll: {nll_new}")


def test_gaussian_rbm(mnist_data):
    np.random.seed(RANDOM_SEED)
    X, _, _, _ = mnist_data
    batch_size = 10
    visible_size = X.shape[1]

    # Gaussian RBMs are VERY sensitive to params on MNIST
    # and `hidden_size == 250` just happens to work.
    hidden_size = 300

    X = (X - np.mean(X, axis=0) / (np.std(X, axis=0)
                                   + np.finfo(np.float32).eps))
    X[np.isnan(X)] = 1.0
    v = X[:batch_size].astype(np.float64)
    model = GaussianRBM(visible_size, hidden_size,
                        estimate_visible_sigma=False)

    # verify shapes
    rbm_verify_shapes(model, v, batch_size, visible_size, hidden_size)

    # train :)
    rbm_train_single_sample(model, v)


def test_batch_bernoulli(mnist_data):
    np.random.seed(RANDOM_SEED)
    X, _, _, _ = mnist_data
    batch_size = 10
    hidden_size = 100
    visible_size = X.shape[1]

    X = np.clip(X, 0, 1)
    v = X[:batch_size]

    # `BernoulliRBM`
    rbm1 = BernoulliRBM(visible_size, hidden_size)
    rbm_verify_shapes(rbm1, v, batch_size, visible_size, hidden_size)
    rbm_train_single_sample(rbm1, v)

    # `SimpleBernoulliRBM`
    rbm2 = SimpleBernoulliRBM(visible_size, hidden_size)
    # rbm_verify_shapes(rbm2, v, batch_size, visible_size, hidden_size)
    rbm_train_single_sample(rbm2, v)

    # `RBM` with bernoulli rvs.
    rbm3 = RBM(visible_size, hidden_size,
               visible_type='bernoulli',
               hidden_type='bernoulli')
    rbm_verify_shapes(rbm3, v, batch_size, visible_size, hidden_size)
    rbm_train_single_sample(rbm3, v)

    # compare the 3 implementations
    rbm1.v_bias, rbm1.h_bias, rbm1.W = rbm1.v_bias, rbm1.h_bias, rbm1.W
    rbm2.v_bias, rbm2.h_bias, rbm2.W = rbm1.v_bias, rbm1.h_bias, rbm1.W
    rbm3.v_bias, rbm3.h_bias, rbm3.W = rbm1.v_bias, rbm1.h_bias, rbm1.W

    h = rbm1.sample_hidden(v)
    assert np.all(rbm1.proba_hidden(v) == rbm3.proba_hidden(v))
    assert np.all(rbm1.proba_visible(h) == rbm3.proba_visible(h))
    assert np.all(rbm1.free_energy(v) == rbm3.free_energy(v))

    for i in range(v.shape[0]):
        assert np.all(rbm1.proba_hidden(v[i].reshape(1, -1))
                      == rbm2.proba_hidden(v[i]))
        assert np.all(rbm1.proba_visible(h[i].reshape(1, -1))
                      == rbm2.proba_visible(h[i]))
        assert np.all(rbm1.free_energy(v[i].reshape(1, -1))
                      == rbm2.free_energy(v[i]))

# def test_command_line_interface():
#     """Test the CLI."""
#     runner = CliRunner()
#     result = runner.invoke(cli.main)
#     assert result.exit_code == 0
#     assert 'ml.cli.main' in result.output
#     help_result = runner.invoke(cli.main, ['--help'])
#     assert help_result.exit_code == 0
#     assert '--help  Show this message and exit.' in help_result.output
