import abc
import logging

from enum import Enum
from tqdm import tqdm

from .. import np
from ..functions import sigmoid, dot_batch

from .rbm import GenericRBM

_log = logging.getLogger("ml")


class UnitType(Enum):
    GAUSSIAN = 1
    BERNOULLI = 2


def bernoulli_from_probas(probas):
    rands = np.random.random(size=probas.shape)
    return (probas > rands).astype(int)


def log_sum_exp(x, alpha):
    return alpha + np.exp(x - alpha).sum().log()


class SimpleRBM:
    """Restricted Boltzmann Machine with either bernoulli or Gaussian
visible/hidden units.

    """
    def __init__(self, num_visible, num_hidden,
                 visible_type='bernoulli', hidden_type='bernoulli',
                 estimate_visible_sigma=False, estimate_hidden_sigma=False):
        super(SimpleRBM, self).__init__()
        self.num_visible = num_visible
        self.num_hidden = num_hidden

        self.visible_type = UnitType.GAUSSIAN if visible_type == 'gaussian' else UnitType.BERNOULLI
        self.hidden_type = UnitType.GAUSSIAN if hidden_type == 'gaussian' else UnitType.BERNOULLI

        self.estimate_visible_sigma = estimate_visible_sigma
        self.estimate_hidden_sigma = estimate_hidden_sigma

        self.v_bias, self.h_bias, self.v_sigma, self.h_sigma, self.W = self.initialize(
            num_visible,
            num_hidden
        )

        self._variables = [self.v_bias, self.h_bias, self.W]
        if self.estimate_visible_sigma:
            self._variables.append(self.v_sigma)
        if self.estimate_hidden_sigma:
            self._variables.append(self.h_sigma)

    @property
    def variables(self):
        return self._variables

    @staticmethod
    def initialize(num_visible, num_hidden):
        # biases for visible and hidden, respectively
        v_bias = np.zeros(num_visible)
        h_bias = np.zeros(num_hidden)

        # weight matrix
        W = np.random.normal(0.0, 0.01, (num_visible, num_hidden))

        # variances
        v_sigma = np.ones(num_visible)
        h_sigma = np.ones(num_hidden)

        return v_bias, h_bias, v_sigma, h_sigma, W

    def energy(self, v, h):
        if self.visible_type == UnitType.BERNOULLI:
            visible = np.matmul(v, self.v_bias)
        elif self.visible_type == UnitType.GAUSSIAN:
            visible = ((v - self.v_bias) ** 2) / (self.v_sigma ** 2 + np.finfo(np.float32).eps)
            visible = 0.5 * np.sum(visible, axis=1)

        # term only dependent on hidden
        if self.hidden_type == UnitType.BERNOULLI:
            hidden = np.matmul(h, self.h_bias)
        elif self.hidden_type == UnitType.GAUSSIAN:
            hidden = ((h - self.h_bias) ** 2) / (self.h_sigma ** 2 + np.finfo(np.float32).eps)
            hidden = 0.5 * np.sum(hidden, axis=1)

        # "covariance" term
        # v^T W = sum_j( (v_j / sigma_j) W_{j \mu} )
        covariance = np.matmul(v, self.W)
        # v^T W h = sum_{\mu} h_{\mu} sum_j( (v_j / sigma_j) W_{j \mu} )
        covariance = dot_batch(h, covariance)

        return - (visible + hidden + covariance)

    def mean_visible(self, h):
        "Computes conditional expectation E[v | h]."
        mean = self.v_bias + self.v_sigma * np.matmul(h / self.h_sigma, self.W.T)
        if self.visible_type == UnitType.BERNOULLI:
            return sigmoid(mean)
        elif self.visible_type == UnitType.GAUSSIAN:
            return mean

    def mean_hidden(self, v):
        "Computes conditional expectation E[h | v]."
        mean = self.h_bias + self.h_sigma * np.matmul(v / self.v_sigma, self.W)
        if self.hidden_type == UnitType.BERNOULLI:
            return sigmoid(mean)
        elif self.hidden_type == UnitType.GAUSSIAN:
            return mean

    def sample_visible(self, h):
        mean = self.mean_visible(h)
        if self.visible_type == UnitType.BERNOULLI:
            # E[v | h] = p(v | h) for Bernoulli
            v = bernoulli_from_probas(mean)
        elif self.visible_type == UnitType.GAUSSIAN:
            v = np.random.normal(loc=mean, scale=self.v_sigma ** 2, size=mean.shape)
        else:
            raise ValueError(f"unknown type {self.visible_type}")

        return v

    def sample_hidden(self, v):
        mean = self.mean_hidden(v)
        if self.visible_type == UnitType.BERNOULLI:
            # E[v | h] = p(v | h) for Bernoulli
            h = bernoulli_from_probas(mean)
        elif self.visible_type == UnitType.GAUSSIAN:
            h = np.random.normal(loc=mean, scale=(self.h_sigma ** 2), size=(mean.shape))
        else:
            raise ValueError(f"unknown type {self.visible_type}")

        return h

    def proba_visible(self, h, v=None):
        mean = self.mean_visible(h)
        if self.visible_type == UnitType.BERNOULLI:
            # E[v | h] = p(v | h) for Bernoulli
            p = mean
        elif self.visible_type == UnitType.GAUSSIAN:
            z = np.clip((v - mean) ** 2 / (2.0 * self.v_sigma ** 2), -30.0, 30.0)
            p = (np.exp(z) / (np.sqrt(2 * np.pi) * self.v_sigma
                              + np.finfo(np.float32).eps))
        else:
            raise ValueError(f"unknown type {self.visible_type}")

        return p

    def proba_hidden(self, v, h=None):
        mean = self.mean_hidden(v)
        if self.hidden_type == UnitType.BERNOULLI:
            # E[v | h] = p(v | h) for Bernoulli
            p = mean
        elif self.hidden_type == UnitType.GAUSSIAN:
            z = np.clip((h - mean) ** 2 / (2.0 * self.h_sigma ** 2), -30.0, 30.0)
            p = (np.exp(z) / (np.sqrt(2 * np.pi) * self.h_sigma
                              + np.finfo(np.float32).eps))
        else:
            raise ValueError(f"unknown type {self.hidden_type}")

        return p

    def free_energy(self, v):
        if self.hidden_type == UnitType.BERNOULLI:
            hidden = self.h_bias + np.matmul((v / self.v_sigma), self.W)
            hidden = - np.sum(np.log(1.0 + np.exp(np.clip(hidden, -30, 30))), axis=1)
        elif self.visible_type == UnitType.GAUSSIAN:
            # TODO: Implement
            # Have the formulas, but gotta make sure yo!
            raise NotImplementedError()

        if self.visible_type == UnitType.BERNOULLI:
            visible = - np.matmul(v, self.v_bias)
        elif self.visible_type == UnitType.GAUSSIAN:
            visible = 0.5 * np.sum(((v - self.v_bias) ** 2)
                                   / (self.v_sigma ** 2 + np.finfo(np.float32).eps), axis=1)
        else:
            raise ValueError(f"unknown type {self.visible_type}")

        # sum across batch to obtain log of joint-likelihood
        return np.sum(hidden + visible)

    def free_energy_hidden(self, h):
        # FIXME: computation of visible is wrong
        raise NotImplementedError("computation of visible is wrong; implement pls")
        visible = - np.sum(self.mean_visible(h), axis=1)
        if self.hidden_type == UnitType.BERNOULLI:
            hidden = - np.matmul(h, self.h_bias)
        elif self.hidden_type == UnitType.GAUSSIAN:
            hidden = 0.5 * np.sum(((h - self.h_bias) ** 2)
                                  / (self.h_sigma ** 2 + np.finfo(np.float32).eps), axis=1)
        else:
            raise ValueError(f"unknown type {self.hidden_type}")

        # sum across batch to obtain log of joint-likelihood
        return np.sum(hidden + visible)

    def contrastive_divergence(self, v_0, k=1, metropolis=False):
        """Contrastive Divergence.

        Parameters
        ----------
        self: type
            description
        v_0: array-like
            Visible state to initialize the chain from.
        k: int
            Number of steps to use in CD-k.
        metropolis: bool, [default=True]
            If `True`, will use the Metropolis-Hastings Gibbs Block sampling method which
            first samples a new state, then accepts or rejects the state with some probability.
            Otherwise, samples will be drawn from the probability distribution
            of the model and accepted no matter what.

        Returns
        -------
        h_0, h, v_0, v: arrays
            `h_0` and `v_0` are the initial states for the hidden and visible units, respectively.
            `h` and `v` are the final states for the hidden and visible units, respectively.
        """
        v = v_0
        h_0 = self.sample_hidden(v)  # don't have `h` yet; random sample
        if metropolis:
            v = self.gibbs_sample_visible(h_0, v)
            h = self.gibbs_sample_hidden(v, h_0)
        else:
            v = self.sample_visible(h_0)
            h = self.sample_hidden(v)

        if k > 1:
            for t in range(k):
                if metropolis:
                    v = self.gibbs_sample_visible(h, v)
                    h = self.gibbs_sample_hidden(v, h)
                else:
                    v = self.sample_visible(h)
                    h = self.sample_hidden(v)

        return h_0, h, v_0, v

    def _update(self, grad, lr=0.1):
        for i in range(len(self.variables)):
            self.variables[i] -= lr * grad[i]

    def _apply_weight_decay(self, lmbda=0.01):
        for i in range(len(self.variables)):
            # default is gradient DEscent, so weight-decay also switches signs
            self.variables[i] += lmbda * self.variables[i]

    def step(self, v, k=1, lr=0.1, lmbda=0.0):
        "Performs a single gradient DEscent step on the batch `v`."
        # compute gradient for each observed visible configuration
        grad = self.grad(v, k=k)

        # update parameters
        self._update(grad, lr=lr)

        # possibly apply weight-decay
        if lmbda > 0.0:
            self._apply_weight_decay(lmbda=lmbda)

    def reconstruct(self, v, num_samples=100):
        samples = self.sample_visible(self.sample_hidden(v))
        for _ in range(num_samples - 1):
            samples += self.sample_visible(self.sample_hidden(v))

        probs = samples / num_samples

        return probs

    def grad(self, v, k=1):
        h_0, h_k, v_0, v_k = self.contrastive_divergence(v, k=k)
        # all expressions below using `v` or `mean_h` will contain
        # AT LEAST one factor of `1 / v_sigma` and `1 / h_sigma`, respectively
        # so we include those right away
        v_0 = v_0 / self.v_sigma
        v_k = v_k / self.v_sigma
        mean_h_0 = self.mean_hidden(v_0) / self.h_sigma
        mean_h_k = self.mean_hidden(v_k) / self.h_sigma

        # Recall: `v_sigma` and `h_sigma` has no affect if they are set to 1
        # v_0 / (v_sigma^2) - v_k / (v_sigma^2)
        delta_v_bias = (v_0 - v_k) / self.v_sigma
        # E[h_0 | v_0] / (h_sigma^2) - E[h_k | v_k] / (h_sigma^2)
        delta_h_bias = (mean_h_0 - mean_h_k) / self.h_sigma

        # Gradient wrt. W
        # (v_0 / v_sigma) (1 / h_sigma) E[h_0 | v_0] - (v_k / v_sigma) (1 / h_sigma) E[h_k | v_k]
        x = mean_h_0.reshape(mean_h_0.shape[0], 1, mean_h_0.shape[1])
        y = v_0.reshape(v_0.shape[0], v_0.shape[1], 1)
        z_0 = np.matmul(y, x)

        x = mean_h_k.reshape(mean_h_k.shape[0], 1, mean_h_k.shape[1])
        y = v_k.reshape(v_k.shape[0], v_k.shape[1], 1)
        z_k = np.matmul(y, x)

        delta_W = z_0 - z_k

        # average over batch take the negative
        delta_v_bias = - np.mean(delta_v_bias, axis=0)
        delta_h_bias = - np.mean(delta_h_bias, axis=0)
        delta_W = - np.mean(delta_W, axis=0)

        grads = [delta_v_bias, delta_h_bias, delta_W]

        # variances
        if self.visible_type == UnitType.GAUSSIAN and self.estimate_visible_sigma:
            # in this case `GaussianRBM`, where only the VISIBLE units are Gaussian,
            # we only compute `v_sigma`
            # (((v_0 - b)^2 / (v_sigma^2)) - (v / (v_sigma)) \sum_{\mu} E[h_{\mu} | v] / sigma_{\mu}) / v_sigma
            delta_v_sigma_data = (((v_0 - (self.v_bias / self.v_sigma)) ** 2)
                        - v_0 * (np.matmul(mean_h_0, self.W.T)))
            delta_v_sigma_model = (((v_k - (self.v_bias / self.v_sigma)) ** 2)
                         - v_k * (np.matmul(mean_h_k, self.W.T)))
            delta_v_sigma = (delta_v_sigma_data - delta_v_sigma_model) / self.v_sigma
            # average over batch take the negative
            delta_v_sigma = - np.mean(delta_v_sigma)

            grads.append(delta_v_sigma)

        if self.visible_type == UnitType.GAUSSIAN and self.estimate_hidden_sigma:
            # TODO: Implement
            raise NotImplementedError("gradients for gaussian hidden units not yet implemented")

            delta_h_sigma_data = (((h_0 - (self.h_bias / self.h_sigma)) ** 2)
                        - h_0 * (np.matmul(mean_h_0, self.W.T)))
            delta_h_sigma_model = (((h_k - (self.h_bias / self.h_sigma)) ** 2)
                         - h_k * (np.matmul(mean_h_k, self.W.T)))
            delta_h_sigma = delta_h_sigma_data - delta_h_sigma_model
            # average over batch take the negative
            delta_h_sigma = - np.mean(delta_h_sigma)

            grads.append(delta_h_sigma)

        return grads

    def fit(self, train_data, k=1, learning_rate=0.01,
            num_epochs=5, batch_size=64,
            test_data=None):
        num_samples = train_data.shape[0]
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        loglikelihood_train = []
        loglikelihood = []

        for epoch in range(1, num_epochs + 1):
            # compute train & test negative log-likelihood
            # TODO: compute train- and test-nll in mini-batches to avoid numerical problems
            nll_train = float(np.mean(self.free_energy(train_data)))
            loglikelihood_train.append(nll_train)
            _log.info(f"[{epoch:02d} / {num_epochs}] NLL (train): {nll_train:>20.5f}")

            if test_data is not None:
                nll = float(np.mean(self.free_energy(test_data)))
                _log.info(f"[{epoch:02d} / {num_epochs}] NLL (test):  {nll:>20.5f}")
                loglikelihood.append(nll)

            # iterate through dataset in batches
            bar = tqdm(total=num_samples)
            for start in range(0, num_samples, batch_size):
                # ensure we don't go out-of-bounds
                end = min(start + batch_size, num_samples)

                # take a gradient-step
                self.step(train_data[start: end], k=k, lr=learning_rate)

                # update progress
                bar.update(end - start)

            bar.close()

            # shuffle indices for next epoch
            np.random.shuffle(indices)

        # compute train & test negative log-likelihood of final batch
        nll_train = float(np.mean(self.free_energy(train_data)))
        loglikelihood_train.append(nll_train)
        _log.info(f"[{epoch:02d} / {num_epochs}] NLL (train): {nll_train:>20.5f}")

        if test_data is not None:
            nll = float(np.mean(self.free_energy(test_data)))
            _log.info(f"[{epoch:02d} / {num_epochs}] NLL (test):  {nll:>20.5f}")
            loglikelihood.append(nll)

        return loglikelihood_train, loglikelihood


class GaussianRBM(SimpleRBM):
    """Restricted Boltzmann Machine with Gaussian visible units and Bernoulli hidden units.

    """
    def __init__(self, num_visible, num_hidden, estimate_visible_sigma=False):
        super(GaussianRBM, self).__init__(
            num_visible, num_hidden,
            visible_type='gaussian',
            hidden_type='bernoulli',
            estimate_visible_sigma=estimate_visible_sigma,
            estimate_hidden_sigma=False
        )
