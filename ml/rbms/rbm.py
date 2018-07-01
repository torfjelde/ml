from tqdm import tqdm

from .. import np, log
from ..functions import sigmoid

import abc


class GenericRBM(metaclass=abc.ABCMeta):
    """Class for generic implementations Restricted Boltzmann Machines (RBMs).

    """
    def __init__(self):
        super(GenericRBM, self).__init__()

    def sample_visible(self, h):
        "Samples visible units from the given hidden units `h`."
        # compute p(V_j = 1 | h)
        probas = self.proba_visible(h)
        # equiv. of V_j ~ p(V_j | h)
        rands = np.random.random(size=probas.shape)
        v = (probas > rands).astype(int)
        return v

    def sample_hidden(self, v):
        "Samples hidden units from the given visible units `v`."
        # compute p(H_{\mu} = 1 | v)
        probas = self.proba_hidden(v)
        # euqiv. of H_{\mu} ~ p(H_{\mu} | h)
        rands = np.random.random(size=probas.shape)
        h = (probas > rands).astype(np.int)
        return h

    def contrastive_divergence(self, v_0, k=1):
        "Perform CD-k, returning the intial and k-th sample `(v_0, v_k)`."
        v = v_0
        for t in range(k):
            h = self.sample_hidden(v)
            v = self.sample_visible(h)

        return v_0, v

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
            nll_train = float(np.mean(self.free_energy(train_data)))
            loglikelihood_train.append(nll_train)
            log.info(f"[{epoch} / {num_epochs}] NLL (train): {nll_train:>20.5f}")

            if test_data is not None:
                nll = float(np.mean(self.free_energy(test_data)))
                log.info(f"[{epoch} / {num_epochs}] NLL (test):  {nll:>20.5f}")
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
        log.info(f"[{epoch} / {num_epochs}] NLL (train): {nll_train:>20.5f}")

        if test_data is not None:
            nll = float(np.mean(self.free_energy(test_data)))
            log.info(f"[{epoch} / {num_epochs}] NLL (test):  {nll:>20.5f}")
            loglikelihood.append(nll)

        return loglikelihood_train, loglikelihood

    def reconstruct(self, v, num_samples=100):
        samples = self.sample_visible(self.sample_hidden(v))
        for _ in range(num_samples - 1):
            samples += self.sample_visible(self.sample_hidden(v))

        probs = samples / num_samples

        return probs

    @property
    def variables(self):
        return self._variables

    @abc.abstractmethod
    def proba_visible(self, h):
        return NotImplemented

    @abc.abstractmethod
    def proba_hidden(self, v):
        return NotImplemented

    @abc.abstractmethod
    def free_energy(self, v):
        return NotImplemented

    @abc.abstractmethod
    def grad(self, *args):
        return NotImplemented


class BatchBernoulliRBM(GenericRBM):
    """
    RBM with Bernoulli variables for hidden and visible states.
    """
    def __init__(self, num_visible, num_hidden):
        super(BatchBernoulliRBM, self).__init__()
        self.num_visible = num_visible
        self.num_hidden = num_hidden

        self.v_bias, self.h_bias, self.W = self.initialize(self.num_visible,
                                                           self.num_hidden)

        self._variables = [self.v_bias, self.h_bias, self.W]

    @staticmethod
    def initialize(num_visible, num_hidden):
        # biases for visible and hidden, respectively
        v_bias = np.zeros(num_visible)
        h_bias = np.zeros(num_hidden)

        # weight matrix
        W = np.random.normal(0.0, 0.01, (num_visible, num_hidden))

        return v_bias, h_bias, W

    def energy(self, v, h):
        return - (np.matmul(v, self.v_bias) +
                  np.matmul(h, self.h_bias) +
                  np.matmul(h, np.matmul(v, self.W).T))

    def proba_visible(self, h):
        "Computes p(v | h)."
        return sigmoid(self.v_bias + np.matmul(h, self.W.T))

    def proba_hidden(self, v):
        "Computes p(h | h)."
        return sigmoid(self.h_bias + np.matmul(v, self.W))

    def free_energy(self, v):
        # unnormalized
        # F(v) = - log \tilde{p}(v) = - \log \sum_{h} \exp ( - E(v, h))
        # using Eq. 2.20 (Fischer, 2015) for \tilde{p}(v)
        if len(v.shape) < 2:
            v = v.reshape(1, -1)
        visible = np.matmul(v, self.v_bias)
        hidden = self.h_bias + np.matmul(v, self.W)
        return - (visible + np.sum(np.log(1 + np.exp(hidden)), axis=1))

    def grad(self, v, k=1):
        "Estimates the gradient of the negative log-likelihood using CD-k."
        v_0, v_k = self.contrastive_divergence(v, k=k)
        proba_h_0 = self.proba_hidden(v_0)
        proba_h_k = self.proba_hidden(v_k)

        delta_v_bias = v_0 - v_k
        delta_h_bias = proba_h_0 - proba_h_k

        x = proba_h_0.reshape(proba_h_0.shape[0], 1, proba_h_0.shape[1])
        y = v_0.reshape(v_0.shape[0], v_0.shape[1], 1)
        z_0 = np.matmul(y, x)

        x = proba_h_k.reshape(proba_h_k.shape[0], 1, proba_h_k.shape[1])
        y = v_k.reshape(v_k.shape[0], v_k.shape[1], 1)
        z_k = np.matmul(y, x)
        delta_W = z_0 - z_k

        delta_v_bias = - np.mean(delta_v_bias, axis=0)
        delta_h_bias = - np.mean(delta_h_bias, axis=0)
        delta_W = - np.mean(delta_W, axis=0)

        # make negatives since we're performing gradient DEscent
        return delta_v_bias, delta_h_bias, delta_W


class BernoulliRBM(GenericRBM):
    """
    RBM with Bernoulli variables for hidden and visible states.
    """
    def __init__(self, num_visible, num_hidden):
        super(BernoulliRBM, self).__init__()
        self.num_visible = num_visible
        self.num_hidden = num_hidden

        self.v_bias, self.h_bias, self.W = self.initialize(self.num_visible,
                                                           self.num_hidden)

        self._variables = [self.v_bias, self.h_bias, self.W]

    @staticmethod
    def initialize(num_visible, num_hidden):
        # biases for visible and hidden, respectively
        v_bias = np.zeros(num_visible)
        h_bias = np.zeros(num_hidden)

        # weight matrix
        W = np.random.normal(0.0, 0.01, (num_visible, num_hidden))

        return v_bias, h_bias, W

    def energy(self, v, h):
        return - (np.dot(self.v_bias, v)
                  + np.dot(self.h_bias, h)
                  + np.dot(v, np.dot(self.W, h)))

    def proba_visible(self, h):
        "Computes p(v | h)."
        return sigmoid(self.v_bias + np.dot(self.W, h))

    def proba_hidden(self, v):
        "Computes p(h | h)."
        return sigmoid(self.h_bias + np.dot(self.W.T, v))

    def free_energy(self, v):
        # unnormalized
        # F(v) = - log \tilde{p}(v) = - \log \sum_{h} \exp ( - E(v, h))
        # using Eq. 2.20 (Fischer, 2015) for \tilde{p}(v)
        if len(np.shape(v)) >= 2:
            res = []
            for v_0 in v:
                visible = np.dot(self.v_bias, v_0)
                hidden = self.h_bias + np.dot(self.W.T, v_0)
                res.append(- (visible + np.sum(np.log(1 + np.exp(hidden)))))
            return res
        else:
            visible = np.dot(self.v_bias, v)
            hidden = self.h_bias + np.dot(self.W.T, v)
            return - (visible + np.sum(np.log(1 + np.exp(hidden))))

    def grad(self, vs, k=1):
        "Estimates the gradient of the negative log-likelihood using CD-k."
        avg_grads = None
        for v in vs:
            v_0, v_k = self.contrastive_divergence(v, k=k)
            proba_h_0 = self.proba_hidden(v_0)
            proba_h_k = self.proba_hidden(v_k)

            delta_v_bias = v_0 - v_k
            delta_h_bias = proba_h_0 - proba_h_k

            # reshape so that we can compute v_j h_{\mu} by
            # taking the dot product to obtain `delta_W`
            v_0 = np.reshape(v_0, (-1, 1))
            proba_h_0 = np.reshape(proba_h_0, (1, -1))

            v_k = np.reshape(v_k, (-1, 1))
            proba_h_k = np.reshape(proba_h_k, (1, -1))

            delta_W = np.dot(v_0, proba_h_0) - np.dot(v_k, proba_h_k)

            if avg_grads is None:
                avg_grads = [-delta_v_bias, -delta_h_bias, -delta_W]
            else:
                avg_grads[0] -= delta_v_bias
                avg_grads[1] -= delta_h_bias
                avg_grads[2] -= delta_W

        return [g / len(vs) for g in avg_grads]
