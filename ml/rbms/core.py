import abc
import logging

from enum import Enum
from tqdm import tqdm

from .. import np
from ..functions import sigmoid, dot_batch, bernoulli_from_probas

_log = logging.getLogger("ml")


class UnitType(Enum):
    GAUSSIAN = 1
    BERNOULLI = 2


class RBMSampler(object):
    """Sampler used in training of RBMs for estimating the gradient.

    """
    def __init__(self, args):
        super(RBMSampler, self).__init__()
        self.args = args


class RBM:
    """
    Restricted Boltzmann Machine with either Bernoulli or Gaussian
    visible/hidden units.

    Attributes
    ---------
    num_visible: int
        Number of visible units.
    num_hidden: int
        Number of hidden units.
    visible_type: UnitType or str, default='bernoulli'
        Type of random variable the visible units are assumed to be.
    hidden_type: UnitType or str, default='bernoulli'
        Type of random variable the hidden units are assumed to be.
    estimate_visible_sigma: bool, default=False
        Whether or not to estimate the variance of the visible units.
        If :attr:`visible_type` is non-Gaussian, then this has no effect.
    estimate_hidden_sigma: bool, default=False
        Whether or not to estimate the variance of the hidden units.
        If :attr:`hidden_type` is non-Gaussian, then this has no effect.
    sampler_method: str, default='cd'
        Specifies the method used in the sampling process when approximating
        the gradient.
        Available methods are:

        - Contrastive Divergence (CD)
        - Persistent Contrastive Divergence (PCD)
        - Parallel Tempering (PT)

        See :func:`RBM.grad` for more information about the
        effects of the different available methods.
    variables: list[array-like]
        Holds the learnable parameters of the machine. This is used by
        :func:`RBM.step` to deduce what parameters to update.

    See Also
    --------
    :func:`RBM.grad` for more information about samplers.

    """
    def __init__(self, num_visible, num_hidden,
                 visible_type='bernoulli', hidden_type='bernoulli',
                 estimate_visible_sigma=False, estimate_hidden_sigma=False,
                 sampler_method='cd'):
        super(RBM, self).__init__()
        self.num_visible = num_visible
        self.num_hidden = num_hidden

        if sampler_method.lower() not in {'cd', 'pcd', 'pt'}:
            raise ValueError(f"{sampler_method} is not supported")
        self.sampler_method = sampler_method.lower()
        # used by `PCD` sampler
        self._prev = None

        if isinstance(visible_type, str):
            self.visible_type = getattr(UnitType, visible_type.upper())
        else:
            self.visible_type = visible_type
        if isinstance(hidden_type, str):
            self.hidden_type = getattr(UnitType, hidden_type.upper())
        else:
            self.hidden_type = hidden_type

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
            visible = ((v - self.v_bias) ** 2) / (self.v_sigma ** 2
                                                  + np.finfo(np.float32).eps)
            visible = 0.5 * np.sum(visible, axis=1)

        # term only dependent on hidden
        if self.hidden_type == UnitType.BERNOULLI:
            hidden = np.matmul(h, self.h_bias)
        elif self.hidden_type == UnitType.GAUSSIAN:
            hidden = ((h - self.h_bias) ** 2) / (self.h_sigma ** 2
                                                 + np.finfo(np.float32).eps)
            hidden = 0.5 * np.sum(hidden, axis=1)

        # "covariance" term
        # v^T W = sum_j( (v_j / sigma_j) W_{j \mu} )
        covariance = np.matmul(v, self.W)
        # v^T W h = sum_{\mu} h_{\mu} sum_j( (v_j / sigma_j) W_{j \mu} )
        covariance = dot_batch(h, covariance)

        return - (visible + hidden + covariance)

    def mean_visible(self, h, beta=1.0):
        r"""
        Computes :math:`\mathbb{E}[\mathbf{v} \mid \mathbf{h}]`.

        It can be shown that this expectation equals: [1]_

        - Bernoulli:

          .. math::
            :nowrap:

            \begin{equation}
            \mathbb{E}[\mathbf{v} \mid \mathbf{h}] =
            p \big( V_{i} = 1 \mid \mathbf{h} \big) = \text{sigmoid}
            \Bigg( \beta \bigg( b_{i} + \sum_{\mu=1}^{|\mathcal{H}|} W_{i \mu} \frac{h_{\mu}}{\sigma_{\mu}} \bigg) \Bigg)
            \end{equation}

        - Gaussian:

          .. math::
            :nowrap:

            \begin{equation*}
            \mathbb{E}[\mathbf{v} \mid \mathbf{h}] = b_i + \sigma_i \sum_{\mu=1}^{|\mathcal{H}|} W_{i \mu} \frac{h_{\mu}}{\sigma_{\mu}}
            \end{equation*}

        where :math:`\sigma_{\mu} = 1` if :math:`H_\mu` is a Bernoulli random variable.
        
        Notes
        -----
        Observe that the expectation when using Gaussian units is
        independent of :math:`\beta`. To see the effect :math:`\beta` has
        on the Gaussian case, see :func:`RBM.proba_visible`.

        References
        ----------
        .. [1] Fjelde, T. E., Restricted Boltzmann Machines, , (),  (2018).
        """
        mean = self.v_bias + (self.v_sigma *
                              np.matmul(h / self.h_sigma, self.W.T))
        if self.visible_type == UnitType.BERNOULLI:
            return sigmoid(mean * beta)
        elif self.visible_type == UnitType.GAUSSIAN:
            return mean

    def mean_hidden(self, v, beta=1.0):
        "Computes conditional expectation E[h | v]."
        mean = self.h_bias + self.h_sigma * np.matmul(v / self.v_sigma, self.W)
        if self.hidden_type == UnitType.BERNOULLI:
            return sigmoid(mean * beta)
        elif self.hidden_type == UnitType.GAUSSIAN:
            return mean

    def sample_visible(self, h, beta=1.0):
        mean = self.mean_visible(h, beta=beta)
        if self.visible_type == UnitType.BERNOULLI:
            # E[v | h] = p(v | h) for Bernoulli
            v = bernoulli_from_probas(mean)
        elif self.visible_type == UnitType.GAUSSIAN:
            v = np.random.normal(loc=mean,
                                 scale=self.v_sigma ** 2 / beta,
                                 size=mean.shape)
        else:
            raise ValueError(f"unknown type {self.visible_type}")

        return v

    def sample_hidden(self, v, beta=1.0):
        mean = self.mean_hidden(v, beta=beta)
        if self.visible_type == UnitType.BERNOULLI:
            # E[v | h] = p(v | h) for Bernoulli
            h = bernoulli_from_probas(mean)
        elif self.visible_type == UnitType.GAUSSIAN:
            h = np.random.normal(loc=mean,
                                 scale=(self.h_sigma ** 2 / beta),
                                 size=(mean.shape))
        else:
            raise ValueError(f"unknown type {self.visible_type}")

        return h

    def proba_visible(self, h, v=None, beta=1.0):
        mean = self.mean_visible(h, beta=beta)
        if self.visible_type == UnitType.BERNOULLI:
            # E[v | h] = p(v | h) for Bernoulli
            p = mean
        elif self.visible_type == UnitType.GAUSSIAN:
            z = np.clip((v - mean) ** 2 / (2.0 * self.v_sigma ** 2),
                        -30.0, 30.0)
            z *= beta
            p = (np.exp(z) / (np.sqrt(2 * np.pi) * self.v_sigma
                              + np.finfo(np.float32).eps))
        else:
            raise ValueError(f"unknown type {self.visible_type}")

        return p

    def proba_hidden(self, v, h=None, beta=1.0):
        mean = self.mean_hidden(v, beta=beta)
        if self.hidden_type == UnitType.BERNOULLI:
            # E[v | h] = p(v | h) for Bernoulli
            p = mean
        elif self.hidden_type == UnitType.GAUSSIAN:
            z = np.clip((h - mean) ** 2 / (2.0 * self.h_sigma ** 2),
                        -30.0, 30.0)
            z *= beta
            p = (np.exp(z) / (np.sqrt(2 * np.pi) * self.h_sigma
                              + np.finfo(np.float32).eps))
        else:
            raise ValueError(f"unknown type {self.hidden_type}")

        return p

    def free_energy(self, v, beta=1.0):
        if self.hidden_type == UnitType.BERNOULLI:
            hidden = self.h_bias + np.matmul((v / self.v_sigma), self.W)
            hidden *= beta
            hidden = - np.sum(np.log(1.0 + np.exp(np.clip(hidden, -30, 30))),
                              axis=1)
        elif self.hidden_type == UnitType.GAUSSIAN:
            # TODO: Implement
            # Have the formulas, but gotta make sure yo!
            hidden = np.sum(
                1 / (2 * self.h_sigma) * (
                    self.h_bias ** 2
                    - (self.h_bias + self.h_sigma * np.matmul(v / self.v_sigma, self.W)) ** 2
                ),
                axis=1
            )
            hidden -= 0.5 * self.num_hidden * np.log(2 * np.pi) + np.sum(np.log(self.h_sigma))
            # raise NotImplementedError()

        if self.visible_type == UnitType.BERNOULLI:
            visible = - np.matmul(v, self.v_bias)
            visible *= beta
        elif self.visible_type == UnitType.GAUSSIAN:
            visible = 0.5 * np.sum(
                ((v - self.v_bias) ** 2)
                / (self.v_sigma ** 2 / beta + np.finfo(np.float32).eps),
                axis=1
            )
        else:
            raise ValueError(f"unknown type {self.visible_type}")

        # sum across batch to obtain log of joint-likelihood
        return np.sum(hidden + visible)

    def contrastive_divergence(self, v_0, k=1,
                               persistent=False,
                               burnin=1000,
                               beta=1.0):
        """Contrastive Divergence.

        Parameters
        ----------
        v_0: array-like
            Visible state to initialize the chain from.
        k: int
            Number of steps to use in CD-k.

        Returns
        -------
        h_0, h, v_0, v: arrays
            ``h_0`` and ``v_0`` are the initial states for the hidden and
            visible units, respectively.
            ``h`` and ``v`` are the final states for the hidden and
            visible units, respectively.
        """
        if persistent and self._prev is not None:
            h_0, v_0 = self._prev
        else:
            h_0 = self.sample_hidden(v_0, beta=beta)

        v = v_0
        h = h_0

        if persistent and self._prev is None and burnin > 0:
            _log.info(f"Running PCD using burnin of {burnin}")
            for i in range(burnin):
                v = self.sample_visible(h, beta=beta)
                h = self.sample_hidden(v, beta=beta)

        for t in range(k):
            v = self.sample_visible(h, beta=beta)
            h = self.sample_hidden(v, beta=beta)

        if persistent:
            self._prev = (h, v)

        return v_0, h_0, v, h

    def reset_sampler(self):
        if self.sampler_method == 'pcd':
            self._prev = None

    def parallel_tempering(self, v, k=1, max_temp=100, num_temps=10,
                           include_negative_shift=False):
        # TODO: Performing sampling in parallel, rather than using a loop
        # 1. Allow `self.contrastive_divergence` to take on arrays of betas
        # 2. Stack betas and initial samples
        # 3. Perform sampling
        # 4. Unstack
        batch_size = v.shape[0]

        # 1. Initialize list of samples
        betas = np.linspace(1, max_temp, num_temps) ** (-1)
        R = len(betas)
        res = []

        if include_negative_shift:
            neg_res = []

        # 2. Perform gibbs sampling for tempered distributions
        for r in range(R):
            v_0, h_0, v_k, h_k = self.contrastive_divergence(
                v,
                k=k,
                beta=betas[r]
            )
            res.append((v_k, h_k))

            if include_negative_shift:
                neg_res.append((v_0, h_0))

        # 3. Simulated Annealing to perform swaps ("exchange particles")
        for r in range(R - 1, 0, -1):
            a = np.exp((betas[r] - betas[r - 1]) *
                       (self.energy(*res[r]) - self.energy(*res[r - 1])))
            u = np.random.random(batch_size)
            # acceptance mask
            acc_mask = (u < a).reshape(batch_size, 1)
            # reject mask
            rej_mask = ~acc_mask

            v = res[r][0] * acc_mask + res[r - 1][0] * rej_mask
            h = res[r][1] * acc_mask + res[r - 1][1] * rej_mask
            res[r - 1] = v, h

            v = res[r - 1][0] * acc_mask + res[r][0] * rej_mask
            h = res[r - 1][1] * acc_mask + res[r][1] * rej_mask
            res[r] = v, h

            # if r == 1:
            #     _log.info(acc_mask[acc_mask].shape)

        # possibly perform same for the negative shift
        if include_negative_shift:
            for r in range(R - 1, 0, -1):
                a = np.exp((betas[r] - betas[r - 1]) *
                           (self.energy(*neg_res[r]) - self.energy(*neg_res[r - 1])))
                u = np.random.random(batch_size)
                # acceptance mask
                acc_mask = (u < a).reshape(batch_size, 1)
                # reject mask
                rej_mask = ~acc_mask

                v = neg_res[r][0] * acc_mask + neg_res[r - 1][0] * rej_mask
                h = neg_res[r][1] * acc_mask + neg_res[r - 1][1] * rej_mask
                neg_res[r - 1] = v, h

                v = neg_res[r - 1][0] * acc_mask + neg_res[r][0] * rej_mask
                h = neg_res[r - 1][1] * acc_mask + neg_res[r][1] * rej_mask
                neg_res[r] = v, h

        # return final state
        if include_negative_shift:
            return neg_res[0], res[0]
        else:
            return res[0]

    def _update(self, grad, lr=0.1):
        gamma = lr
        for i in range(len(self.variables)):
            if np.shape(lr):
                gamma = lr[i]
            self.variables[i] -= gamma * grad[i]

    def _apply_weight_decay(self, lmbda=0.01):
        for i in range(len(self.variables)):
            # default is gradient DEscent, so weight-decay also switches signs
            self.variables[i] += lmbda * self.variables[i]

    def step(self, v, k=1, lr=0.1, lmbda=0.0, **sampler_kwargs):
        "Performs a single gradient DEscent step on the batch `v`."
        # compute gradient for each observed visible configuration
        grad = self.grad(v, k=k, **sampler_kwargs)

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

    def grad(self, v, **sampler_kwargs):
        if self.sampler_method.lower() == 'cd':
            v_0, h_0, v_k, h_k = self.contrastive_divergence(
                v,
                **sampler_kwargs
            )
        elif self.sampler_method.lower() == 'pcd':
            # Persistent Contrastive Divergence
            v_0, h_0, v_k, h_k = self.contrastive_divergence(
                v,
                persistent=True,
                **sampler_kwargs
            )
        elif self.sampler_method.lower() == 'pt':
            if sampler_kwargs.get("include_negative_shift", False):
                v_0, h_0, v_k, h_k = self.parallel_tempering(
                    v,
                    **sampler_kwargs
                )
            else:
                v_0 = v
                v_k, h_k = self.parallel_tempering(
                    v,
                    **sampler_kwargs
                )
        else:
            raise ValueError(f"{self.sampler_method} is not supported")
        
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
        if self.visible_type == UnitType.GAUSSIAN \
           and self.estimate_visible_sigma:
            # in `GaussianRBM`, where only VISIBLE units Gaussian,
            # we only compute `v_sigma`
            # (((v_0 - b)^2 / (v_sigma^2)) - (v / (v_sigma)) \sum_{\mu} E[h_{\mu} | v] / sigma_{\mu}) / v_sigma
            delta_v_sigma_data = (((v_0 - (self.v_bias / self.v_sigma)) ** 2)
                        - v_0 * (np.matmul(mean_h_0, self.W.T)))
            delta_v_sigma_model = (((v_k - (self.v_bias / self.v_sigma)) ** 2)
                         - v_k * (np.matmul(mean_h_k, self.W.T)))
            delta_v_sigma = (delta_v_sigma_data - delta_v_sigma_model) / self.v_sigma
            # average over batch take the negative
            delta_v_sigma = - np.mean(delta_v_sigma, axis=0)

            grads.append(delta_v_sigma)

        if self.hidden_type == UnitType.GAUSSIAN \
           and self.estimate_hidden_sigma:
            # TODO: Implement
            raise NotImplementedError("gradients for gaussian hidden"
                                      " units not yet implemented")

            delta_h_sigma_data = (((h_0 - (self.h_bias / self.h_sigma)) ** 2)
                        - h_0 * (np.matmul(mean_h_0, self.W.T)))
            delta_h_sigma_model = (((h_k - (self.h_bias / self.h_sigma)) ** 2)
                         - h_k * (np.matmul(mean_h_k, self.W.T)))
            delta_h_sigma = delta_h_sigma_data - delta_h_sigma_model
            # average over batch take the negative
            delta_h_sigma = - np.mean(delta_h_sigma, axis=0)

            grads.append(delta_h_sigma)

        return grads

    def fit(self, train_data,
            k=1,
            learning_rate=0.01,
            num_epochs=5,
            batch_size=64,
            test_data=None,
            show_progress=True,
            weight_decay=0.0,
            early_stopping=-1,
            callbacks={},
            **sampler_kwargs):
        """
        Parameters
        ----------
        train_data: array-like
            Data to fit RBM on.
        k: int, default=1
            Number of sampling steps to perform. Used by CD-k, PCD-k and PT.
        learning_rate: float or array, default=0.01
            Learning rate used when updating the parameters.
            Can also be array of same length as `self.variables`, in
            which case the learning rate at index `i` will be used to
            to update ``RBM.variables[i]``.
        num_epochs: int, default=5
            Number of epochs to train.
        batch_size: int, default=64
            Batch size to within the epochs.
        test_data: array-like, default=None
            Data similar to ``train_data``, but this will only be used as
            validation data, not trained on.
            If specified, will compute and print the free energy / negative
            log-likelihood on this dataset after each epoch.
        show_progress: bool, default=True
            If true, will display progress bar for each epoch.
        weight_decay: float, default=0.0
            If greater than 0.0, weight decay will be applied to the
            parameter updates. See :func:`RBM.step` for more information.
        early_stopping: int, default=-1
            If ``test_data`` is given and ``early_stopping > 0``, training
            will terminate after epoch if the free energy of the
            ``test_data`` did not improve over the fast ``early_stopping``
            epochs.

        Returns
        -------
        nlls_train, nlls_test : array-like, array-like
            Returns the free energy of both ``train_data`` and ``test_data``
            as computed at each epoch.

        """
        num_samples = train_data.shape[0]
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        nlls_train = []
        nlls = []

        prev_best = None

        for epoch in range(1, num_epochs + 1):
            if "pre_epoch" in callbacks:
                for c in callbacks["pre_epoch"]:
                    c(self, epoch)

            # reset sampler at beginning of epoch
            # Used by methods such as PCD to reset the
            # initialization value.
            self.reset_sampler()

            # compute train & test negative log-likelihood
            # TODO: compute train- and test-nll in mini-batches
            # to avoid numerical problems
            nll_train = float(np.mean(self.free_energy(train_data)))
            nlls_train.append(nll_train)
            _log.info(f"[{epoch:03d} / {num_epochs:03d}] NLL (train):"
                      f" {nll_train:>20.5f}")

            if test_data is not None:
                nll = float(np.mean(self.free_energy(test_data)))
                _log.info(f"[{epoch:03d} / {num_epochs:03d}] NLL (test):"
                          f"  {nll:>20.5f}")
                nlls.append(nll)

                # stop early if all `early_stopping` previous
                # evaluations on `test_data` did not improve.
                if early_stopping > 0:
                    if epoch > early_stopping and \
                       np.all([a >= prev_best for a in nlls[epoch - early_stopping:]]):
                        _log.info("Hasn't improved in {early_stopping} epochs; stopping early")
                        break
                    else:
                        # update `prev_best`
                        if prev_best is None:
                            prev_best = nll
                        elif nll < prev_best:
                            prev_best = nll

            # iterate through dataset in batches
            if show_progress:
                bar = tqdm(total=num_samples)
            for start in range(0, num_samples, batch_size):
                # ensure we don't go out-of-bounds
                end = min(start + batch_size, num_samples)

                # take a gradient-step
                self.step(train_data[start: end],
                          k=k,
                          lr=learning_rate,
                          lmbda=weight_decay,
                          **sampler_kwargs)

                if "post_step" in callbacks:
                    for c in callbacks["post_step"]:
                        c(self, epoch, end)

                # update progress
                if show_progress:
                    bar.update(end - start)

            if show_progress:
                bar.close()

            # shuffle indices for next epoch
            np.random.shuffle(indices)

            if "post_epoch" in callbacks:
                for c in callbacks["post_epoch"]:
                    c(self, epoch)

        # compute train & test negative log-likelihood of final batch
        nll_train = float(np.mean(self.free_energy(train_data)))
        nlls_train.append(nll_train)
        _log.info(f"[{epoch:03d} / {num_epochs:03d}] NLL (train): "
                  f"{nll_train:>20.5f}")

        if test_data is not None:
            nll = float(np.mean(self.free_energy(test_data)))
            _log.info(f"[{epoch:03d} / {num_epochs:03d}] NLL (test):  "
                      f"{nll:>20.5f}")
            nlls.append(nll)

        return nlls_train, nlls

    def dump(self, path, *attrs):
        import pickle

        if not attrs:
            attrs = [
                'num_visible',
                'num_hidden',
                'visible_type',
                'hidden_type',
                'estimate_visible_sigma',
                'estimate_hidden_sigma',
                'variables',
                'v_bias',
                'h_bias',
                'W',
                'v_sigma',
                'h_sigma'
            ]

        state = {}

        for a in attrs:
            state[a] = getattr(self, a)

        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path):
        import pickle

        with open(path, "rb") as f:
            state = pickle.load(f)

        model = cls(num_visible=state['num_visible'],
                    num_hidden=state['num_hidden'],
                    visible_type=state['visible_type'],
                    hidden_type=state['hidden_type'],
                    estimate_visible_sigma=state['estimate_visible_sigma'],
                    estimate_hidden_sigma=state['estimate_hidden_sigma'])
        for a in state:
            setattr(model, a, state[a])

        return model


class BernoulliRBM(RBM):
    """Restricted Boltzmann Machine (RBM) with both hidden and visible
    variables assumed to be Bernoulli random variables.

    """
    def __init__(self, num_visible, num_hidden):
        super(BernoulliRBM, self).__init__(
            num_visible,
            num_hidden,
            visible_type='bernoulli',
            hidden_type='bernoulli'
        )
