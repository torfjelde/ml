"""
Implementation of the core of the MCMC sampler, which will make use of different kernels to
implement the different methods

A `Kernel` defines the transition probability.

Some kernels work directly with conditional probabilities :math:`p(x | y)` and :math:`p(y | x)`, 
e.g. Gibbs sampling, while other methods use the probability distribution we want to sample from 
:math:`p(x)` to construct the a transition probability / kernel for the sampler, e.g. 
Metropolis-Hastings.
"""
import abc
from tqdm import tqdm

from .. import np

import logging
_log = logging.getLogger("ml")


# TODO: Implement the samplers themselves, which performs the actual ``runs``.
class Sampler:
    """Generic sampler which takes samples using the provided `:class:Kernel`

    """
    def __init__(self, kernel, show_progress=False):
        super(Sampler, self).__init__()
        self.kernel = kernel
        self.show_progress = show_progress

    def run(self, n=1000, burnin=1000, every=10, initial=None):
        """
        Produces ``n`` samples.

        Parameters
        ----------
        n: int
            Number of samples to draw.
        burnin: int
            Number of steps to take before sampling.
        every: int
            Number of steps to take between each sample. Useful for managing autocorrelation
            between successive samples.

        Returns
        -------
        chain: array-like
            The resulting chain of length ``n``.

        """
        if initial is None:
            raise NotImplementedError("initialization of chain not yet implemented")

        state = initial
        for _ in range(burnin):
            self.kernel.sample(state)

        chain = []

        n_range = range(n)
        if self.show_progress:
            n_range = tqdm(n_range)

        for i in n_range:
            for _ in range(every):
                state = self.kernel.sample(state)

            chain.append(state)

        return chain


class Kernel(metaclass=abc.ABCMeta):
    """
    Generic representation of a kernel, providing the minimal set of methods and properties which
    a MCMC ``Kernel`` needs to implement.

    """
    def __init__(self):
        super(Kernel, self).__init__()
        
    def sample(self, *args, **kwargs):
        """
        Produces a sample according to the kernel.
        """
        raise NotImplementedError()


class MetropolisHastingsKernel(Kernel):
    r"""
    Implementation of the Metropolis-Hastings MCMC algorithm.

    Requires specification of the following methods:
    
    - sample: to produce samples from the proposal distribution, i.e. :math:`x' \sim T(x, x')`
    - sample probability: computes probability of sample given prev sample, i.e. :math:`T(x, x')`
    - probability: computes the probability of a sample, i.e. :math:`p(x)`

    Metropolis-Hastings uses the following expression to decide whether or not to accept a new state:
    
    .. math::
        :nowrap:
        
        \begin{equation*}
        A(x' \mid x) = \min \Bigg( 1, \frac{P(x')}{P(x)} \frac{g(x \mid x')}{g(x' \mid x)} \Bigg)
        \end{equation*}


    """
    def __init__(self, p, sampler, sampler_p):
        super(MetropolisHastingsKernel, self).__init__()
        self.p = p
        self.sampler = sampler
        self.sampler_p = sampler_p
        
    def sample(self, x):
        """
        Produces sample according to target distribution.

        Parameters
        ----------
        x: array-like
            The state to condition on for producing the next sample.
        """
        y = self.sampler(x)
        p_y = self.p(y)
        p_sample_y = self.sampler_p(x, y)

        p_x = self.p(x)
        p_sample_x = self.sampler_p(y, x)

        a = (p_y  * p_sample_x) / (p_x * p_sample_y)
        u = np.random.uniform(0, 1, size=a.shape)

        if u < a:
            return y
        else:
            return x


class GibbsKernel(Kernel):
    r"""
    Implementation of a Gibbs kernel.

    A Gibbs kernel simply produces samples from the conditional probability :math:`p(x' \mid x)`
    and accepts with probability 1, i.e. all samples.

    """
    def __init__(self, sampler):
        super(GibbsKernel, self).__init__()
        self.sampler = sampler

    def sample(self, x, **sampler_kwargs):
        """
        Samples according to the provided conditional distribution.

        Parameters
        ----------
        x: array-like
            The state to condition on for producing the next sample.

        """
        return self.sampler(x, **sampler_kwargs)


class BlockGibbsKernel(Kernel):
    r"""
    Implements a Block Gibbs sampling kernel.

    Should be used when there are groups of random variables which do not have any inter-dependence.
    Suppose we have a set of random variables :math:`` which we can separate into a set of groups
    :math:`` such that

    .. math::
        :nowrap:

        test

    We can then sample the entire group of variables :math:`` in parallel conditioned on the other
    groups, due to the conditional independence between all variables in this group.

    Assumes :math:`p(\mathcal{X}_i \mid \mathcal{X}_1, \dots, \mathcal{X}_{i - 1}, \mathcal{X}_{i + 1}, \dots, \mathcal{X}_G)` is
    represented by ``samplers[i]``.

    Parameters
    ----------
    samplers: list of methods
        List of callables from which to obtain samples.
    groups: list[int]
        List of integers corresponding to the index used to access the value of group ``i``.
    names: list[str]
        List of names of same length as ``groups``. 
    order: bool, default=False
        If ``True``, will sample from groups in random order.
        Otherwise, will sample sequentially.

    """
    def __init__(self, samplers, groups, names=None, order=None):
        super(BlockGibbsKernel, self).__init__()
        self.samplers = samplers
        self.groups = groups

        self.names = names or np.arange(len(groups))

    def sample(self, x, **sampler_kwargs):
        r"""
        Samples all groups sequentially or randomly and returns new state.

        Parameters
        ----------
        x: array-like
            Current state of all groups, assuming current state of group ``j`` to be accessed by
            ``x[j]``.

        """
        # perform copy initially, so we don't have to do this for every group-sample
        y = x.copy()

        for i in self.groups:
            y = self.sample_group(y, i, copy=False)

        return y
        
    def sample_group(self, x, i, copy=True, **sampler_kwargs):
        r"""
        Creates samples for group ``i`` from the conditional distribution
        :math:`condtional`, conditioning on all other groups of variables.

        Parameters
        ----------
        x: array-like
            Current state of all groups, assuming current state of group ``j`` to be accessed by
            ``x[j]``.
        i: int
            Index of group to obtain sample for.
        copy: bool
            If ``True``, will call ``x.copy()`` before replacing i-th entry with new sample.

        """
        sample = self.samplers[i]

        if i == 0:
            y_i = sample(x[1:], **sampler_kwargs)
        elif i == len(self.samplers) - 1:
            y_i = sample(x[:-1], **sampler_kwargs)
        else:
            y_i = sample(np.hstack([x[:i], x[i + 1:]]), **sampler_kwargs)

        if copy:
            y = x.copy()
        else:
            y = x

        y[i] = y_i
        return y
