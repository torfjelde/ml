from .core import RBM


class GaussianRBM(RBM):
    """Restricted Boltzmann Machine with Gaussian visible units
    and Bernoulli hidden units.

    """
    def __init__(self, num_visible, num_hidden, estimate_visible_sigma=False, **kwargs):
        super(GaussianRBM, self).__init__(
            num_visible, num_hidden,
            visible_type='gaussian',
            hidden_type='bernoulli',
            estimate_visible_sigma=estimate_visible_sigma,
            estimate_hidden_sigma=False,
            **kwargs
        )
