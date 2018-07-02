from . import np


def sigmoid(z):
    # clip the values due to possibility of overflow
    return 1.0 / (1.0 + np.exp(-np.maximum(np.minimum(z, 30), -30)))


def flip(x, p, max=1.0):
    "Assumes `x` is an array of 0s and 1s, and flips these based on probabilities `p`."
    x = x.copy()
    mask = np.random.random(size=x.shape) < p
    flipped = (~(x.astype(np.bool))).astype(np.int) * max
    x[mask] = flipped[mask]
    return x


def dot_batch(x, y):
    """Computes dot-product between batch of two vectors.
    `x` and `y` assumed to be of shape `(batch_size, vector_size)`.

    """
    return np.sum(x * y, axis=1)
