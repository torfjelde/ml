import matplotlib.pyplot as plt
from matplotlib import gridspec

from .. import np

from ..functions import flip

import logging
log = logging.getLogger("ml")


def plot_reconstructions(rbm, data, noise=0.0):
    # plot some reconstructions
    n_rows = 6
    n_cols = 8

    fig, axes = plt.subplots(n_rows, n_cols,
                             sharex=True, sharey=True,
                             figsize=(16, 12),
                             # make it tight
                             gridspec_kw=dict(wspace=-0.1, hspace=-0.01))

    for i in range(n_rows):
        for j in range(n_cols // 2):
            v_0 = data[np.random.randint(len(data))]

            # introduce noise
            v = flip(v_0, noise, max=np.max(v_0))
            probs = rbm.reconstruct(v, num_samples=1000)

            # in case we've substituted with `cupy`
            reconstruction_error = float(np.mean(np.abs(v_0 - probs)))
            log.info(f"Reconstruction error of {(i, 2 * j)}: " +
                     f"{reconstruction_error:0.10f} (noise: {noise})")

            if np.__name__ != "numpy":
                v = np.asnumpy(v)
                probs = np.asnumpy(probs)

            axes[i][2 * j].imshow(np.reshape(v, (28, 28)))
            axes[i][2 * j + 1].imshow(np.reshape(probs, (28, 28)))

            # customization; remove labels
            axes[i][2 * j].set_xticklabels([])
            axes[i][2 * j].set_yticklabels([])

            axes[i][2 * j + 1].set_xticklabels([])
            axes[i][2 * j + 1].set_yticklabels([])

    return fig, axes
