# -*- coding: utf-8 -*-

"""Console script for ml."""
import sys
import click
import logging

from . import np


@click.group()
def rbm():
    """Stuff to do with rbms."""
    pass


@click.command("train")
@click.option("-k", default=1,
              help="Number of steps to use in Contrastive Divergence (CD-k)",
              show_default=True)
@click.option("--batch-size", type=int, default=64, show_default=True)
@click.option("--hidden-size", type=int, default=500, show_default=True,
              help="Number of hidden units")
@click.option("--epochs", type=int, default=10, show_default=True,
              help="Number of epochs; one epoch runs through entire training data")
@click.option("--lr", type=float, default=0.01, show_default=True,
              help="Learning rate used multiplied by the gradients")
@click.option("--dataset", type=click.Choice(["mnist"]),
              help="Dataset to train on.", default="mnist",
              show_default=True)
@click.option("--dataset-path", type=click.Path(),
              help="Path to dataset to train on. Will download if does not exist.",
              default="mnist-original.mat", show_default=True)
@click.option("--rbm-type", default='bernoulli',
              type=click.Choice(["bernoulli", "gaussian"]),
              show_default=True,
              help="Type of RBM to use.")
@click.option("--sampler", default='cd',
              type=click.Choice(["CD", "PCD", "PT"]),
              show_default=True)
@click.option("--gpu", is_flag=True, show_default=True,
              help="Whether or not to use the GPU. Requires CUDA and cupy installed.")
@click.option("--output", type=str, default="sample.png", show_default=True,
              help="Output file for reconstructed images from test data")
@click.option("--show", is_flag=True, show_default=True,
              help="Whether or not to display image; useful when running on remote computer")
@click.option("--noise", type=float, default=0.1, show_default=True,
              help="Noise to use when attempting reconstructions.")
@click.option("-L", "--loglevel", type=str, default="INFO", show_default=True,
              help="Set the logging level, e.g. INFO, DEBUG, WARNING.")
def train_rbm(dataset, dataset_path,
              k, batch_size, hidden_size, epochs, lr,
              rbm_type, sampler, gpu,
              output, show, noise, loglevel):
    """Train an RBM on some dataset."""
    log = logging.getLogger("ml")
    log.setLevel(getattr(logging, loglevel))

    if gpu:
        from . import initialize_gpu
        initialize_gpu()

    from .rbms import RBM
    from .rbms.gaussian import GaussianRBM
    from . import datasets

    # load data
    data = getattr(datasets, dataset)
    X_train, X_test, y_train, y_test = data.load(dataset_path)
    input_size = X_train.shape[1]

    # MNIST takes on values in [0, 255], so we clip to [0, 1]
    if rbm_type == "bernoulli":
        X_train = X_train.clip(0, 1)
        X_test = X_test.clip(0, 1)
    # eps = np.finfo(np.float32).eps
    # X_train = X_train / np.linalg.norm(X_train + eps, axis=0)
    # X_test = X_test / np.linalg.norm(X_test + eps, axis=0)
    # X_train = (X_train - np.mean(X_train, axis=0)) / (np.var(X_train, axis=0) + eps)
    # X_test = (X_test - np.mean(X_test, axis=0)) / (np.var(X_test, axis=0) + eps)

    # use non-batch implementation if using CPU as it seems to be faster
    if rbm_type == "gaussian":
        model = GaussianRBM(input_size, hidden_size, sampler_method=sampler)
    else:
        model = RBM(input_size, hidden_size,
                    visible_type='bernoulli',
                    hidden_type='bernoulli',
                    sampler_method=sampler)

    log.info("Training...")
    nll_train, nll_test = model.fit(
        X_train,
        k=k, learning_rate=lr, num_epochs=epochs,
        batch_size=batch_size, test_data=X_test
    )

    if not show:
        import matplotlib
        matplotlib.use('Agg')

    import matplotlib.pyplot as plt

    from .viz import plot_reconstructions
    fig, axes = plot_reconstructions(model, X_test, noise=noise)

    fig.suptitle(f"RBM with {hidden_size} hidden units training for {epochs}")

    log.info(f"Saving to {output}")
    plt.savefig(output)

    if show:
        plt.show()


# add commands to group
rbm.add_command(train_rbm)
