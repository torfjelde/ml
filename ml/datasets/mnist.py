import os
from six.moves import urllib
from sklearn.datasets import fetch_mldata
from scipy.io import loadmat

from .. import np

import logging
log = logging.getLogger("ml")


def load(mnist_path="mnist-original.mat", random_seed=42):
    # Alternative method to load MNIST, since mldata.org is often down...
    mnist_alternative_url = "https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat"

    if os.path.exists(mnist_path):
        log.info(f"Found existing file at {mnist_path}; loading...")
        mnist_raw = loadmat(mnist_path)
        mnist = {
            "data": mnist_raw["data"].T,
            "target": mnist_raw["label"][0],
            "COL_NAMES": ["label", "data"],
            "DESCR": "mldata.org dataset: mnist-original",
        }
    else:
        log.info(f"Dataset not found at {mnist_path}; downloading...")
        response = urllib.request.urlopen(mnist_alternative_url)
        with open(mnist_path, "wb") as f:
            content = response.read()
            f.write(content)
        mnist_raw = loadmat(mnist_path)
        mnist = {
            "data": mnist_raw["data"].T,
            "target": mnist_raw["label"][0],
            "COL_NAMES": ["label", "data"],
            "DESCR": "mldata.org dataset: mnist-original",
        }
        log.info("Success!")

    # train-test split
    from sklearn.model_selection import train_test_split

    # in case we want to use `cupy` to run on the GPU
    X = np.asarray(mnist["data"])
    y = np.asarray(mnist["target"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_seed)
    return X_train, X_test, y_train, y_test
