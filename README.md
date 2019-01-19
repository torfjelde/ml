# ml
This package came about as a result of my summer 2018 research project. More regarding that can be found at https://github.com/torfjelde/summer-project-2018.

This package provides implementation of Bernoulli- and Gaussian RBMs using [cupy](https://github.com/cupy/cupy).

The idea was for it to be a place where I could accumulate my implementations of different learning algorithms and analysis tools, mainly for personal use. As of now, it only contains such for Restricted Boltzmann Machines (RBMs).

# Example
```python
from ml import np  # proxy for Cupy or Numpy

# uncomment the following line if one GPU; assigns `np` to `cupy` (quite a hackish method, I know)
# initialize_gpu()

from ml.rbms import RBM

# create sample data using Gaussian Mixture model
visible_size = 6
hidden_size = 6

means = np.arange(visible_size) + np.random.random(size=visible_size) * 3.0

cov = np.zeros((visible_size, visible_size))
for i in range(visible_size):
    cov[i, i] = 1.0
    cov[max(i - 1, 0), i] = 1.0
    cov[min(i + 1, visible_size - 1), i] = 1.0

cov = np.matmul(cov, cov)

data = np.random.multivariate_normal(means, np.matmul(cov, cov) , size=100)

# train / test split
train_data, test_data = data[:80], data[80:]

# training parameters
LR = 0.001
BATCH_SIZE = 128
NUM_EPOCHS = 10
K = 1
V_SIGMA = 0.1

# instatiate model
rbm = RBM(visible_size, hidden_size,
          visible_type='gaussian', hidden_type='bernoulli',
          sampler_method='cd',
          estimate_visible_sigma=True)

# fit
stats = rbm.fit(
    train_data,
    k=K,
    batch_size=BATCH_SIZE,
    num_epochs=NUM_EPOCHS,
    learning_rate=LR,
    test_data=test_data
)

# visualize
import matplotlib.pyplot as plt

fig = figure()
plt.plot(stats['nll_train'])
plt.plot(stats['nll_test'])
plt.show()

# sample
samples_v, samples_h = gibbs_sample_rbm(rbm, 100000, data[123], burnin=10000)

# visualize RBM samples vs. true samples
def plot_gaussian_mixtures(data, samples_v, title=None, include_means=False):
    fig, axes = plt.subplots(data.shape[1], 1, figsize=(15, 16), sharex=True, sharey=True)
    
    if title is not None:
        fig.suptitle(title, fontsize='x-large')

    for j in range(data.shape[1]):
        axes[j].hist(samples_v[:, j], alpha=0.5, bins=100, density=True, label=f"{j} fake")
        axes[j].hist(data[:, j], alpha=0.5, bins=100, density=True, label=f"{j} real")
        if include_means:
            axes[j].vlines(np.mean(samples_v[:, j]), ymin=0, ymax=0.5)
            axes[j].vlines(np.mean(data[:, j]), ymin=0, ymax=0.1, color='g', linewidth=5, alpha=0.7)
        axes[j].legend()
        axes[j].set_xlim(-5, 15)
        
    return fig, axes

fig = figure()
plot_gaussian_mixtures(data, samples_v, title=f"CD-k with $k={K}$")
plt.show()
```
