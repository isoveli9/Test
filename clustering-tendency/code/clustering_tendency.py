from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np

X = np.loadtxt("data/clustering-tendency.csv", delimiter=",")
assert X.shape == (1000, 3) and X.min() >= 0 and X.max() <= 1

fig, axes = plt.subplots(3, 1, figsize=(7, 4.2), sharex=True, sharey=True,
                         layout="constrained")
for j, ax in enumerate(axes):
    ax.hist(X[:, j], bins=64, range=(0, 1))
    ax.text(0.99, 0.9, f"component {j}", transform=ax.transAxes,
            ha="right", va="top")
    ax.set(xlim=(0, 1), ylabel="count")
axes[2].set_xlabel("value")
fig.savefig("figures/histograms.pdf")

fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.7), layout="constrained")
for ax, (i, j) in zip(axes, combinations(range(3), 2)):
    ax.scatter(X[:, i], X[:, j], s=3, alpha=0.5, linewidths=0)
    ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal",
           xlabel=f"component {i}", ylabel=f"component {j}")
fig.savefig("figures/scatter.pdf")


def entropy(dims, m=64):
    k = len(dims)
    counts, _ = np.histogramdd(X[:, dims], bins=round(m ** (1 / k)),
                               range=[(0, 1)] * k)
    p = counts.ravel() / len(X)
    q = np.concatenate([p, 1 - p])
    q = q[q > 0]
    return -np.sum(q * np.log(q))


for k in (3, 2, 1):
    for dims in combinations(range(3), k):
        print(dims, round(entropy(dims), 3))
