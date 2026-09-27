"""CS-E4650 homework: clustering tendency and entropy-based measures.

Writes figures/histograms.pdf and figures/scatter.pdf and prints the
entropies and the stages of the greedy searches.
"""
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
X = np.loadtxt(ROOT / "data/clustering-tendency.csv", delimiter=",")
assert X.shape == (1000, 3) and X.min() >= 0 and X.max() <= 1
DIMS = (0, 1, 2)
M = 64  # grid cells in every entropy calculation

# histograms, 64 bins
fig, axes = plt.subplots(3, 1, figsize=(7, 4.2), sharex=True, sharey=True)
for j, ax in enumerate(axes):
    ax.hist(X[:, j], bins=64, range=(0, 1))
    ax.text(0.99, 0.92, f"component {j}", transform=ax.transAxes, ha="right", va="top")
    ax.set(xlim=(0, 1), ylabel="count")
axes[-1].set_xlabel("value")
fig.tight_layout()
fig.savefig(ROOT / "figures/histograms.pdf", bbox_inches="tight")

# scatter plots of the pairs, equal aspect ratio
fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.7))
for ax, (i, j) in zip(axes, combinations(DIMS, 2)):
    ax.scatter(X[:, i], X[:, j], s=3, alpha=0.5, linewidths=0)
    ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal",
           xlabel=f"component {i}", ylabel=f"component {j}")
fig.tight_layout()
fig.savefig(ROOT / "figures/scatter.pdf", bbox_inches="tight")


def plogp(q):
    """Sum of q ln q with 0 ln 0 = 0."""
    q = q[q > 0]
    return np.sum(q * np.log(q))


def prob_entropy(p):
    """Aggarwal eq. (6.2): E = -sum_i [p_i ln p_i + (1 - p_i) ln(1 - p_i)]."""
    return float(-(plogp(p) + plogp(1 - p)))


def entropy(dims):
    """E of the dimensions dims on a uniform grid of M cells in [0,1]^k."""
    k = len(dims)
    phi = round(M ** (1 / k))  # ranges per dimension: 4, 8 or 64
    idx = np.minimum((X[:, dims] * phi).astype(int), phi - 1)  # x = 1 -> last range
    counts = np.bincount(np.ravel_multi_index(idx.T, (phi,) * k), minlength=M)
    return prob_entropy(counts / len(X))


subsets = [s for k in (3, 2, 1) for s in combinations(DIMS, k)]
E = {s: entropy(list(s)) for s in subsets}
print(f"E_max = {prob_entropy(np.full(M, 1 / M)):.3f}")
for s in subsets:
    print(s, f"{E[s]:.3f}")


def backward():
    """Drop the worst feature (lowest E after removal) while E decreases."""
    cur = DIMS
    while len(cur) > 1:
        cands = [tuple(d for d in cur if d != r) for r in cur]
        print(" ", cur, {c: round(E[c], 3) for c in cands})
        best = min(cands, key=E.get)
        if E[best] >= E[cur]:
            break
        cur = best
    return cur


def forward():
    """Add the best feature (lowest E after addition) while E decreases.
    The empty set has no entropy, so the best single feature starts."""
    cur = ()
    while len(cur) < len(DIMS):
        cands = [tuple(sorted(cur + (a,))) for a in DIMS if a not in cur]
        print(" ", cur, {c: round(E[c], 3) for c in cands})
        best = min(cands, key=E.get)
        if cur and E[best] >= E[cur]:
            break
        cur = best
    return cur


print("backward selection:")
print("  result", backward())
print("forward selection:")
print("  result", forward())
