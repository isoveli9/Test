import matplotlib.pyplot as plt
import numpy as np

P = np.array([[0, 0], [2, 0.5], [4, 1], [3, -1], [5, -1.5], [7, -2]])
D = np.linalg.norm(P[:, None] - P, axis=2)
A = np.zeros((6, 6), bool)
A[np.arange(6)[:, None], np.argsort(D)[:, 1:3]] = True
W = np.where(A | A.T, 1 - D / np.sqrt(53), 0)
L = np.diag(W.sum(1)) - W
lam, V = np.linalg.eigh(L)
print(L.round(3), lam.round(3), V.round(3), sep="\n")

pca = np.array([-3.468, -1.746, -0.025, -0.306, 1.746, 3.799])
fig, ax = plt.subplot_mosaic([["a", "b"], ["a", "c"]], figsize=(7.5, 2.8),
                             width_ratios=[1.2, 1], layout="constrained")
for i, j in zip(*np.nonzero(np.triu(W))):
    ax["a"].plot(*P[[i, j]].T, color="0.8", zorder=0)
for v, s, name in (((0.944, -0.331), 2, "$v_1$"),
                   ((0.331, 0.944), 1, "$v_2$")):
    ax["a"].annotate(name, P.mean(0), P.mean(0) + s * np.array(v),
                     arrowprops={"arrowstyle": "<-"})
panels = (("a", P, "(a) data"),
          ("b", np.c_[V[:, 1], np.zeros(6)], "(b) spectral embedding"),
          ("c", np.c_[pca, np.zeros(6)], "(c) PCA"))
for k, X, title in panels:
    for f, marker in ((0, "o"), (3, "s")):
        ax[k].scatter(*X[f:f + 3].T, marker=marker)
    for i, p in enumerate(X, 1):
        ax[k].annotate(i, p, xytext=(-3, 5 + 7 * (i % 3) * (k != "a")),
                       textcoords="offset points")
    ax[k].set_title(title)
ax["a"].set(aspect="equal", xlabel="x", ylabel="y")
ax["b"].set_yticks([])
ax["c"].set_yticks([])
fig.savefig("figures/embedding.pdf")

x = np.array([1, 2, 3, 4, 8, 10, 11, 100])
for rep, inits in ((np.mean, [(8, 11), (1, 8)]),
                   (np.median, [(8, 11), (1, 8), (1, 100)])):
    for c in inits:
        print(rep.__name__, c)
        lab = None
        while True:
            new = np.argmin(abs(x[:, None] - np.array(c)), axis=1)
            c = [rep(x[new == j]) for j in (0, 1)]
            print(" ", new + 1, np.round(c, 2))
            if lab is not None and (new == lab).all():
                break
            lab = new

for C in ([[1, 2, 3, 4], [8, 10, 11, 100]],
          [[1, 2, 3, 4, 8, 10, 11], [100]]):
    S = []
    for k, Ck in enumerate(C):
        for p in Ck:
            if len(Ck) == 1:
                S.append(0)
                continue
            a = np.mean([abs(p - q) for q in Ck if q != p])
            b = min(np.mean([abs(p - q) for q in Cj])
                    for Cj in C if Cj is not Ck)
            S.append((b - a) / max(a, b))
            print(p, round(a, 2), round(b, 2), round(S[-1], 3))
    print("SC", round(np.mean(S), 3))
