import matplotlib.pyplot as plt
import numpy as np

P = np.array([[0, 0], [2, 0.5], [4, 1], [3, -1], [5, -1.5], [7, -2]])
D = np.linalg.norm(P[:, None] - P, axis=2)
A = np.zeros((6, 6), bool)
A[np.arange(6)[:, None], np.argsort(D)[:, 1:3]] = True
W = np.where(A | A.T, 1 - D / np.sqrt(53), 0)
L = np.diag(W.sum(1)) - W
lam, V = np.linalg.eigh(L)
print(W.round(3), L.round(3), lam.round(3), V.round(3), sep="\n")

fig, ax = plt.subplots(figsize=(4.5, 2.4), layout="constrained")
for f, marker in ((0, "o"), (3, "s")):
    ax.scatter(*P[f:f + 3].T, marker=marker)
for i, p in enumerate(P, 1):
    ax.annotate(i, p, xytext=(-3, 5), textcoords="offset points")
for v, s, name in (((0.944, -0.331), 2, "$v_1$"),
                   ((0.331, 0.944), 1, "$v_2$")):
    ax.annotate(name, P.mean(0), P.mean(0) + s * np.array(v),
                arrowprops={"arrowstyle": "<-"})
ax.set(aspect="equal", xlabel="x", ylabel="y")
fig.savefig("figures/pca.pdf")

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
