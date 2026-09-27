#!/usr/bin/env python3
"""Clustering tendency and entropy-based measures (CS-E4650).

Reproduces every figure, table and number of the report:
  1. loads the data and checks that it has 1000 three-dimensional vectors
     with all values in [0, 1],
  2. draws the 64-bin histograms and the pairwise scatter plots,
  3. divides every subset of the dimensions into m = 64 uniform grid cells
     and computes the probability-based entropy of Aggarwal's eq. (6.2),
  4. simulates greedy backward and forward selection with these entropies,
  5. runs the extra experiments (Shannon entropy, uniform-data baseline,
     pairwise dependence and m = 729).

Usage: python3 code/clustering_tendency.py
"""
import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only, no display needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "clustering-tendency.csv"
FIG_DIR, RES_DIR = ROOT / "figures", ROOT / "results"
TEX_DIR = RES_DIR / "latex"

M = 64  # number of grid cells in every entropy calculation
M_EXTRA = 729  # the next m that is both a square and a cube (extra experiment)
DIMS = (0, 1, 2)
# all non-empty subsets of the dimensions: (0,1,2), (0,1), (0,2), (1,2), (0,), (1,), (2,)
SUBSETS = [s for k in (3, 2, 1) for s in itertools.combinations(DIMS, k)]
N_SIM = 10_000  # simulated uniform data sets for the entropy baseline
N_PERM = 2_000  # permutations in the independence test
rng = np.random.default_rng(2025)


# ------------------------------------------------------------------ data ---
def load_data(path=DATA_FILE):
    """Load the data and check the properties stated in the task."""
    X = np.loadtxt(path, delimiter=",")
    assert X.shape == (1000, 3), f"unexpected shape {X.shape}"
    assert np.isfinite(X).all(), "non-numeric values"
    assert X.min() >= 0.0 and X.max() <= 1.0, "values outside [0, 1]"
    return X


# ------------------------------------------------------------- entropies ---
def ranges_per_dim(m, k):
    """Ranges per dimension phi such that a k-dimensional grid has phi**k = m cells."""
    phi = round(m ** (1.0 / k))
    if phi**k != m:
        raise ValueError(f"m = {m} is not the {k}th power of an integer")
    return phi


def cell_probabilities(X, dims, m=M):
    """Fractions p_1..p_m of the points in the m uniform grid cells of
    [0,1]^k spanned by the dimensions `dims` (k = len(dims))."""
    Y = X[:, list(dims)]
    k = Y.shape[1]
    phi = ranges_per_dim(m, k)
    # range index floor(phi * x) of every value; x = 1 goes to the last range
    idx = np.minimum(np.floor(Y * phi).astype(int), phi - 1)
    counts = np.bincount(np.ravel_multi_index(idx.T, (phi,) * k), minlength=m)
    # sanity check against numpy's own multidimensional histogram
    edges = [np.linspace(0.0, 1.0, phi + 1)] * k
    assert np.array_equal(counts, np.histogramdd(Y, bins=edges)[0].ravel())
    return counts / len(Y)


def xlogx(p):
    """Elementwise p ln p with the convention 0 ln 0 = 0."""
    p = np.asarray(p, dtype=float)
    out = np.zeros_like(p)
    out[p > 0] = p[p > 0] * np.log(p[p > 0])
    return out


def entropy(p):
    """Probability-based entropy of Aggarwal's eq. (6.2), natural logarithm:
    E = -sum_i [p_i ln p_i + (1 - p_i) ln(1 - p_i)]."""
    return -np.sum(xlogx(p) + xlogx(1.0 - p), axis=-1)


def shannon(p):
    """Shannon entropy -sum_i p_i ln p_i (extra experiment)."""
    return -np.sum(xlogx(p), axis=-1)


def all_entropies(X, m=M, measure=entropy):
    """Entropy of every non-empty subset of the dimensions."""
    return {s: float(measure(cell_probabilities(X, s, m))) for s in SUBSETS}


# ------------------------------------------------------ greedy searches ---
def greedy_backward(E, dims=DIMS):
    """Start from all dimensions and repeatedly drop the dimension whose
    removal gives the lowest entropy; stop when even the best removal
    would not decrease the entropy."""
    current, stages = tuple(dims), []
    while len(current) > 1:
        candidates = [tuple(d for d in current if d != r) for r in current]
        best = min(candidates, key=E.get)
        accepted = E[best] < E[current]
        stages.append((current, candidates, best, accepted))
        if not accepted:
            break
        current = best
    return current, stages


def greedy_forward(E, dims=DIMS):
    """Start from the empty set and repeatedly add the dimension that gives
    the lowest entropy; stop when even the best addition would not decrease
    the entropy. The empty set has no entropy on a 64-cell grid, so the
    best single dimension is always taken in the first stage."""
    current, stages = (), []
    while len(current) < len(dims):
        candidates = [tuple(sorted(current + (a,))) for a in dims if a not in current]
        best = min(candidates, key=E.get)
        accepted = not current or E[best] < E[current]
        stages.append((current, candidates, best, accepted))
        if not accepted:
            break
        current = best
    return current, stages


# ---------------------------------------------------- extra experiments ---
def uniform_baseline(m, n):
    """Entropies of N_SIM simulated data sets of n uniformly distributed
    points. Every cell of a uniform grid has probability 1/m whatever the
    dimensionality, so the cell counts are multinomial(n; 1/m, ..., 1/m)."""
    counts = rng.multinomial(n, np.full(m, 1.0 / m), size=N_SIM)
    return entropy(counts / n)


def mutual_information(x, y, phi=8):
    """Mutual information (nats) of two components on the phi x phi grid."""
    p = cell_probabilities(np.column_stack([x, y]), (0, 1), phi * phi)
    p = p.reshape(phi, phi)
    return float(shannon(p.sum(axis=1)) + shannon(p.sum(axis=0)) - shannon(p.ravel()))


def dependence(X, i, j):
    """Pearson correlation, mutual information and a permutation test of
    independence (shuffling component j destroys any dependence)."""
    mi = mutual_information(X[:, i], X[:, j])
    null = np.array(
        [mutual_information(X[:, i], rng.permutation(X[:, j])) for _ in range(N_PERM)]
    )
    return {
        "r": float(np.corrcoef(X[:, i], X[:, j])[0, 1]),
        "mi": mi,
        "null_mean": float(null.mean()),
        "null_q99": float(np.quantile(null, 0.99)),
        "p": (1 + int(np.sum(null >= mi))) / (1 + N_PERM),
    }


def describe_groups(X):
    """Rough sizes and centres of the three groups seen in the (0, 2) plane,
    separated by the lines x2 = 0.5 and x0 = 0.5 (quoted in the report)."""
    low = X[:, 2] < 0.5
    groups = {"A": low & (X[:, 0] < 0.5), "B": low & (X[:, 0] >= 0.5), "C": ~low}
    return {g: (int(mask.sum()), X[mask].mean(axis=0)) for g, mask in groups.items()}


# --------------------------------------------------------------- figures ---
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "savefig.bbox": "tight"})


def plot_histograms(X):
    """Histograms of the three components with M = 64 bins on [0, 1]."""
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 6.6), sharey=True)
    for j, ax in enumerate(axes):
        ax.hist(X[:, j], bins=M, range=(0, 1), color="tab:blue",
                edgecolor="white", linewidth=0.4)
        ax.axhline(len(X) / M, color="black", linestyle="--", linewidth=1,
                   label=f"uniform expectation {len(X)}/{M} = {len(X) / M:.1f}")
        ax.set(xlim=(0, 1), title=f"Component {j}",
               xlabel=f"value of component {j}", ylabel="count")
    axes[0].legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "histograms.pdf")
    plt.close(fig)


def plot_scatter(X, i, j):
    """Scatter plot of components i and j with equal axis scales. The grey
    lines are the 8 x 8 grid used for the two-dimensional entropies."""
    fig, ax = plt.subplots(figsize=(2.4, 2.4))
    eighths = np.linspace(0, 1, 9)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_xticks(eighths, minor=True)
    ax.set_yticks(eighths, minor=True)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:g}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:g}")
    ax.tick_params(labelsize=10)
    ax.grid(which="both", color="0.85", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.scatter(X[:, i], X[:, j], s=4, alpha=0.6, linewidths=0, color="tab:blue")
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_xlabel(f"component {i}", fontsize=12)
    ax.set_ylabel(f"component {j}", fontsize=12)
    ax.set_aspect("equal")
    fig.savefig(FIG_DIR / f"scatter_{i}_{j}.pdf")
    plt.close(fig)


# ---------------------------------------------------------- LaTeX output ---
def key(s):
    return "".join(map(str, s))


def tex_set(s):
    return r"$\{" + ",".join(map(str, s)) + r"\}$" if s else r"$\emptyset$"


def entropy_table(E, empty, e_max):
    rank = {s: r for r, s in enumerate(sorted(E, key=E.get), 1)}
    rows = [r"\begin{tabular}{lcccccc}", r"\toprule",
            r"Dimensions & $k$ & Grid & Empty cells & $E$ & $E/E_{\max}$ & Rank \\"]
    for s in SUBSETS:
        phi = ranges_per_dim(M, len(s))
        if s in ((0, 1, 2), (0, 1), (0,)):  # a rule before each dimensionality
            rows.append(r"\midrule")
        grid = "$" + r"\times".join([str(phi)] * len(s)) + "$"
        rows.append(f"{tex_set(s)} & {len(s)} & {grid} & {empty[s]} & {E[s]:.3f} & "
                    f"{100 * E[s] / e_max:.1f}\\,\\% & {rank[s]} \\\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(rows) + "\n"


def greedy_table(E, stages, backward):
    rows = [r"\begin{tabular}{cllrrl}", r"\toprule",
            r"Stage & Current set ($E$) & Candidate set & $E$ & $\Delta E$ & Decision \\"]
    for n, (current, candidates, best, accepted) in enumerate(stages, 1):
        rows.append(r"\midrule")
        head = f"{n} & {tex_set(current)}" + (f" ({E[current]:.3f})" if current else "")
        for c in candidates:
            moved = (set(current) ^ set(c)).pop()
            what = f"(remove {moved})" if backward else f"(add {moved})"
            delta = f"${E[c] - E[current]:+.3f}$" if current else "--"
            value = f"{E[c]:.3f}"
            decision = ""
            if c == best:
                value = r"\textbf{" + value + "}"
                decision = "selected" if accepted else r"stop: $E$ would increase"
            rows.append(f"{head} & {tex_set(c)} {what} & {value} & {delta} & {decision} \\\\")
            head = "&"  # stage and current set only on the first row of a stage
    rows += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(rows) + "\n"


def extra_table(H, z, E_big, z_big):
    rows = [r"\begin{tabular}{lcccc}", r"\toprule",
            r" & \multicolumn{2}{c}{$m=64$} & \multicolumn{2}{c}{$m=729$} \\",
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
            r"Dimensions & Shannon $H$ & $z$ of $E$ & $E$ & $z$ of $E$ \\"]
    for s in SUBSETS:
        if s in ((0, 1, 2), (0, 1), (0,)):
            rows.append(r"\midrule")
        rows.append(f"{tex_set(s)} & {H[s]:.4f} & ${z[s]:.1f}$ & {E_big[s]:.3f} & "
                    f"${z_big[s]:.1f}$ \\\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(rows) + "\n"


def dependence_table(dep):
    rows = [r"\begin{tabular}{lccccc}", r"\toprule",
            r" & & & \multicolumn{2}{c}{MI of permuted data} & \\",
            r"\cmidrule(lr){4-5}",
            r"Pair & Pearson $r$ & MI & mean & 99\,\% quantile & $p$-value \\",
            r"\midrule"]
    for (i, j), d in dep.items():
        p = r"$<0.001$" if d["p"] < 0.001 else f"{d['p']:.2f}"
        rows.append(f"{tex_set((i, j))} & {d['r']:.3f} & {d['mi']:.3f} & "
                    f"{d['null_mean']:.3f} & {d['null_q99']:.3f} & {p} \\\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(rows) + "\n"


def write_macros(values):
    """Numbers quoted in the report text, read by report.tex as \\val{name}."""
    with open(TEX_DIR / "macros.tex", "w") as f:
        f.write("% generated by code/clustering_tendency.py -- do not edit\n")
        for name, value in values.items():
            f.write(f"\\expandafter\\def\\csname val@{name}\\endcsname{{{value}}}\n")


# ------------------------------------------------------------------ main ---
def main():
    for d in (FIG_DIR, RES_DIR, TEX_DIR):
        d.mkdir(parents=True, exist_ok=True)
    X = load_data()
    n = len(X)

    # visual inspection (tasks 2-3)
    plot_histograms(X)
    for i, j in itertools.combinations(DIMS, 2):
        plot_scatter(X, i, j)

    # entropies with m = 64 cells (tasks 5-7)
    P = {s: cell_probabilities(X, s) for s in SUBSETS}
    E = {s: float(entropy(p)) for s, p in P.items()}
    empty = {s: int(np.sum(p == 0)) for s, p in P.items()}
    e_max = float(entropy(np.full(M, 1.0 / M)))

    # greedy searches (tasks 9-10)
    back, back_stages = greedy_backward(E)
    fwd, fwd_stages = greedy_forward(E)

    # extra experiments
    H = all_entropies(X, M, shannon)
    base = uniform_baseline(M, n)
    z = {s: (E[s] - base.mean()) / base.std() for s in SUBSETS}
    E_big = all_entropies(X, M_EXTRA)
    base_big = uniform_baseline(M_EXTRA, n)
    z_big = {s: (E_big[s] - base_big.mean()) / base_big.std() for s in SUBSETS}
    e_max_big = float(entropy(np.full(M_EXTRA, 1.0 / M_EXTRA)))
    dep = {(i, j): dependence(X, i, j) for i, j in itertools.combinations(DIMS, 2)}
    groups = describe_groups(X)
    greedy_extra = {
        "Shannon, m=64": (greedy_backward(H)[0], greedy_forward(H)[0]),
        "eq. (6.2), m=729": (greedy_backward(E_big)[0], greedy_forward(E_big)[0]),
    }

    # ---- text summary
    out = [f"data: {X.shape[0]} vectors x {X.shape[1]} components, "
           f"min {X.min():.4f}, max {X.max():.4f}, "
           f"values exactly 0 or 1: {int(np.sum((X == 0) | (X == 1)))}",
           f"max entropy (uniform p_i = 1/{M}): E_max = {e_max:.4f}",
           f"uniform data, n = {n}: E = {base.mean():.4f} +- {base.std():.4f}",
           "", "subset      grid   empty  E(6.2)   E/Emax  Shannon  z(unif)  E(m=729)  z(m=729)"]
    for s in SUBSETS:
        phi = ranges_per_dim(M, len(s))
        out.append(f"{str(s):10s} {phi:>3d}^{len(s)} {empty[s]:>5d}  {E[s]:.4f}  "
                   f"{E[s] / e_max:6.1%}  {H[s]:.4f}  {z[s]:7.1f}  {E_big[s]:.4f}  "
                   f"{z_big[s]:8.1f}")
    for title, stages, result in (("backward", back_stages, back),
                                  ("forward", fwd_stages, fwd)):
        out += ["", f"greedy {title} selection:"]
        for current, candidates, best, accepted in stages:
            cands = ", ".join(f"{c}: {E[c]:.4f}" for c in candidates)
            cur = f"{current} (E = {E[current]:.4f})" if current else "()"
            out.append(f"  from {cur}: {cands} -> "
                       f"{'take ' + str(best) if accepted else 'stop'}")
        out.append(f"  result: {result}, E = {E[result]:.4f}")
    out += ["", f"uniform data, m = {M_EXTRA}: E_max = {e_max_big:.4f}, "
                f"E = {base_big.mean():.4f} +- {base_big.std():.4f}"]
    for name, (b, f) in greedy_extra.items():
        out.append(f"greedy with {name}: backward {b}, forward {f}")
    out.append("")
    for (i, j), d in dep.items():
        out.append(f"pair ({i},{j}): r = {d['r']:.3f}, MI = {d['mi']:.4f}, "
                   f"MI under independence {d['null_mean']:.4f} "
                   f"(99% {d['null_q99']:.4f}), p = {d['p']:.4f}")
    for g, (size, centre) in groups.items():
        out.append(f"group {g}: {size} points, mean {np.round(centre, 3)}")
    summary = "\n".join(out)
    print(summary)
    (RES_DIR / "summary.txt").write_text(summary + "\n")

    # ---- machine-readable results
    with open(RES_DIR / "entropies.csv", "w") as f:
        f.write("dimensions,k,ranges_per_dim,cells,empty_cells,E_eq6_2,E_over_Emax,"
                "shannon,z_uniform,E_eq6_2_m729,z_uniform_m729\n")
        for s in SUBSETS:
            f.write(f"{key(s)},{len(s)},{ranges_per_dim(M, len(s))},{M},{empty[s]},"
                    f"{E[s]:.6f},{E[s] / e_max:.6f},{H[s]:.6f},{z[s]:.3f},"
                    f"{E_big[s]:.6f},{z_big[s]:.3f}\n")

    # ---- LaTeX tables and macros for the report
    (TEX_DIR / "entropy_table.tex").write_text(entropy_table(E, empty, e_max))
    (TEX_DIR / "backward_table.tex").write_text(greedy_table(E, back_stages, True))
    (TEX_DIR / "forward_table.tex").write_text(greedy_table(E, fwd_stages, False))
    (TEX_DIR / "extra_table.tex").write_text(extra_table(H, z, E_big, z_big))
    (TEX_DIR / "dependence_table.tex").write_text(dependence_table(dep))

    counts1 = np.round(P[(1,)] * n).astype(int)
    v = {"N": n, "NBoundary": int(np.sum((X == 0) | (X == 1))),
         "Emax": f"{e_max:.3f}", "EmaxBig": f"{e_max_big:.3f}",
         "UnifMean": f"{base.mean():.3f}", "UnifSd": f"{base.std():.3f}",
         "UnifMin": f"{base.min():.3f}",
         "UnifMeanBig": f"{base_big.mean():.3f}", "UnifSdBig": f"{base_big.std():.3f}",
         "HistMinOne": counts1.min(), "HistMaxOne": counts1.max(),
         "LowTwo": int(np.sum(X[:, 2] < 0.5)), "HighTwo": int(np.sum(X[:, 2] >= 0.5)),
         "GapTwo": int(np.sum((X[:, 2] >= 0.45) & (X[:, 2] < 0.55))),
         "NSim": f"{N_SIM:,}".replace(",", r"\,"), "NPerm": f"{N_PERM:,}".replace(",", r"\,"),
         "Back": tex_set(back), "Fwd": tex_set(fwd)}
    for s in SUBSETS:
        v |= {f"E{key(s)}": f"{E[s]:.3f}", f"H{key(s)}": f"{H[s]:.4f}",
              f"Z{key(s)}": f"{z[s]:.1f}", f"EBig{key(s)}": f"{E_big[s]:.3f}",
              f"ZBig{key(s)}": f"{z_big[s]:.1f}", f"Empty{key(s)}": empty[s],
              f"Rel{key(s)}": f"{100 * E[s] / e_max:.1f}",
              f"Diff{key(s)}": f"{E[s] - H[s]:.3f}"}
        for t in SUBSETS:
            v[f"D{key(s)}-{key(t)}"] = f"{E[s] - E[t]:+.3f}"
    for (i, j), d in dep.items():
        v |= {f"R{i}{j}": f"{d['r']:.3f}", f"MI{i}{j}": f"{d['mi']:.3f}",
              f"MINull{i}{j}": f"{d['null_mean']:.3f}"}
    for g, (size, centre) in groups.items():
        v |= {f"Size{g}": size, f"Centre{g}": f"({centre[0]:.2f}, {centre[2]:.2f})"}
    for name, (b, f) in zip(("Sh", "Big"), greedy_extra.values()):
        v |= {f"Back{name}": tex_set(b), f"Fwd{name}": tex_set(f)}
    write_macros(v)


if __name__ == "__main__":
    main()
