"""Fig. 7: variant shares at each phase's deepest-ordered sweep point (uniform-prior CuAu ablation).

    python scripts/figures/fig7_variant_coverage.py --sweep results/sweeps/<cuau-uniform-tag>.npz

Bars are the unweighted share of gated samples per degenerate variant, diamonds the share after
the e^A importance weights.
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

from frogs.analysis.sweep_io import load_sweep
from frogs.analysis.variants import PHASES, SWEEP_KEYS, deepest_ordered_points, variant_histograms
from frogs.paths import figure_path

VARIANT_COLORS = ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a", "#eb6834"]
PHASE_COLORS = {"Cu3Au": "#2a78d6", "CuAu": "#008300", "CuAu3": "#eb6834"}
FS_LETTER, FS_AXIS, FS_TICK, FS_VAL, FS_PHASE, FS_LEG = 17, 15, 13, 11, 15, 15


def plot_points(ax, dmu, T, points):
    for k, ph in enumerate(PHASES):
        i, j = points[ph.name]
        ax.plot(dmu[i], T[j], "o", ms=13, mfc="none", mec=PHASE_COLORS[ph.name], mew=2.4, zorder=5)
        # the three points sit close together, so stagger the labels
        ax.annotate(ph.label, (dmu[i], T[j]), textcoords="offset points", xytext=(0, 14 + 26 * k),
                    ha="center", fontsize=FS_PHASE, fontweight="bold", color=PHASE_COLORS[ph.name])
    pad_x, pad_y = 0.06 * (dmu[-1] - dmu[0]), 0.10 * (T[-1] - T[0])
    ax.set_xlim(dmu[0] - pad_x, dmu[-1] + pad_x)
    ax.set_ylim(T[0] - pad_y, T[-1] + pad_y)
    ax.set_xlabel(r"$\Delta\mu$ (eV)", fontsize=FS_AXIS)
    ax.set_ylabel("$T$ (K)", fontsize=FS_AXIS)
    ax.tick_params(labelsize=FS_TICK)


def plot_shares(ax, counts, wsum, labels):
    g = len(labels)
    p, pw = counts / counts.sum(), wsum / wsum.sum()
    x = np.arange(g)
    ax.bar(x, p, color=VARIANT_COLORS[:g], edgecolor="white", linewidth=1.5)
    dia, = ax.plot(x, pw, "D", color="#0b0b0b", ms=8, zorder=5)
    ax.axhline(1 / g, color="#52514e", ls="--", lw=1.2)
    top = min(1.06, max(0.44, 1.18 * max(p.max(), pw.max())))
    for k in range(g):
        ax.text(k, p[k] + 0.02 * top, f"{p[k]:.2f}", ha="center", va="bottom",
                fontsize=FS_VAL, color="#52514e")
        if abs(pw[k] - p[k]) > 0.03 * top:
            ax.text(k, pw[k] + 0.02 * top, f"{pw[k]:.2f}", ha="center", va="bottom",
                    fontsize=FS_VAL, color="#0b0b0b", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FS_TICK)
    ax.set_ylim(0, top)
    ax.set_ylabel("share of ordered walkers", fontsize=FS_AXIS)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    return dia


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sweep", required=True, help="CuAu sweep with A_samples and sigma_packed")
    ap.add_argument("--out", default=None, help="default: figures/fig7_variant_coverage.png")
    args = ap.parse_args()

    sweep = load_sweep(args.sweep, keys=SWEEP_KEYS)
    dmu, T = sweep["dmu"], sweep["T"]
    stats = variant_histograms(sweep, verbose=True)
    del sweep
    points = deepest_ordered_points(stats, T)

    fig = plt.figure(figsize=(14.5, 10.5))
    gs = gridspec.GridSpec(2, 2, hspace=0.32, wspace=0.28, left=0.07, right=0.895, top=0.94, bottom=0.07)
    ax = fig.add_subplot(gs[0, 0])
    plot_points(ax, dmu, T, points)
    ax.set_title("(a)", fontsize=FS_LETTER, fontweight="bold", loc="left")
    for letter, pos, ph in zip("bcd", (gs[0, 1], gs[1, 0], gs[1, 1]), PHASES):
        i, j = points[ph.name]
        st = stats[ph.name]
        counts, wsum = st.counts[i, j].astype(float), st.wsum[i, j]
        ax = fig.add_subplot(pos)
        dia = plot_shares(ax, counts, wsum, ph.variants)
        ax.set_title(f"({letter})", fontsize=FS_LETTER, fontweight="bold", loc="left")
        print(f"{ph.name:6s} dmu={dmu[i]:+.3f} T={T[j]:.0f} K  gated {st.n_gated[i, j]}  "
              f"shares {np.round(counts / counts.sum(), 3).tolist()}  "
              f"weighted {np.round(wsum / wsum.sum(), 3).tolist()}")
    fig.legend([dia], ["IS-weighted"], loc="upper left", bbox_to_anchor=(0.9, 0.965),
               fontsize=FS_LEG, frameon=False, handletextpad=0.6, markerscale=1.2)

    out = args.out or figure_path("fig7_variant_coverage.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
