"""Fig. 9: 2D Ising at L = 10, 15, 20 on shared axes (§A.12), from fig2_ising.py results files.

    python scripts/figures/fig9_ising_multiscale.py --results results/ising/ising_L{10,15,20}.npz
"""
from __future__ import annotations

import argparse
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from frogs.analysis.ising_exact import T_C
from frogs.analysis.ising_readout import load_results
from frogs.paths import RESULTS_DIR, figure_path

L_COLORS = {10: "tab:green", 15: "tab:orange", 20: "tab:blue"}
FS = 16


def load_run(path):
    common, arms, _ = load_results(path)
    a = arms["transport"]
    L = int(common["L"])
    j = int(np.argmin(np.abs(common["T"] - common["T_gr"])))
    return dict(L=L, N=L * L, T=common["T"], ess=a["ess"], logZ=a["logZ"], logZ_se=a["logZ_se"],
                logZ_exact=common["logZ_exact"], pos_mass=a["pos_mass"],
                pos_mass_se=a["pos_mass_se"], A=a["A"], r=common["r"], G=a["G"][j],
                G_se=a["G_se"][j], G_wolff=common["G_wolff"], G_wolff_se=common["G_wolff_se"])


def _finish(ax, xlabel, ylabel, letter, tc=True):
    if tc:
        ax.axvline(T_C, color="0.55", lw=1.0, ls=":", zorder=0)
        ax.set_xticks([1.5, 2.0, 2.5, 3.0])
    ax.set_xlabel(xlabel, fontsize=FS)
    ax.set_ylabel(ylabel, fontsize=FS)
    ax.tick_params(axis="both", labelsize=FS)
    ax.text(-0.135, 1.03, letter, transform=ax.transAxes, fontsize=20, fontweight="bold",
            va="bottom", ha="left")


def plot(runs, out):
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 11.0))
    (ax_ess, ax_dz), (ax_bal, ax_gr) = axes
    style = dict(marker="o", ms=5.5, lw=1.8)
    for r in runs:
        c = L_COLORS.get(r["L"])
        ax_ess.plot(r["T"], r["ess"], color=c, **style)
        ax_dz.errorbar(r["T"], (r["logZ"] - r["logZ_exact"]) / r["N"],
                       yerr=r["logZ_se"] / r["N"], color=c, capsize=3, **style)
        ax_bal.errorbar(r["T"], r["pos_mass"], yerr=r["pos_mass_se"], color=c, capsize=3,
                        **style)
        ax_gr.errorbar(r["r"], r["G"], yerr=r["G_se"], color=c, capsize=3, **style)
        ax_gr.plot(r["r"], r["G_wolff"], lw=1.6, ls="--", color=c, alpha=0.8)

    ax_ess.set_ylim(0.0, 1.0)
    _finish(ax_ess, "$T$", "ESS", "a")
    ax_dz.axhline(0.0, color="0.3", lw=1.2, ls="--", zorder=0)
    _finish(ax_dz, "$T$", r"$(\log \hat{Z} - \log Z_{\mathrm{exact}})\,/\,N$", "b")
    ax_bal.axhline(0.5, color="0.3", lw=1.2, ls="--", zorder=0)
    ax_bal.set_ylim(0.0, 1.0)
    _finish(ax_bal, "$T$", r"IS weight on $m>0$", "c")
    ax_gr.set_xticks(range(0, max(int(r["r"].max()) for r in runs) + 1, 2))
    _finish(ax_gr, "$r$", r"$G(r)$ at $T_c$", "d", tc=False)
    ax_gr.legend(handles=[Line2D([], [], color="0.3", lw=1.8, label="FrOGS"),
                          Line2D([], [], color="0.3", lw=1.6, ls="--", label="Wolff (same $L$)")],
                 fontsize=FS, loc="lower left", framealpha=0.9)

    fig.legend(handles=[Line2D([], [], color=L_COLORS.get(r["L"]), marker="o", lw=2.0,
                               label=f"$L={r['L']}$") for r in runs],
               loc="upper center", ncol=len(runs), fontsize=FS, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def report(runs):
    """At T_c: ESS, the extensive-corrected Var[A]/N, the log Z error and G(r_max) vs Wolff."""
    print(f"at T_c = {T_C:.4f}")
    print(f"{'L':>3}{'ESS':>8}{'Var[A]/N':>10}{'dlogZ/N':>11}{'m>0':>7}  G(r_max) - Wolff")
    for r in runs:
        i = int(np.argmin(np.abs(r["T"] - T_C)))
        dz = (r["logZ"][i] - r["logZ_exact"][i]) / r["N"]
        d = r["G"][-1] - r["G_wolff"][-1]
        sig = math.hypot(r["G_se"][-1], r["G_wolff_se"][-1])
        print(f"{r['L']:>3}{r['ess'][i]:>8.3f}{r['A'][i].var(ddof=1) / r['N']:>10.4f}"
              f"{dz:>11.2e}{r['pos_mass'][i]:>7.3f}  {d:+.3f} ({d / sig:+.1f} sigma)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", nargs="+",
                    default=[os.path.join(RESULTS_DIR, "ising", f"ising_L{L}.npz")
                             for L in (10, 15, 20)])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    runs = sorted((load_run(p) for p in args.results), key=lambda r: r["L"])
    plot(runs, args.out or figure_path("fig9_ising_multiscale.png"))
    report(runs)


if __name__ == "__main__":
    main()
