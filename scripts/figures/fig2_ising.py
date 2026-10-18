"""Fig. 2: 2D Ising at h = 0 against the exact finite-L Kaufman solution and a Wolff reference.

Rolls out M walkers per temperature with the learned transport and with the prior alone (Q = 0),
and writes a results npz that `--results` and fig9_ising_multiscale.py replot:
    python scripts/figures/fig2_ising.py --ckpt <run>/last.ckpt --M 5000
    python scripts/figures/fig2_ising.py --results results/ising/ising_L20.npz
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

from frogs.analysis.ising_exact import T_C, kaufman_thermo, wolff_reference
from frogs.analysis.ising_readout import heat_capacity, load_results, readout, save_results
from frogs.analysis.sweep_io import unpack_sigma
from frogs.evaluate import ISEvaluator, load_model
from frogs.paths import RESULTS_DIR, ensure_dir, figure_path

# Dense through the critical window, inside the training box T in [1.5, 3.0].
T_DEFAULT = [1.50, 1.65, 1.80, 1.95, 2.05, 2.15, 2.22, T_C, 2.32, 2.40, 2.50, 2.65, 2.80, 2.95]
ARMS = {"transport": ("autoregressive + transport", "tab:orange", 5),
        "prior": ("no transport at eval", "tab:blue", 3)}
FS = 21


def rollout(model, T, M, transport, chunk, seed, device):
    """Log-weights A (n_T, M), spins (n_T, M, L, L) and the packed spins at h = 0."""
    ev = ISEvaluator(model, M=M, n_mcmc=0, seed=seed, device=device, transport=transport)
    res = ev.run([0.0], T, chunk=chunk, n_boot=0, store_A=True, store_sigma=True, verbose=True)
    L = model.energy.L
    packed = res["sigma_packed"][0]
    return res["A_samples"][0], unpack_sigma(packed, L * L).reshape(len(T), M, L, L), packed


def evaluate(args):
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device)
    L = model.energy.L
    N = L * L
    T = np.array(sorted(args.T or T_DEFAULT), dtype=np.float64)

    arms = {}
    for name in ARMS:
        print(f"[{name}] M={args.M} at {len(T)} temperatures")
        A, spins, packed = rollout(model, T, args.M, name == "transport", args.chunk,
                                   args.seed, device)
        arms[name] = readout(A, spins, T, seed=args.seed)
        arms[name]["sigma_packed"] = packed

    exact = np.array([kaufman_thermo(1.0 / t, L) for t in T])
    j_gr = int(np.argmin(np.abs(T - T_C)))
    print(f"Wolff reference at T={T[j_gr]:.4f} (4 chains x {args.wolff_meas} measurements)")
    G_w, G_w_se = wolff_reference(1.0 / T[j_gr], L, n_meas=args.wolff_meas)
    common = dict(L=L, T=T, T_gr=T[j_gr], r=np.arange(1, L // 2 + 1),
                  logZ_exact=exact[:, 0], F_exact=-T * exact[:, 0] / N, U_exact=exact[:, 1],
                  C_exact=exact[:, 2], C_exact_smoothed=heat_capacity(T, exact[:, 1])[0],
                  G_wolff=G_w, G_wolff_se=G_w_se)
    meta = dict(ckpt=os.path.basename(args.ckpt), M=args.M, n_steps=int(model.config.n_steps),
                n_mcmc=0, seed=args.seed, wolff_meas=args.wolff_meas)
    out = args.out or os.path.join(ensure_dir(os.path.join(RESULTS_DIR, "ising")),
                                   f"ising_L{L}.npz")
    save_results(out, common, arms, meta)
    print(f"saved {out}")
    return out


def _finish(ax, xlabel, ylabel, handles=None, letter="", tc=True):
    if tc:
        ax.axvline(T_C, color="grey", ls="--", lw=1.2, zorder=0)
    ax.set_xlabel(xlabel, fontsize=FS)
    ax.set_ylabel(ylabel, fontsize=FS)
    ax.tick_params(axis="both", labelsize=FS)
    if handles:
        ax.legend(handles=handles, fontsize=FS, loc="lower left", bbox_to_anchor=(0.0, 1.0),
                  borderaxespad=0.0, framealpha=0.9, ncol=len(handles), handlelength=1.6)
    ax.text(-0.135, 1.03, letter, transform=ax.transAxes, fontsize=26, fontweight="bold",
            va="bottom", ha="left")


def plot(common, arms, out):
    T = common["T"]
    tlabel = "$T$ ($J/k_B$)"
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 10.4))
    (ax_f, ax_ess), (ax_c, ax_g) = axes

    for name, (_, color, z) in ARMS.items():
        a = arms[name]
        ax_f.errorbar(T, a["F_per"], yerr=a["F_per_se"], marker="o", ms=5.5, lw=1.8, color=color,
                      capsize=3, zorder=z)
        ax_ess.plot(T, a["ess"], marker="o", ms=5.5, lw=1.8, color=color, zorder=z)
    ax_f.plot(T, common["F_exact"], "k-", lw=2.4, alpha=0.7, zorder=1)
    for ax in (ax_f, ax_ess):
        ax.set_xticks([1.5, 2.0, 2.5, 3.0])
    _finish(ax_f, tlabel, "$F/N$", letter="(a)")
    ax_ess.set_yscale("log")
    _finish(ax_ess, tlabel, "ESS", letter="(b)")

    # (c) C/N = d(U/N)/dT of the transport arm; the exact U goes through the same smoother.
    a = arms["transport"]
    ref, = ax_c.plot(T, common["C_exact_smoothed"], "k--", lw=2.2, alpha=0.75, zorder=1,
                     label="reference")
    ax_c.errorbar(T, a["C"], yerr=a["C_se"], marker="o", ms=6.0, lw=2.2, color="tab:orange",
                  capsize=3, zorder=3)
    finite = np.isfinite(common["C_exact_smoothed"]) | np.isfinite(a["C"])
    if finite.any():
        ax_c.set_xlim(T[finite].min() - 0.06, T[finite].max() + 0.06)
    _finish(ax_c, tlabel, "$C/N$ ($k_B$)", handles=[ref], letter="(c)")

    j = int(np.argmin(np.abs(T - common["T_gr"])))
    r = common["r"]
    for name, (_, color, z) in ARMS.items():
        ax_g.errorbar(r, arms[name]["G"][j], yerr=arms[name]["G_se"][j], marker="o", ms=6.5,
                      lw=1.9, capsize=3, color=color, zorder=z)
    wolff = ax_g.errorbar(r, common["G_wolff"], yerr=common["G_wolff_se"], marker="^", ms=5.5,
                          lw=1.5, ls="--", color="k", capsize=3, alpha=0.8, zorder=2,
                          label="Wolff")
    ax_g.set_xticks(r)
    ax_g.grid(alpha=0.3)
    _finish(ax_g, "$r$ (lattice spacings)", r"$G(r)=\langle \sigma_0\sigma_r\rangle$",
            handles=[wolff], letter="(d)", tc=False)

    shared = [Line2D([], [], color=c, marker="o", ms=6.5, lw=2.0, label=lab)
              for lab, c, _ in ARMS.values()]
    shared.append(Line2D([], [], color="k", lw=2.4, alpha=0.7, label="Kaufman exact"))
    fig.legend(handles=shared, loc="upper center", ncol=3, fontsize=FS, frameon=False,
               bbox_to_anchor=(0.5, 1.005), columnspacing=2.0)
    fig.tight_layout(w_pad=3.0, h_pad=2.6, rect=(0.0, 0.0, 1.0, 0.955))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def report(common, arms):
    N = int(common["L"]) ** 2
    print(f"{'arm':>10}{'T':>8}{'ESS':>9}{'F/N':>10}{'exact':>10}{'U/N':>10}{'exact':>10}"
          f"{'m>0 (w)':>9}{'m>0 (raw)':>10}")
    for name, a in arms.items():
        for j, t in enumerate(common["T"]):
            print(f"{name:>10}{t:>8.4f}{a['ess'][j]:>9.4f}{a['F_per'][j]:>10.5f}"
                  f"{common['F_exact'][j]:>10.5f}{a['U_per'][j]:>10.5f}"
                  f"{common['U_exact'][j]:>10.5f}{a['pos_mass'][j]:>9.3f}{a['pos_raw'][j]:>10.3f}")
    dz = (arms["transport"]["logZ"] - common["logZ_exact"]) / N
    print("transport (log Z - exact) / N:", " ".join(f"{d:+.1e}" for d in dz))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt", help="trained 2D-Ising checkpoint")
    src.add_argument("--results", help="replot a results npz written by an earlier run")
    ap.add_argument("--M", type=int, default=5000, help="walkers per temperature")
    ap.add_argument("--T", type=float, nargs="+", default=None)
    ap.add_argument("--chunk", type=int, default=40,
                    help="walkers per rollout batch (40 needs about 5 GB of GPU memory at L = 20)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wolff-meas", type=int, default=20000, help="measurements per Wolff chain")
    ap.add_argument("--out", default=None, help="default results/ising/ising_L<L>.npz")
    ap.add_argument("--fig", default=None, help="default figures/fig2_ising_L<L>.png")
    args = ap.parse_args()

    path = args.results or evaluate(args)
    common, arms, _ = load_results(path)
    report(common, arms)
    plot(common, arms, args.fig or figure_path(f"fig2_ising_L{int(common['L'])}.png"))


if __name__ == "__main__":
    main()
