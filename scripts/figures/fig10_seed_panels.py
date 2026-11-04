"""Fig. 10: CuAu binodals from independently trained models, one panel each.

Each input is a sweep npz or a rungs npz written by fig3_phase_diagram.py:
    python scripts/figures/fig10_seed_panels.py \
        --panel "seed 1=figures/fig3_cuau_<sweep_s1>.npz" \
        --panel "seed 2=<sweep_s2>.npz" --panel "seed 3=..." --panel "seed 4=..."
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from frogs.analysis.phase_diagram import PhaseDiagram, construct_phase_diagram
from frogs.analysis.plotting import draw_binodal
from frogs.paths import figure_path


def load_diagram(path):
    with np.load(path, allow_pickle=True) as z:
        is_rungs = "rung_T" in z.files
    return PhaseDiagram.load(path) if is_rungs else construct_phase_diagram(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--panel", action="append", required=True, metavar="LABEL=PATH")
    ap.add_argument("--system", default="CuAu")
    ap.add_argument("--t-max", type=float, default=900.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    panels = []
    for spec in args.panel:
        if "=" not in spec:
            ap.error(f"--panel needs LABEL=PATH, got {spec!r}")
        label, path = (s.strip() for s in spec.split("=", 1))
        panels.append((label, load_diagram(path)))
    T0 = panels[0][1].T
    for label, diagram in panels[1:]:
        if diagram.T.shape != T0.shape or not np.allclose(diagram.T, T0):
            raise SystemExit(f"panel {label!r} has a different T grid from {panels[0][0]!r}")

    ncol = 2
    nrow = -(-len(panels) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.0 * ncol, 5.0 * nrow), sharex=True,
                             sharey=True, squeeze=False)
    axes = axes.ravel()
    for k, (label, diagram) in enumerate(panels):
        ax = axes[k]
        draw_binodal(ax, diagram, args.system, label="FrOGS", t_max=args.t_max, fontsize=13,
                     legend=False)
        if k == 0:
            ax.legend(loc="upper right", fontsize=12, framealpha=0.93)
        ax.set_title(label, fontsize=12)
        if k % ncol:
            ax.set_ylabel("")
        if k < len(panels) - ncol:
            ax.set_xlabel("")
        print(f"{label:24s} {diagram.n_rungs:3d} rungs, {len(diagram.lobes)} lobes, "
              f"ordered compounds at x = {[round(x, 3) for x, _ in diagram.compounds()]}")
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.tight_layout()
    out = args.out or figure_path("fig10_seed_panels.png")
    fig.savefig(out, dpi=160)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
