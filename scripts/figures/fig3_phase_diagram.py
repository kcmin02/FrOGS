"""Fig. 3: binodal from the §A.4 constructor, over the metadynamics reference.

    python scripts/figures/fig3_phase_diagram.py --sweep results/sweeps/cuau_128.npz
    python scripts/figures/fig3_phase_diagram.py --sweep <segal_sweep>.npz --label SEGAL
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from frogs.analysis.phase_diagram import construct_phase_diagram
from frogs.analysis.plotting import draw_binodal
from frogs.analysis.reference import SYSTEMS, system_from_meta
from frogs.paths import figure_path

T_MAX = {"AgPd": 700.0, "CuAu": 900.0}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sweep", required=True, help="sweep npz with A_samples and N_samples")
    ap.add_argument("--label", default="FrOGS", help="legend label of the sampler")
    ap.add_argument("--system", choices=sorted(SYSTEMS), default=None,
                    help="default: read from the sweep meta")
    ap.add_argument("--t-max", type=float, default=None, help="upper T axis limit (display only)")
    ap.add_argument("--out", default=None, help="figure path; rungs are saved next to it as .npz")
    args = ap.parse_args()

    diagram = construct_phase_diagram(args.sweep)
    system = args.system or system_from_meta(diagram.meta)
    stem = os.path.splitext(os.path.basename(args.sweep))[0]
    out = args.out or figure_path(f"fig3_{(system or 'sweep').lower()}_{stem}.png")

    print(f"{system}: N={diagram.n_sites}, {len(diagram.T)} T rows, settings {diagram.settings}")
    if diagram.partial:
        print(f"partial sweep: measured-absence route off ({diagram.n_suppressed} gaps suppressed)")
    for j in sorted(diagram.rungs):
        if diagram.rungs[j]:
            gaps = "  ".join(f"[{g.x_lo:.3f},{g.x_hi:.3f}]"
                             + ("abs" if g.depth == float("inf") else f"{g.depth / g.se:.1f}s")
                             for g in diagram.rungs[j])
            print(f"T={diagram.T[j]:5.0f} K (SE x{diagram.inflation[j]:.2f}): {gaps}")
    print(f"certified rungs: {diagram.n_rungs}, lobes: {len(diagram.lobes)}")
    for L in diagram.lobes:
        print(f"  x in [{L['x_lo']:.3f}, {L['x_hi']:.3f}]  T {L['T_bot']:.0f}-{L['T_top']:.0f} K"
              f"  ({L['n_rungs']} rungs)")
    print("ordered compounds at x =", [round(x, 3) for x, _ in diagram.compounds()])

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    draw_binodal(ax, diagram, system, label=args.label,
                 t_max=args.t_max if args.t_max is not None else T_MAX.get(system),
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    rungs_path = os.path.splitext(out)[0] + ".npz"
    diagram.save(rungs_path)
    print(f"saved {out}\nsaved {rungs_path}")


if __name__ == "__main__":
    main()
