"""Figs. 4 and 6: ESS over the (dmu, T) grid of one or more sweeps, side by side.

    python scripts/figures/fig4_ess_map.py --sweep <agpd>.npz <cuau>.npz
    python scripts/figures/fig4_ess_map.py --sweep <no_transport>.npz <uniform_prior>.npz \
        --labels "autoregressive only" "transport only" --out figures/fig6_ess_ablation.png
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from frogs.analysis.plotting import ESS_FLOOR, draw_ess_map, ess_summary
from frogs.analysis.reference import system_from_meta
from frogs.analysis.sweep_io import load_sweep
from frogs.paths import figure_path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sweep", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", default=None, help="panel titles, one per sweep")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.labels is not None and len(args.labels) != len(args.sweep):
        ap.error("--labels needs one entry per --sweep")

    n = len(args.sweep)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n + 0.9, 3.8), squeeze=False,
                             constrained_layout=True)
    for k, (ax, path) in enumerate(zip(axes[0], args.sweep)):
        s = load_sweep(path, keys=("dmu", "T", "ess"))
        label = args.labels[k] if args.labels else (system_from_meta(s["meta"]) or path)
        im = draw_ess_map(ax, s["dmu"], s["T"], s["ess"])
        ax.set_title(label)
        st = ess_summary(s["ess"])
        missing = st["n_total"] - st["n"]
        print(f"{label}: median ESS = {st['median']:.3f}, ESS >= {ESS_FLOOR} on "
              f"{st['percent_above']:.2f}% of the grid ({st['n_above']}/{st['n']})"
              + (f", {missing} points not computed" if missing else ""))
    fig.colorbar(im, ax=axes[0].tolist(), label="ESS", fraction=0.05, pad=0.02)
    out = args.out or figure_path("fig4_ess_map.png")
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
