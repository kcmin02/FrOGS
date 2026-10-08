"""Drawing shared by the binodal figures (Figs. 3, 10) and the ESS maps (Figs. 4, 6)."""
from __future__ import annotations

import matplotlib
import numpy as np
from matplotlib.colors import Normalize

from frogs.analysis.reference import SYSTEMS, boundary_lines, metadynamics_boundary

ESS_FLOOR = 0.1


def draw_binodal(ax, diagram, system=None, label="FrOGS", t_max=None,
                 reference_label="metadynamics", fontsize=11.0, legend=True):
    """Certified tie-lines of a `PhaseDiagram` over the metadynamics boundary of `system`."""
    T = diagram.T
    boundary = metadynamics_boundary(system) if system in SYSTEMS else None
    if boundary is not None:
        for k, (x, t) in enumerate(boundary_lines(boundary)):
            ax.plot(x, t, ls="--", lw=1.5, color="darkgoldenrod", alpha=0.85, zorder=3,
                    label=reference_label if k == 0 else None)
    first = True
    for j, gaps in diagram.rungs.items():
        for a, b, *_ in gaps:
            ax.plot([a, b], [T[j], T[j]], color="0.8", lw=1.2, zorder=5)
            ax.plot([a, b], [T[j], T[j]], "o", ms=4.5, color="tab:blue", lw=0, zorder=6,
                    label=label if first else None)
            first = False

    dT = float(np.median(np.diff(T))) if len(T) > 1 else 25.0
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(float(T.min()) - dT / 2, t_max if t_max is not None else float(T.max()) + dT / 2)
    species = SYSTEMS[system]["species"] if system in SYSTEMS else "B"
    ax.set_xlabel(rf"$x_\mathrm{{{species}}}$", fontsize=fontsize)
    ax.set_ylabel("T (K)", fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=0.9 * fontsize)
    ax.grid(True, alpha=0.3)
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper right", fontsize=0.7 * fontsize, framealpha=0.93)


def draw_ess_map(ax, dmu, T, ess, cmap="inferno"):
    """ESS over the (dmu, T) grid as a [0, 1] heatmap; uncomputed points are grey."""
    dmu, T = np.asarray(dmu, float), np.asarray(T, float)
    ddmu = dmu[1] - dmu[0] if len(dmu) > 1 else 0.02
    dT = T[1] - T[0] if len(T) > 1 else 25.0
    extent = [dmu[0] - ddmu / 2, dmu[-1] + ddmu / 2, T[0] - dT / 2, T[-1] + dT / 2]
    cm = matplotlib.colormaps[cmap].copy()
    cm.set_bad("0.82")
    im = ax.imshow(np.asarray(ess, float).T, extent=extent, origin="lower", aspect="auto",
                   cmap=cm, norm=Normalize(0.0, 1.0), interpolation="nearest")
    ax.set_xlabel(r"$\Delta\mu$ (eV)")
    ax.set_ylabel("T (K)")
    return im


def ess_summary(ess, floor=ESS_FLOOR):
    """Median ESS and the percentage of computed grid points with ESS >= floor."""
    ess = np.asarray(ess, float)
    fin = np.isfinite(ess)
    n_ok, n = int((ess[fin] >= floor).sum()), int(fin.sum())
    return dict(median=float(np.median(ess[fin])) if n else float("nan"),
                percent_above=100.0 * n_ok / n if n else float("nan"),
                n_above=n_ok, n=n, n_total=int(ess.size))
