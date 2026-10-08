"""Metadynamics phase boundaries of Damewood et al. (2022) on the same CuAu-128 and AgPd-125 cells.

Read as in their notebooks: per temperature, every convex-hull edge of F(x) skipping more than
`gap_steps` compositions. This read applies no error criterion.
"""
from __future__ import annotations

import os

import numpy as np

from frogs.paths import REFERENCE_DIR

SYSTEMS = {
    "CuAu": dict(species="Au", file="CuAu_128_ref_meta_data_final_submission.npy",
                 T_range=(200.0, 1200.0), gap_steps=7),
    "AgPd": dict(species="Ag", file="AgPd_125_ref_meta_data_final_submission.npy",
                 T_range=(200.0, 900.0), gap_steps=8),
}


def system_from_meta(meta):
    """'CuAu', 'AgPd' or None, from the system / checkpoint / label recorded in a sweep meta."""
    text = " ".join(str(meta.get(k, "")) for k in ("system", "ckpt", "config", "run_label"))
    text = text.lower()
    for name in SYSTEMS:
        if name.lower() in text:
            return name
    return None


def metadynamics_boundary(system):
    """dict(T_edges, x_edges): both ends of every boundary edge, or None if the data are absent."""
    from scipy.spatial import ConvexHull

    spec = SYSTEMS[system]
    path = os.path.join(REFERENCE_DIR, spec["file"])
    if not os.path.exists(path):
        return None
    F = np.load(path)                                  # (n_T, n_x) free energy per T row
    temps = np.linspace(*spec["T_range"], F.shape[0])
    concs = np.linspace(0.0, 1.0, F.shape[1])
    cutoff = concs[1] * spec["gap_steps"]
    T_edges, x_edges = [], []
    for T, row in zip(temps, F):
        xs = np.sort(concs[ConvexHull(np.column_stack([concs, row])).vertices])
        for a, b in zip(xs[:-1], xs[1:]):
            if b - a > cutoff:
                T_edges += [T, T]
                x_edges += [a, b]
    return dict(T_edges=np.array(T_edges), x_edges=np.array(x_edges))


def boundary_lines(boundary):
    """Link boundary edges on adjacent rows into (x, T) polylines, one outline per region.

    Edges whose x-intervals overlap on adjacent rows belong to one region. When an edge overlaps
    several open regions (they merge as the phase between them dissolves), the merged region
    continues the leftmost region's left side and the rightmost region's right side.
    """
    Te = np.asarray(boundary["T_edges"], float)
    xe = np.asarray(boundary["x_edges"], float)
    rows = {}
    for k in range(0, Te.size - 1, 2):
        rows.setdefault(float(Te[k]), []).append((float(xe[k]), float(xe[k + 1])))

    fams = []
    for T in sorted(rows):
        open_f = [f for f in fams if f["open"]]
        taken, born = set(), []
        for a, b in sorted(rows[T]):
            hits = [f for f in open_f if id(f) not in taken
                    and min(b, f["hi"][-1]) - max(a, f["lo"][-1]) > 0]
            taken.update(id(f) for f in hits)
            if len(hits) == 1:
                f = hits[0]
                f["T"].append(T)
                f["lo"].append(a)
                f["hi"].append(b)
            elif len(hits) >= 2:
                hits.sort(key=lambda f: f["lo"][-1])
                for f in hits:
                    f["open"] = False
                    f["merged"] = True
                hits[0]["lo_continued"] = True
                hits[-1]["hi_continued"] = True
                born.append(dict(T=[T], lo=[a], hi=[b], open=True,
                                 lo_parent=hits[0], hi_parent=hits[-1]))
            else:
                born.append(dict(T=[T], lo=[a], hi=[b], open=True))
        for f in open_f:
            if id(f) not in taken:
                f["open"] = False
        fams.extend(born)

    def chain(f, side):
        X, T = [], []
        while f is not None:
            X = list(f[side]) + X
            T = list(f["T"]) + T
            f = f.get(side + "_parent")
        return X, T

    lines = []
    for f in fams:
        if f.get("merged"):
            for side in ("lo", "hi"):
                if not f.get(side + "_continued"):
                    lines.append(chain(f, side))
        else:
            xl, Tl = chain(f, "lo")
            xr, Tr = chain(f, "hi")
            lines.append((xl + xr[::-1], Tl + Tr[::-1]))
    return lines
