"""Phase-diagram construction of §A.4.

Fixed-composition free energies f(x, T) from the importance weights (Eqs. 29-32), lower convex
hull of f(x) per temperature, and the width / depth / persistence gates that certify tie-lines.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from frogs.analysis.sweep_io import load_sweep
from frogs.constants import K_B

W_MIN = 4          # minimum tie-line width, lattice steps
N_SIGMA = 3        # minimum chord depth, standard errors
W_FLOOR = 5.0      # minimum pooled effective sample count of a bin entering the chi2 median
PERSIST = "below"
PERSIST_RULES = ("off", "below", "below_gap1")
SWEEP_KEYS = ("dmu", "T", "A_samples", "N_samples")


class Gap(NamedTuple):
    """A lower-hull facet [x_lo, x_hi] of f(x) and the largest excess of f above its chord."""
    x_lo: float
    x_hi: float
    depth: float     # eV/site; inf when no condition sampled the interior
    se: float
    n_missing: int   # interior compositions never sampled


def fixed_composition_free_energy(A, N, dmu, T, n_sites, w_floor=W_FLOOR):
    """f(x) [eV/site] at one temperature, merged over the dmu conditions (Eqs. 29-32).

    A, N: (n_mu, M) log-weights and species-1 counts. Returns (f, f_se, inflation) where f, f_se
    have length n_sites + 1 (NaN where never sampled) and f_se includes the Birge inflation.
    """
    M = A.shape[-1]
    n_mu = len(dmu)
    est = np.full((n_mu, n_sites + 1), np.nan)
    ses = np.full((n_mu, n_sites + 1), np.nan)
    beta = 1.0 / (K_B * T)
    for i in range(n_mu):
        if not np.isfinite(N[i]).all():   # condition not computed (partial sweep)
            continue
        n = N[i].astype(int)
        amax = A[i].max()
        w = np.exp(A[i] - amax)
        s1 = np.bincount(n, weights=w, minlength=n_sites + 1)
        s2 = np.bincount(n, weights=w * w, minlength=n_sites + 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ln_z = amax + np.log(s1) - np.log(M)
            se = np.sqrt(s2) / s1
        good = np.isfinite(ln_z)
        est[i, good] = (ln_z - beta * dmu[i] * np.arange(n_sites + 1))[good]
        ses[i, good] = se[good]

    # A bin whose w**2 all underflow gets se = 0, infinite precision, and reads as unsampled.
    with np.errstate(divide="ignore", invalid="ignore"):
        wiv = np.where(np.isfinite(est) & np.isfinite(ses), 1.0 / ses ** 2, 0.0)
        est0 = np.nan_to_num(est)
        w_tot = wiv.sum(0)
        mean = (wiv * est0).sum(0) / w_tot
        se = 1.0 / np.sqrt(w_tot)
        k = (wiv > 0).sum(0)
        chi2 = np.where(k > 1, (wiv * (est0 - mean[None, :]) ** 2).sum(0)
                        / np.maximum(k - 1, 1), np.nan)
        use = (k > 1) & (w_tot >= w_floor)
    if not use.any():
        use = k > 1
    inflation = float(np.sqrt(max(1.0, np.nanmedian(chi2[use])))) if use.any() else 1.0
    kT = K_B * T
    f = np.where(w_tot > 0, -kT * mean / n_sites, np.nan)
    f_se = np.where(w_tot > 0, kT * se / n_sites * inflation, np.nan)
    return f, f_se, inflation


def lower_hull(x, y):
    """Indices of the lower convex hull of (x, y), x non-decreasing (Andrew's monotone chain)."""
    hull = []
    for i in range(len(x)):
        while len(hull) >= 2:
            ox, oy = x[hull[-2]], y[hull[-2]]
            ax, ay = x[hull[-1]], y[hull[-1]]
            if (ax - ox) * (y[i] - oy) - (ay - oy) * (x[i] - ox) <= 0:
                hull.pop()
            else:
                break
        hull.append(i)
    return hull


def hull_gaps(f, f_se, n_sites):
    """Lower-hull facets of f(x) spanning more than one composition step, as `Gap`s."""
    x = np.arange(n_sites + 1) / n_sites
    m = np.isfinite(f)
    xm, fm = x[m], f[m]
    hull = np.column_stack([xm, fm])[lower_hull(xm, fm)]
    gaps = []
    for k in range(1, len(hull)):
        a, b = hull[k - 1, 0], hull[k, 0]
        if b - a <= 1.5 / n_sites:
            continue
        ia, ib = int(round(a * n_sites)), int(round(b * n_sites))
        xs = np.arange(ia, ib + 1) / n_sites
        chord = f[ia] + (f[ib] - f[ia]) * (xs - a) / (b - a)
        inner = (f[ia:ib + 1] - chord)[1:-1]
        n_missing = int(np.isnan(inner).sum())
        if np.all(np.isnan(inner)):
            gaps.append(Gap(a, b, np.inf, 0.0, n_missing))
            continue
        k2 = int(np.nanargmax(inner)) + 1
        lam = (xs[k2] - a) / (b - a)
        se = np.sqrt(f_se[ia + k2] ** 2 + ((1 - lam) * f_se[ia]) ** 2
                     + (lam * f_se[ib]) ** 2)
        gaps.append(Gap(a, b, float(inner[k2 - 1]), float(se), n_missing))
    return gaps


def gate_gaps(gaps, n_sites, w_min=W_MIN, n_sigma=N_SIGMA, measured_absence=True):
    """Width and depth gates (i)-(ii) of §A.4. Returns (candidates, n_suppressed).

    An unsampled interior (infinite depth) passes as measured absence, or is counted in
    n_suppressed when `measured_absence` is off, as it must be for a partial sweep.
    """
    cand, n_suppressed = [], 0
    for gap in gaps:
        a, b, depth, se, _ = gap
        if (b - a) * n_sites < w_min:
            continue
        if np.isinf(depth):
            if measured_absence:
                cand.append(gap)
            else:
                n_suppressed += 1
            continue
        if depth > n_sigma * se:
            cand.append(gap)
    return cand, n_suppressed


def _overlaps(gap, others):
    return any(min(gap[1], h[1]) - max(gap[0], h[0]) > 0 for h in others)


def apply_persistence(candidates, certified, j, rule=PERSIST):
    """Gate (iii): keep row j's candidates that x-overlap a gap certified on a lower row.

    'below' looks at row j-1, 'below_gap1' at rows j-1 and j-2, 'off' keeps everything.
    The lowest row anchors the chain.
    """
    if rule not in PERSIST_RULES:
        raise ValueError(f"unknown persistence rule {rule!r}; expected one of {PERSIST_RULES}")
    if rule == "off" or j == 0:
        return list(candidates)
    below = list(certified.get(j - 1, []))
    if rule == "below_gap1":
        below += certified.get(j - 2, [])
    return [g for g in candidates if _overlaps(g, below)]


def certify(gaps_by_row, n_sites, w_min=W_MIN, n_sigma=N_SIGMA, persist=PERSIST,
            measured_absence=True):
    """Gate every temperature row from the bottom up. Returns ({row: [Gap]}, n_suppressed)."""
    rungs, n_suppressed = {}, 0
    for j, gaps in enumerate(gaps_by_row):
        cand, n_sup = gate_gaps(gaps, n_sites, w_min, n_sigma, measured_absence)
        n_suppressed += n_sup
        rungs[j] = apply_persistence(cand, rungs, j, persist)
    return rungs, n_suppressed


def find_lobes(rungs, T, persist=PERSIST):
    """Group certified rungs into two-phase regions linked by x-overlap across rows.

    `persist` must be the rule that certified the rungs, so a row skipped under 'below_gap1'
    does not split a region.
    """
    if persist not in PERSIST_RULES:
        raise ValueError(f"unknown persistence rule {persist!r}; expected one of {PERSIST_RULES}")
    back = (1,) if persist == "below" else (1, 2)
    keys = [(j, k) for j in sorted(rungs) for k in range(len(rungs[j]))]
    parent = {key: key for key in keys}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for j in sorted(rungs):
        for k, g in enumerate(rungs[j]):
            for d in back:
                for kk, h in enumerate(rungs.get(j - d, [])):
                    if min(g[1], h[1]) - max(g[0], h[0]) > 0:
                        ra, rb = find((j - d, kk)), find((j, k))
                        if ra != rb:
                            parent[rb] = ra

    groups = {}
    for key in keys:
        groups.setdefault(find(key), []).append(key)
    lobes = []
    for members in groups.values():
        js = [j for j, _ in members]
        gs = [rungs[j][k] for j, k in members]
        mids = [0.5 * (g[0] + g[1]) for g in gs]
        lobes.append(dict(
            n_rungs=len(members),
            T_bot=float(T[min(js)]), T_top=float(T[max(js)]),
            x_lo=float(min(g[0] for g in gs)), x_hi=float(max(g[1] for g in gs)),
            mid_bot=float(mids[int(np.argmin(js))]), mid_top=float(mids[int(np.argmax(js))]),
        ))
    lobes.sort(key=lambda d: d["mid_bot"])
    return lobes


def ordered_compounds(rungs, n_sites, drift=2, j_anchor=None):
    """Compositions where one tie-line ends and the next begins, chained upward (§A.13).

    A terminus counts if it appears on the anchor row (default: the lowest) and continues on
    every row above, moving at most `drift` lattice steps per row. Returns [(x, n_rows)].
    """
    term = {}
    for j, row in rungs.items():
        lo = {int(round(g[0] * n_sites)) for g in row}
        hi = {int(round(g[1] * n_sites)) for g in row}
        term[j] = {k for k in lo & hi if 2 < k < n_sites - 2}
    if not term:
        return []
    j0 = min(term) if j_anchor is None else j_anchor
    if j0 not in term:
        return []
    out = []
    for k0 in sorted(term[j0]):
        k, chain, j = k0, [k0], j0 + 1
        while True:
            cand = [q for q in term.get(j, ()) if abs(q - k) <= drift]
            if not cand:
                break
            k = min(cand, key=lambda q: abs(q - k))
            chain.append(k)
            j += 1
        out.append((float(np.mean(chain)) / n_sites, len(chain)))
    return out


def site_count(sweep):
    """Number of lattice sites, from the sweep meta; the sampled counts are only a cross-check."""
    n_max = int(round(float(np.nanmax(sweep["N_samples"]))))
    n_sites = int(sweep.get("meta", {}).get("N") or n_max)
    if n_max > n_sites:
        raise ValueError(f"N_samples reaches {n_max} > meta N = {n_sites}")
    return n_sites


def sweep_gaps(sweep, n_sites=None, w_floor=W_FLOOR):
    """Threshold-independent half of the construction: hull gaps and inflation per T row."""
    n_sites = site_count(sweep) if n_sites is None else n_sites
    A, N, dmu, T = sweep["A_samples"], sweep["N_samples"], sweep["dmu"], sweep["T"]
    gaps_by_row, inflation = [], np.zeros(len(T))
    for j in range(len(T)):
        f, f_se, inflation[j] = fixed_composition_free_energy(
            A[:, j], N[:, j], dmu, T[j], n_sites, w_floor)
        gaps_by_row.append(hull_gaps(f, f_se, n_sites))
    return gaps_by_row, inflation


@dataclass
class PhaseDiagram:
    """Certified tie-lines (`rungs`: T-row index -> [Gap]) and the two-phase regions they form."""
    T: np.ndarray
    n_sites: int
    rungs: dict
    lobes: list
    inflation: np.ndarray
    partial: bool = False
    n_suppressed: int = 0
    settings: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    @property
    def n_rungs(self):
        return sum(len(v) for v in self.rungs.values())

    def rung_table(self):
        """(n_rungs, 6) array of T, x_lo, x_hi, depth, se, n_missing."""
        rows = [(self.T[j],) + tuple(g) for j in sorted(self.rungs) for g in self.rungs[j]]
        return np.array(rows, dtype=float).reshape(-1, 6)

    def compounds(self, drift=2):
        return ordered_compounds(self.rungs, self.n_sites, drift)

    def save(self, path):
        r = self.rung_table()
        info = dict(n_sites=self.n_sites, partial=self.partial, n_suppressed=self.n_suppressed,
                    settings=self.settings, meta=self.meta)
        np.savez(path, T=np.asarray(self.T, float), inflation=self.inflation,
                 rung_T=r[:, 0], rung_x_lo=r[:, 1], rung_x_hi=r[:, 2],
                 rung_depth=r[:, 3], rung_se=r[:, 4], rung_n_missing=r[:, 5],
                 lobes_json=np.array(json.dumps(self.lobes), dtype=object),
                 meta_json=np.array(json.dumps(info), dtype=object))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=True) as z:
            T = z["T"]
            info = json.loads(str(z["meta_json"]))
            rungs = {j: [] for j in range(len(T))}
            for t, a, b, d, se, nm in zip(z["rung_T"], z["rung_x_lo"], z["rung_x_hi"],
                                          z["rung_depth"], z["rung_se"], z["rung_n_missing"]):
                rungs[int(np.argmin(np.abs(T - t)))].append(Gap(a, b, d, se, int(nm)))
            return cls(T=T, n_sites=info["n_sites"], rungs=rungs,
                       lobes=json.loads(str(z["lobes_json"])), inflation=z["inflation"],
                       partial=info["partial"], n_suppressed=info["n_suppressed"],
                       settings=info["settings"], meta=info["meta"])


def construct_phase_diagram(sweep, w_min=W_MIN, n_sigma=N_SIGMA, persist=PERSIST,
                            w_floor=W_FLOOR, partial=None):
    """Run the §A.4 constructor on a sweep (path or `load_sweep` dict).

    `partial=None` treats the sweep as partial if its meta says so or any condition is missing;
    a partial sweep certifies no measured-absence rungs.
    """
    if isinstance(sweep, (str, os.PathLike)):
        sweep = load_sweep(sweep, keys=SWEEP_KEYS)
    n_sites = site_count(sweep)
    if partial is None:
        partial = bool(sweep.get("meta", {}).get("partial", False)) \
            or not np.isfinite(sweep["N_samples"]).all()
    T = np.asarray(sweep["T"], float)
    gaps_by_row, inflation = sweep_gaps(sweep, n_sites, w_floor)
    rungs, n_suppressed = certify(gaps_by_row, n_sites, w_min, n_sigma, persist,
                                  measured_absence=not partial)
    settings = dict(w_min=w_min, n_sigma=n_sigma, persist=persist, w_floor=w_floor)
    return PhaseDiagram(T=T, n_sites=n_sites, rungs=rungs, lobes=find_lobes(rungs, T, persist),
                        inflation=inflation, partial=bool(partial), n_suppressed=n_suppressed,
                        settings=settings, meta=dict(sweep.get("meta", {})))
