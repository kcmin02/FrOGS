"""Degenerate ordering variants of the CuAu phases on the 8x4x4 cell (Table 1, Fig. 7).

L1_0 CuAu has g = 6 variants (3 cubic orientations x 2 anti-phase domains); L1_2 Cu3Au and
CuAu3 have g = 4 (the minority species on one of the 4 simple-cubic sublattices).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.stats import chi2 as chi2_dist

from frogs.analysis.sweep_io import unpack_sigma

LATTICE = (8, 4, 4)
N_SITES = int(np.prod(LATTICE))
L10_AXES = ((0, 1), (0, 2), (1, 2))
COMP_TOL = 2          # |N_Au - ideal| <= COMP_TOL sites
ORDER_FRAC = 0.5      # order parameter >= ORDER_FRAC x its perfectly ordered value
MIN_GATED = 50        # gated samples for a sweep point to count
# Order parameters of the perfect configurations: S(k*)/N = N at the L1_0 variant's own k*;
# an L1_2 variant has S/N = N/4 at each of the 3 k*, summed.
S_PERFECT_L10 = float(N_SITES)
S_PERFECT_L12 = 0.75 * N_SITES
SWEEP_KEYS = ("dmu", "T", "A_samples", "N_samples", "sigma_packed")


@dataclass(frozen=True)
class Phase:
    name: str
    label: str
    g: int
    n_au: int            # ideal number of Au (+1) sites
    variants: tuple      # variant labels, in variant-index order


PHASES = (
    Phase("Cu3Au", r"Cu$_3$Au", 4, N_SITES // 4,
          tuple(f"Au@SL$_{s}$" for s in range(4))),
    Phase("CuAu", "CuAu", 6, N_SITES // 2,
          tuple(f"$v_{v}^{a}$" for v in range(3) for a in "+-")),
    Phase("CuAu3", r"CuAu$_3$", 4, 3 * N_SITES // 4,
          tuple(f"Cu@SL$_{s}$" for s in range(4))),
)


@dataclass
class VariantStats:
    """Per-sweep-point variant histograms of one phase on the (n_mu, n_T) grid."""
    counts: np.ndarray   # (n_mu, n_T, g) samples per variant
    wsum: np.ndarray     # (n_mu, n_T, g) importance weight per variant, weights normalised over all M
    w2: np.ndarray       # (n_mu, n_T) sum of squared weights of the classified samples
    n_gated: np.ndarray  # (n_mu, n_T) samples passing the composition and order gate


def _grid():
    return np.meshgrid(*[np.arange(s) for s in LATTICE], indexing="ij")


def perfect_l10(orientation: int) -> np.ndarray:
    """Perfect L1_0 spins (+1 = Au) of the given orientation (0, 1, 2), one anti-phase domain."""
    g = _grid()
    a, b = L10_AXES[orientation]
    return (1.0 - 2.0 * ((g[a] + g[b]) % 2)).reshape(-1)


def sublattice_index() -> np.ndarray:
    """Simple-cubic sublattice (0..3) of every site, from its pair parities."""
    g = _grid()
    code = ((g[1] + g[2]) % 2 * 4 + (g[0] + g[2]) % 2 * 2 + (g[0] + g[1]) % 2).reshape(-1)
    return np.searchsorted(np.unique(code), code)


def perfect_l12(sublattice: int, minority: int = 1) -> np.ndarray:
    """Perfect L1_2 spins with the minority species (+1 = Au: Cu3Au, -1 = Cu: CuAu3) on one sublattice."""
    m = (sublattice_index() == sublattice).astype(float)
    return 2 * m - 1 if minority > 0 else 1 - 2 * m


def _sign_code(re_f: np.ndarray) -> np.ndarray:
    sg = np.sign(re_f).astype(int)
    return (sg[..., 0] + 1) * 9 + (sg[..., 1] + 1) * 3 + (sg[..., 2] + 1)


@lru_cache(maxsize=None)
def calibration():
    """X-point plane waves (N, 3) and the L1_2 sign-pattern lookup tables, from perfect configurations."""
    g = _grid()
    kstars = []
    for o in range(3):
        s = perfect_l10(o).reshape(LATTICE)
        S = np.abs(np.fft.fftn(s - s.mean())) ** 2 / N_SITES
        flat = S.reshape(-1).copy()
        flat[0] = -1
        kstars.append(np.unravel_index(int(flat.argmax()), LATTICE))
    pw = np.zeros((3, N_SITES), dtype=np.complex128)
    for j, k in enumerate(kstars):
        ph = 2 * np.pi * sum(k[a] * g[a] / LATTICE[a] for a in range(3))
        pw[j] = np.exp(-1j * ph).reshape(-1)
    pw = pw.T
    luts = {}
    for name, minority in (("Cu3Au", 1), ("CuAu3", -1)):
        lut = np.full(27, -1, dtype=int)
        for s in range(4):
            lut[_sign_code(np.real(perfect_l12(s, minority) @ pw))] = s
        assert (lut >= 0).sum() == 4
        luts[name] = lut
    return pw, luts


def structure_factors(spins: np.ndarray) -> np.ndarray:
    """Fourier amplitudes F(k*) at the three X points, shape (..., 3)."""
    return np.asarray(spins, dtype=np.float64) @ calibration()[0]


def _assign(F: np.ndarray, n_au: np.ndarray, phase: Phase):
    """Gate mask and variant index (-1 where unassigned) for amplitudes F (n, 3)."""
    S = np.abs(F) ** 2 / N_SITES
    re_f = np.real(F)
    variant = np.full(len(F), -1, dtype=int)
    locked = np.abs(n_au - phase.n_au) <= COMP_TOL
    if phase.g == 6:
        gated = locked & (S.max(1) >= ORDER_FRAC * S_PERFECT_L10)
        v = S[gated].argmax(1)
        anti = np.take_along_axis(re_f[gated], v[:, None], 1)[:, 0] > 0
        variant[gated] = 2 * v + anti.astype(int)
    else:
        gated = locked & (S.sum(1) >= ORDER_FRAC * S_PERFECT_L12)
        variant[gated] = calibration()[1][phase.name][_sign_code(re_f[gated])]
    return gated, variant


def classify(spins: np.ndarray) -> dict:
    """Map phase name -> (gated mask, variant index or -1) for spin configurations (n, N)."""
    spins = np.atleast_2d(spins)
    F = structure_factors(spins)
    n_au = (spins > 0).sum(-1)
    return {ph.name: _assign(F, n_au, ph) for ph in PHASES}


def variant_histograms(sweep: dict, verbose: bool = False) -> dict:
    """Phase name -> VariantStats for a CuAu sweep with `sigma_packed` (see `SWEEP_KEYS`)."""
    A, n_au, packed = sweep["A_samples"], sweep["N_samples"], sweep["sigma_packed"]
    if int(sweep["meta"].get("N", N_SITES)) != N_SITES or packed.shape[-1] != -(-N_SITES // 8):
        raise ValueError(f"variant classification needs the {LATTICE} cell ({N_SITES} sites)")
    n_mu, n_T, M = A.shape
    pw = calibration()[0]
    out = {ph.name: VariantStats(np.zeros((n_mu, n_T, ph.g), np.int64),
                                 np.zeros((n_mu, n_T, ph.g)),
                                 np.zeros((n_mu, n_T)),
                                 np.zeros((n_mu, n_T), np.int64)) for ph in PHASES}
    for i in range(n_mu):
        sig = unpack_sigma(packed[i].reshape(n_T * M, -1), N_SITES).astype(np.float64)
        F = (sig @ pw).reshape(n_T, M, 3)
        for j in range(n_T):
            a = A[i, j]
            if not np.isfinite(a).all():
                continue
            w = np.exp(a - a.max())
            w /= w.sum()
            for ph in PHASES:
                gated, variant = _assign(F[j], n_au[i, j], ph)
                st = out[ph.name]
                st.n_gated[i, j] = gated.sum()
                ok = variant >= 0
                st.counts[i, j] = np.bincount(variant[ok], minlength=ph.g)
                st.wsum[i, j] = np.bincount(variant[ok], weights=w[ok], minlength=ph.g)
                st.w2[i, j] = (w[ok] ** 2).sum()
        if verbose:
            print(f"  classified dmu slice {i + 1}/{n_mu}", flush=True)
    return out


def chi2_per_dof(counts: np.ndarray) -> np.ndarray:
    """Pearson chi^2/dof of histograms (last axis) against the flat 1/g expectation; NaN if empty."""
    g = counts.shape[-1]
    n = counts.sum(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        exp = n[..., None] / g
        return np.where(n[..., None] > 0, (counts - exp) ** 2 / exp, np.nan).sum(-1) / (g - 1)


def weighted_chi2_per_dof(wsum: np.ndarray, w2: np.ndarray):
    """Importance-weighted chi^2/dof and the Kish size n_eff = (sum w)^2 / sum w^2 it uses."""
    g = wsum.shape[-1]
    tot = wsum.sum(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        n_eff = np.where(w2 > 0, tot ** 2 / w2, np.nan)
        p = wsum / np.where(tot > 0, tot, np.nan)[..., None]
        chi2 = (n_eff[..., None] * (p - 1 / g) ** 2 / (1 / g)).sum(-1) / (g - 1)
    return chi2, n_eff


def null_median(g: int) -> float:
    """Median of chi^2_{g-1}/(g-1): what a perfectly balanced sampler gives."""
    return float(chi2_dist.ppf(0.5, g - 1) / (g - 1))


def chi2_table(stats: dict, min_gated: int = MIN_GATED) -> dict:
    """Table 1 entries per phase: medians over the sweep points with >= min_gated classified samples."""
    rows = {}
    for ph in PHASES:
        st = stats[ph.name]
        gated = st.counts.sum(-1) >= min_gated
        raw = chi2_per_dof(st.counts)[gated]
        wtd, n_eff = weighted_chi2_per_dof(st.wsum, st.w2)
        any_ = bool(gated.any())
        rows[ph.name] = dict(
            unweighted=float(np.median(raw)) if any_ else np.nan,
            weighted=float(np.median(wtd[gated])) if any_ else np.nan,
            null=null_median(ph.g),
            points=int(gated.sum()),
            n_eff=float(np.median(n_eff[gated])) if any_ else np.nan,
            min_occupied=int((st.counts[gated] > 0).sum(-1).min()) if any_ else 0)
    return rows


def deepest_ordered_points(stats: dict, T: np.ndarray) -> dict:
    """Phase name -> (i_mu, j_T) of the sweep point with the most gated samples, ties to colder T."""
    return {name: np.unravel_index(int(np.argmax(st.n_gated - 1e-3 * T[None, :])),
                                   st.n_gated.shape)
            for name, st in stats.items()}
