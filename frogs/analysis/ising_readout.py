"""Importance-weighted 2D-Ising observables at h = 0 from walker log-weights A (Fig. 2, Fig. 9).

Error bars resample walkers (bootstrap); log Z_0 = 0 for the autoregressive prior.
"""
from __future__ import annotations

import json
import warnings

import numpy as np

_BOOT_KEYS = ("logZ", "F_per", "U_per", "pos_mass")


def weights(A):
    a = A - A.max()
    w = np.exp(a)
    return w / w.sum()


def log_mean_exp(A):
    return float(A.max() + np.log(np.mean(np.exp(A - A.max()))))


def ess(A):
    """Normalized ESS (sum w)^2 / (M sum w^2)."""
    a = A - A.max()
    return float(np.mean(np.exp(a)) ** 2 / np.mean(np.exp(2 * a)))


def spin_observables(spins):
    """Per-walker magnetisation (M,) and G(r) = <sigma_0 sigma_r>, r = 1..L//2, (M, L//2)."""
    s = np.asarray(spins, dtype=np.float64)
    L = s.shape[-1]
    G = np.empty((s.shape[0], L // 2))
    for r in range(1, L // 2 + 1):
        gx = (s * np.roll(s, -r, axis=2)).mean(axis=(1, 2))
        gy = (s * np.roll(s, -r, axis=1)).mean(axis=(1, 2))
        G[:, r - 1] = 0.5 * (gx + gy)
    return s.mean(axis=(1, 2)), G


def _thermo(A, e, m, T, N):
    w = weights(A)
    logZ = log_mean_exp(A)
    return dict(logZ=logZ, F_per=-T * logZ / N, U_per=float(w @ e) / N, ess=ess(A),
                pos_mass=float(w[m > 0].sum()), pos_raw=float(np.mean(m > 0)))


def thermo_readout(A, e, m, T, N, n_boot=400, seed=0):
    """log Z, F/N, U/N, ESS and the weighted / raw mass on m > 0, with bootstrap SEs."""
    A = np.asarray(A, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    out = _thermo(A, e, m, T, N)
    rng = np.random.default_rng(seed)
    boot = {k: [] for k in _BOOT_KEYS}
    for _ in range(n_boot):
        i = rng.integers(0, A.size, A.size)
        b = _thermo(A[i], e[i], m[i], T, N)
        for k in _BOOT_KEYS:
            boot[k].append(b[k])
    for k in _BOOT_KEYS:
        out[k + "_se"] = float(np.std(boot[k], ddof=1))
    return out


def weighted_correlator(A, G_per, n_boot=500, seed=0):
    """IS-weighted G(r) and its bootstrap SE."""
    A = np.asarray(A, dtype=np.float64)
    G = weights(A) @ G_per
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        i = rng.integers(0, A.size, A.size)
        boot.append(weights(A[i]) @ G_per[i])
    return G, np.std(np.array(boot), axis=0, ddof=1)


def local_slope(x, y, sig, window=5):
    """dy/dx from a 1/sig^2-weighted quadratic through each point's `window` nearest neighbours.

    Points whose window is one-sided (the first and last window//2) are NaN.
    """
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    n = x.size
    half = window // 2
    rank = np.empty(n, dtype=int)
    rank[np.argsort(x)] = np.arange(n)
    out = np.full(n, np.nan)
    for i in range(n):
        if not half <= rank[i] < n - half:
            continue
        j = np.argsort(np.abs(x - x[i]))[:window]
        dx = x[j] - x[i]
        W = np.diag(1.0 / sig[j] ** 2)
        X = np.vstack([np.ones_like(dx), dx, dx ** 2]).T
        out[i] = np.linalg.solve(X.T @ W @ X, X.T @ W @ y[j])[1]
    return out


def heat_capacity(T, U_per, se=None, window=5, n_boot=500, seed=0):
    """C/N = d(U/N)/dT by `local_slope`; the SE resamples U/N within its error bars."""
    T = np.asarray(T, np.float64)
    sig = np.ones(T.size) if se is None else np.maximum(np.asarray(se, float), 1e-12)
    C = local_slope(T, U_per, sig, window)
    if se is None:
        return C, None
    rng = np.random.default_rng(seed)
    boot = np.array([local_slope(T, np.asarray(U_per) + rng.normal(0, sig), sig, window)
                     for _ in range(n_boot)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # the all-NaN edge columns
        return C, np.nanstd(boot, axis=0, ddof=1)


def readout(A, spins, T, n_boot=400, n_boot_G=500, window=5, seed=0):
    """All Fig. 2 observables of one arm from A (n_T, M) and spins (n_T, M, L, L).

    Temperature j is bootstrapped with seed + j.
    """
    T = np.asarray(T, np.float64)
    L = spins.shape[-1]
    N = L * L
    rows, G, G_se = [], [], []
    for j, t in enumerate(T):
        m, G_per = spin_observables(spins[j])
        e = -2.0 * N * G_per[:, 0]  # H at h = 0, J = 1: two bonds per site
        rows.append(thermo_readout(A[j], e, m, t, N, n_boot=n_boot, seed=seed + j))
        g, g_se = weighted_correlator(A[j], G_per, n_boot=n_boot_G, seed=seed + j)
        G.append(g)
        G_se.append(g_se)
    out = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    out["G"], out["G_se"] = np.array(G), np.array(G_se)
    out["C"], out["C_se"] = heat_capacity(T, out["U_per"], out["U_per_se"], window=window,
                                          seed=seed)
    out["A"] = np.asarray(A, np.float64)
    return out


def save_results(path, common, arms, meta):
    """One npz: shared arrays, per-arm arrays prefixed '<arm>_', and a JSON `meta`."""
    payload = dict(common)
    for name, d in arms.items():
        payload.update({f"{name}_{k}": v for k, v in d.items()})
    meta = dict(meta, arms=list(arms))
    payload["meta_json"] = np.array(json.dumps(meta))
    np.savez(path, **payload)


def load_results(path):
    """Inverse of `save_results`: (common, arms, meta)."""
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta_json"]))
        arms = {name: {} for name in meta["arms"]}
        common = {}
        for key in z.files:
            if key == "meta_json":
                continue
            name = next((a for a in arms if key.startswith(a + "_")), None)
            if name is None:
                common[key] = z[key]
            else:
                arms[name][key[len(name) + 1:]] = z[key]
    return common, arms, meta
