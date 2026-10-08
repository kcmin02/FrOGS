"""Exact finite-L results for the zero-field 2D Ising torus (Kaufman 1949) and a Wolff reference.

Units: J = k_B = 1, coupling K = 1/T. T_C, K_C are Onsager's critical point.
"""
from __future__ import annotations

import math

import numpy as np

T_C = 2.0 / math.log(1.0 + math.sqrt(2.0))
K_C = 0.5 * math.log(1.0 + math.sqrt(2.0))


def _log_2cosh(x):
    x = np.asarray(x, dtype=np.float64)
    return np.where(x < 20.0, np.log(2.0 * np.cosh(np.minimum(x, 20.0))),
                    x + np.log1p(np.exp(-2.0 * np.maximum(x, 20.0))))


def _log_2sinh(x):
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(divide="ignore"):
        small = np.log(2.0 * np.sinh(np.minimum(x, 20.0)))
        large = x + np.log1p(-np.exp(-2.0 * np.maximum(x, 20.0)))
    return np.where(x < 20.0, small, large)


def kaufman_logZ(K: float, m: int, n: int) -> float:
    """Exact log Z of the m x n periodic square lattice at coupling K > 0, zero field."""
    K = float(K)
    if K <= 0.0:
        raise ValueError("Kaufman formula assumes K > 0")
    c2, s2 = math.cosh(2.0 * K), math.sinh(2.0 * K)
    l = np.arange(2 * n, dtype=np.float64)
    cosh_g = c2 * c2 / s2 - np.cos(np.pi * l / n)
    abs_g = np.arccosh(np.maximum(cosh_g, 1.0))
    lc = _log_2cosh(0.5 * m * abs_g)
    ls = _log_2sinh(0.5 * m * abs_g)

    logs = np.array([lc[1::2].sum(), ls[1::2].sum(), lc[0::2].sum(), ls[0::2].sum()])
    # Only gamma_0 = 2K + log tanh K changes sign (at K_c), so only S_even is signed.
    signs = np.array([1.0, 1.0, 1.0, math.copysign(1.0, 2.0 * K + math.log(math.tanh(K)))])
    finite = np.isfinite(logs)
    logs, signs = logs[finite], signs[finite]
    top = logs.max()
    total = float(np.sum(signs * np.exp(logs - top)))
    if total <= 0.0:
        raise FloatingPointError(f"Kaufman bracket non-positive at K={K} ({m}x{n})")
    log_bracket = top + math.log(total)
    return -math.log(2.0) + 0.5 * m * n * math.log(2.0 * s2) + log_bracket


def kaufman_nn_correlation(K: float, m: int, n: int, eps: float = 1e-5) -> float:
    """<sigma_0 sigma_1> = (d log Z / dK) / (2 m n), by a central difference."""
    dp = kaufman_logZ(K + eps, m, n)
    dm = kaufman_logZ(K - eps, m, n)
    return (dp - dm) / (2.0 * eps) / (2.0 * m * n)


def kaufman_thermo(K: float, L: int, eps: float = 1e-3):
    """(log Z, U/N, C/N) of the L x L torus from five-point K-derivatives of log Z."""
    N = L * L
    z0 = kaufman_logZ(K, L, L)
    zp1 = kaufman_logZ(K + eps, L, L)
    zm1 = kaufman_logZ(K - eps, L, L)
    zp2 = kaufman_logZ(K + 2.0 * eps, L, L)
    zm2 = kaufman_logZ(K - 2.0 * eps, L, L)
    d1 = (-zp2 + 8.0 * zp1 - 8.0 * zm1 + zm2) / (12.0 * eps)
    d2 = (-zp2 + 16.0 * zp1 - 30.0 * z0 + 16.0 * zm1 - zm2) / (12.0 * eps ** 2)
    return z0, -d1 / N, K * K * d2 / N


def wolff_gr(K: float, L: int, n_meas: int = 20000, n_therm: int = 2000,
             sweeps_per_meas: int = 3, seed: int = 0):
    """One Wolff single-cluster chain at h = 0: G(r) for r = 1..L//2 and m per measurement."""
    rng = np.random.default_rng(seed)
    s = np.where(rng.random((L, L)) < 0.5, -1, 1).astype(np.int8)
    p_add = 1.0 - math.exp(-2.0 * K)
    rmax = L // 2
    G_acc = np.zeros(rmax)
    m_samples = np.empty(n_meas)

    def one_cluster():
        i0, j0 = rng.integers(L), rng.integers(L)
        old = s[i0, j0]
        s[i0, j0] = -old
        stack = [(i0, j0)]
        while stack:
            i, j = stack.pop()
            for ni, nj in (((i - 1) % L, j), ((i + 1) % L, j), (i, (j - 1) % L), (i, (j + 1) % L)):
                if s[ni, nj] == old and rng.random() < p_add:
                    s[ni, nj] = -old
                    stack.append((ni, nj))

    for _ in range(n_therm):
        one_cluster()
    for k in range(n_meas):
        for _ in range(sweeps_per_meas):
            one_cluster()
        sf = s.astype(np.float64)
        m_samples[k] = sf.mean()
        for r in range(1, rmax + 1):
            G_acc[r - 1] += 0.5 * (float((sf * np.roll(sf, -r, axis=1)).mean())
                                   + float((sf * np.roll(sf, -r, axis=0)).mean()))
    return G_acc / n_meas, m_samples


def wolff_reference(K: float, L: int, n_chains: int = 4, n_meas: int = 20000,
                    n_therm: int = 5000, seed: int = 7):
    """Mean G(r) over Wolff chains with seeds seed, seed+1, ... and its SE across chains."""
    runs = np.array([wolff_gr(K, L, n_meas=n_meas, n_therm=n_therm, seed=seed + i)[0]
                     for i in range(n_chains)])
    return runs.mean(axis=0), runs.std(axis=0, ddof=1) / math.sqrt(n_chains)
