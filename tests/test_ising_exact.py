import math

import numpy as np
import pytest

from frogs.analysis.ising_exact import (K_C, T_C, kaufman_logZ, kaufman_nn_correlation,
                                        kaufman_thermo, wolff_gr, wolff_reference)
from frogs.analysis.ising_readout import (ess, heat_capacity, load_results, log_mean_exp, readout,
                                          save_results, spin_observables, thermo_readout,
                                          weighted_correlator)


def all_states(L):
    n = L * L
    idx = np.arange(1 << n, dtype=np.uint32)
    bits = (idx[:, None] >> np.arange(n, dtype=np.uint32)[None, :]) & 1
    return (bits.astype(np.float64) * 2 - 1).reshape(-1, L, L)


def bond_sum(s):
    return ((s * np.roll(s, -1, axis=2)).sum(axis=(1, 2))
            + (s * np.roll(s, -1, axis=1)).sum(axis=(1, 2)))


def enumerate_logZ_and_G(K, L):
    s = all_states(L)
    logw = K * bond_sum(s)
    top = logw.max()
    w = np.exp(logw - top)
    p = w / w.sum()
    _, G_per = spin_observables(s)
    return top + math.log(w.sum()), p @ G_per


def test_onsager_constants():
    assert T_C == pytest.approx(2.269185314213022, abs=1e-15)
    assert K_C * T_C == pytest.approx(1.0, abs=1e-15)


@pytest.mark.parametrize("L", [3, 4])
@pytest.mark.parametrize("K", [0.1, 0.25, K_C, 0.6, 1.0])
def test_kaufman_matches_enumeration(L, K):
    logZ, G = enumerate_logZ_and_G(K, L)
    assert kaufman_logZ(K, L, L) == pytest.approx(logZ, rel=1e-9)
    assert kaufman_nn_correlation(K, L, L) == pytest.approx(G[0], abs=1e-6)


@pytest.mark.parametrize("T", [1.8, T_C, 3.0])
def test_kaufman_thermo_matches_enumeration(T):
    L = 4
    logZ, G = enumerate_logZ_and_G(1.0 / T, L)
    z, U, C = kaufman_thermo(1.0 / T, L)
    assert z == pytest.approx(logZ, rel=1e-9)
    assert U == pytest.approx(-2.0 * G[0], abs=1e-5)
    s = all_states(L)
    H = -bond_sum(s)
    p = np.exp(-(H - H.min()) / T)
    p /= p.sum()
    assert C == pytest.approx((p @ H ** 2 - (p @ H) ** 2) / (T * T * L * L), rel=1e-5)


def test_wolff_matches_enumeration():
    _, G_exact = enumerate_logZ_and_G(K_C, 4)
    G, G_se = wolff_reference(K_C, 4, n_chains=4, n_meas=5000, n_therm=500, seed=1)
    assert G.shape == G_se.shape == (2,)
    assert np.all(G_se > 0)
    assert np.all(np.abs(G - G_exact) < 5 * G_se + 1e-3)
    G1, m = wolff_gr(K_C, 4, n_meas=5000, n_therm=500, seed=1)
    np.testing.assert_array_equal(G1, wolff_gr(K_C, 4, n_meas=5000, n_therm=500, seed=1)[0])
    assert m.shape == (5000,) and abs(m.mean()) < 0.1


def test_wolff_nn_correlation_at_criticality():
    G, G_se = wolff_reference(K_C, 10, n_chains=3, n_meas=3000, n_therm=300, seed=2)
    assert abs(G[0] - kaufman_nn_correlation(K_C, 10, 10)) < 5 * G_se[0] + 2e-3
    assert np.all(np.diff(G) < 0)


def test_spin_observables_give_the_energy():
    rng = np.random.default_rng(0)
    L = 6
    s = np.where(rng.random((5, L, L)) < 0.5, -1.0, 1.0)
    m, G = spin_observables(s)
    H = -bond_sum(s)
    np.testing.assert_allclose(-2.0 * L * L * G[:, 0], H, atol=1e-9)
    np.testing.assert_allclose(m, s.mean(axis=(1, 2)))
    assert G.shape == (5, L // 2)


def test_readout_on_uniform_and_degenerate_weights():
    rng = np.random.default_rng(1)
    M, N, T = 400, 16, 2.0
    e = rng.normal(-20.0, 3.0, M)
    m = rng.uniform(-1.0, 1.0, M)
    flat = thermo_readout(np.full(M, -3.0), e, m, T, N, n_boot=50)
    assert flat["ess"] == pytest.approx(1.0)
    assert flat["logZ"] == pytest.approx(-3.0)
    assert flat["F_per"] == pytest.approx(3.0 * T / N)
    assert flat["U_per"] == pytest.approx(e.mean() / N)
    assert flat["pos_mass"] == pytest.approx(flat["pos_raw"])
    assert flat["logZ_se"] == pytest.approx(0.0, abs=1e-12)

    A = np.zeros(M)
    A[7] = 50.0
    peaked = thermo_readout(A, e, m, T, N, n_boot=50)
    assert peaked["ess"] == pytest.approx(1.0 / M, rel=1e-6)
    assert peaked["U_per"] == pytest.approx(e[7] / N, rel=1e-6)
    assert peaked["pos_mass"] == pytest.approx(float(m[7] > 0), abs=1e-6)

    A = rng.normal(0.0, 1.5, M)
    assert log_mean_exp(A) == pytest.approx(np.log(np.mean(np.exp(A))))
    w = np.exp(A) / np.exp(A).sum()
    assert ess(A) == pytest.approx(1.0 / (M * (w ** 2).sum()))
    G_per = rng.random((M, 3))
    G, G_se = weighted_correlator(A, G_per, n_boot=50)
    np.testing.assert_allclose(G, w @ G_per)
    assert np.all(G_se > 0)


def test_heat_capacity_is_exact_on_a_quadratic():
    T = np.linspace(1.5, 3.0, 9)
    C, se = heat_capacity(T, 0.7 * T ** 2 - T)
    assert se is None
    assert np.all(np.isnan(C[:2])) and np.all(np.isnan(C[-2:]))
    np.testing.assert_allclose(C[2:-2], 1.4 * T[2:-2] - 1.0, atol=1e-10)


def test_readout_shapes():
    rng = np.random.default_rng(2)
    T = np.array([1.8, 2.2, T_C, 2.5, 2.9])
    spins = np.where(rng.random((5, 30, 4, 4)) < 0.5, -1, 1).astype(np.int8)
    out = readout(rng.normal(size=(5, 30)), spins, T, n_boot=20, n_boot_G=20)
    assert out["G"].shape == (5, 2) and out["ess"].shape == (5,)
    assert np.isfinite(out["C"][2]) and np.isnan(out["C"][0])


def test_results_roundtrip(tmp_path):
    common = dict(L=4, T=np.array([2.0, 2.5]), r=np.arange(1, 3))
    arms = {"transport": dict(ess=np.array([0.5, 0.6])), "prior": dict(ess=np.array([0.1, 0.2]))}
    path = str(tmp_path / "ising.npz")
    save_results(path, common, arms, dict(M=30))
    c, a, meta = load_results(path)
    assert meta["M"] == 30 and meta["arms"] == ["transport", "prior"]
    np.testing.assert_array_equal(c["T"], common["T"])
    np.testing.assert_array_equal(a["prior"]["ess"], arms["prior"]["ess"])
