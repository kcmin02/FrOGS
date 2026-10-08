"""Identities of the energies and the annealing path (Eq. 11, Eq. 33), checked by brute force."""
import itertools
import math

import numpy as np
import pytest
import torch

from frogs.constants import K_B
from frogs.energy import AlloyEnergy, ClusterExpansion, IsingEnergy
from frogs.nets import ARConvPrior
from frogs.paths import data_file
from frogs.sampler import ess

pytestmark = pytest.mark.usefixtures("float64")

CELL = (2, 2, 4)
TABLE = "cuau_cluster_tables_2x2x4.npz"
ALLOY_BOX = ([-0.05, 200.0], [0.05, 900.0])
ISING_BOX = ([-0.05, 1.5], [0.05, 3.0])


def tiny_prior(shape):
    torch.manual_seed(0)
    prior = ARConvPrior(shape, num_layers=2, num_channels=4)
    with torch.no_grad():  # FiLM starts as the identity; give the prior a real c-dependence
        for film in (prior.film_initial, *prior.films):
            film.net[-1].weight.normal_(0, 0.1)
            film.net[-1].bias.normal_(0, 0.1)
    return prior


def alloy(learned=True):
    prior = tiny_prior(CELL) if learned else None
    return AlloyEnergy(data_file(TABLE), CELL, prior=prior, cond_low=ALLOY_BOX[0],
                       cond_high=ALLOY_BOX[1])


def ising(L=3):
    return IsingEnergy(L, tiny_prior((1, L, L)), cond_low=ISING_BOX[0], cond_high=ISING_BOX[1])


def build(system):
    """(energy, lattice shape, one condition c = (dmu or h, T)) for a tiny learned-prior system."""
    if system == "ising":
        return ising(3), (3, 3), (0.02, 2.2)
    return alloy(), CELL, (0.02, 500.0)


def random_spins(n, shape):
    return torch.where(torch.rand(n, *shape) < 0.5, -1.0, 1.0)


def all_spins(shape):
    n = int(np.prod(shape))
    return torch.tensor(list(itertools.product([-1.0, 1.0], repeat=n))).reshape(-1, *shape)


def conditions(n, dmu, T):
    return torch.tensor([[dmu, T]]).expand(n, -1).contiguous()


def bare_alloy(e, sigma, c):
    s = sigma.reshape(sigma.shape[0], -1)
    n_plus = ((s + 1) / 2).sum(dim=1)
    return (e.ce.energy(s) - c[:, 0] * n_plus) / (K_B * c[:, 1])


def table_energy(path, shape, sigma):
    """U_CE straight from the npz: N * sum_c eci_c * mean over occurrences of prod sigma."""
    d = np.load(path)
    prim = d["prim_coords"]
    s = sigma[:, (prim[:, 0] * shape[1] + prim[:, 1]) * shape[2] + prim[:, 2]]
    total = np.zeros(len(sigma))
    for key, eci in zip(d["eci_keys"], d["eci_values"]):
        inds = d[f"cluster_{key}_indices"]
        total += eci * (np.prod(s[:, inds], axis=2).mean(axis=1) if inds.size else 1.0)
    return int(d["n_sites"]) * total


@pytest.mark.parametrize("name,shape", [
    (TABLE, CELL), ("cuau_cluster_tables_8x4x4.npz", (8, 4, 4)),
    ("agpd_cluster_tables_L5.npz", (5, 5, 5))])
def test_cluster_expansion_matches_the_table_and_explicit_flips(name, shape):
    ce = ClusterExpansion(data_file(name), shape)
    sigma = random_spins(4, (ce.n_sites,))
    sigma[0] = 1.0
    U = ce.energy(sigma)
    np.testing.assert_allclose(U.numpy(), table_energy(data_file(name), shape, sigma.numpy()),
                               rtol=0, atol=1e-12)
    assert torch.allclose(torch.stack([ce.energy_single(s) for s in sigma]), U, atol=1e-12)
    for site in (0, 7, ce.n_sites - 1):
        flipped = sigma.clone()
        flipped[:, site] *= -1
        direct = ce.energy(flipped) - U
        assert torch.allclose(ce.flip_delta_at(sigma, site), direct, atol=1e-12)
        singles = torch.stack([ce.flip_deltas_single(s)[site] for s in sigma])
        assert torch.allclose(singles, direct, atol=1e-12)


def test_ising_hamiltonian_matches_explicit_bond_sum():
    L, e = 4, ising(4)
    sigma, h = random_spins(6, (L, L)), torch.full((6,), 0.03)
    H = e.hamiltonian(sigma, h)
    for b in range(6):
        s = sigma[b]
        bonds = sum(float(s[i, j] * (s[(i + 1) % L, j] + s[i, (j + 1) % L]))
                    for i in range(L) for j in range(L))
        assert H[b].item() == pytest.approx(-bonds - 0.03 * float(s.sum()), abs=1e-12)


@pytest.mark.parametrize("system", ["ising", "alloy", "alloy-uniform"])
def test_flip_energies_match_explicit_flips(system):
    """Delta U_t feeds the rates, the corrector (Eq. 24) and the Metropolis moves."""
    if system == "alloy-uniform":
        e, shape, c = alloy(learned=False), CELL, (0.01, 600.0)
    else:
        e, shape, c = build(system)
    sigma, t, c = random_spins(4, shape), torch.full((4,), 0.6), conditions(4, *c)
    with torch.no_grad():
        U, dU = e.Ut(sigma, t, c)
        flat = sigma.reshape(4, -1)
        state = e.mcmc_state(flat, t, c)
        for site in range(flat.shape[1]):
            flipped = flat.clone()
            flipped[:, site] *= -1
            direct = e.Ut(flipped.reshape(sigma.shape), t, c)[0] - U
            assert torch.allclose(dU.reshape(4, -1)[:, site], direct, atol=1e-10)
            assert torch.allclose(e.mcmc_delta(flat, t, c, site, state)[0], direct, atol=1e-10)


@pytest.mark.parametrize("system", ["ising", "alloy"])
def test_learned_prior_normalizes_to_one(system):
    """Z_0 = 1 for any weights, which is why E[e^A] estimates Z_1 itself (Eq. 2)."""
    e, shape, c = build(system)
    sigma = all_spins(shape)
    B = sigma.shape[0]
    with torch.no_grad():
        U0, _ = e.Ut(sigma, torch.zeros(B), conditions(B, *c), need_dU=False)
    assert torch.exp(-U0).sum().item() == pytest.approx(1.0, abs=1e-10)
    assert e.log_Z0() == 0.0


@pytest.mark.parametrize("system", ["ising", "alloy"])
def test_endpoints_of_the_learned_prior_path(system):
    """U_0 = -log rho_0 and U_1 = beta H_c: the prior leaves no trace at t = 1."""
    e, shape, c = build(system)
    sigma, c = random_spins(8, shape), conditions(8, *c)
    with torch.no_grad():
        U0, _ = e.Ut(sigma, torch.zeros(8), c, need_dU=False)
        U1, _ = e.Ut(sigma, torch.ones(8), c, need_dU=False)
        log_rho0 = e.log_prior(sigma, c)
    if system == "ising":
        target = e.hamiltonian(sigma, c[:, 0]) / c[:, 1]
    else:
        target = bare_alloy(e, sigma, c)
    assert torch.allclose(U1, target, atol=1e-10)
    assert torch.allclose(U0, -log_rho0, atol=1e-12)


@pytest.mark.parametrize("system", ["ising", "alloy"])
def test_learned_prior_path_is_linear_and_increments_telescope(system):
    """Eq. (33) is linear in t, so the log-weight increments sum exactly to U_1 - U_0."""
    e, shape, c = build(system)
    sigma, c = random_spins(6, shape), conditions(6, *c)
    with torch.no_grad():
        U0, _ = e.Ut(sigma, torch.zeros(6), c, need_dU=False)
        U1, _ = e.Ut(sigma, torch.ones(6), c, need_dU=False)
        for t in (0.1, 0.5, 0.9):
            assert torch.allclose(e.dtUt(sigma, torch.full((6,), t), c), U1 - U0, atol=1e-10)
        ts = torch.cat([torch.zeros(1), torch.sort(torch.rand(7)).values, torch.ones(1)])
        total = sum(e.path_increment(sigma, t0.expand(6), (t1 - t0).expand(6), c)
                    for t0, t1 in zip(ts[:-1], ts[1:]))
    assert torch.allclose(total, U1 - U0, atol=1e-10)


def test_step_terms_equal_the_separate_operations():
    e = alloy()
    sigma = random_spins(3, CELL)
    c = torch.tensor([[0.01, 500.0], [-0.02, 600.0], [0.03, 700.0]])
    t, dt = torch.tensor([0.1, 0.4, 0.8]), torch.tensor([0.1, 0.2, 0.2])
    with torch.no_grad():
        U, dU = e.Ut(sigma, t, c)
        increment = e.path_increment(sigma, t, dt, c)
        terms = e.step_terms(sigma, t, dt, c)
    for actual, expected in zip(terms, (U, dU, increment)):
        assert torch.equal(actual, expected)


def test_alloy_hamiltonian_is_ce_minus_mu_n():
    """H_c = U_CE - dmu N_+ (Eq. 11): dmu multiplies only the count of species +1."""
    e = alloy()
    sigma = random_spins(6, CELL)
    n_plus = ((sigma.reshape(6, -1) + 1) / 2).sum(dim=1)
    ones = torch.ones(6)
    with torch.no_grad():
        Ua, _ = e.Ut(sigma, ones, conditions(6, 0.0, 500.0), need_dU=False)
        Ub, _ = e.Ut(sigma, ones, conditions(6, 0.03, 500.0), need_dU=False)
    assert torch.allclose(Ua - Ub, 0.03 * n_plus / (K_B * 500.0), atol=1e-10)


def test_uniform_prior_path():
    """The LEAPS ablation starts from the uniform distribution: U_0 = 0 and Z_0 = 2^N."""
    e = alloy(learned=False)
    sigma = all_spins(CELL)[::97]
    B = sigma.shape[0]
    c = conditions(B, 0.04, 400.0)
    U0, _ = e.Ut(sigma, torch.zeros(B), c)
    assert torch.equal(U0, torch.zeros(B))
    assert e.log_Z0() == pytest.approx(16 * math.log(2.0))
    U1, _ = e.Ut(sigma, torch.ones(B), c)
    assert torch.allclose(U1, bare_alloy(e, sigma, c), atol=1e-10)
    t, h = torch.full((B,), 0.3), 1e-6
    fd = (e.Ut(sigma, t + h, c)[0] - e.Ut(sigma, t - h, c)[0]) / (2 * h)
    assert torch.allclose(e.dtUt(sigma, t, c), fd, atol=1e-5)
    s = e.sample_prior(c[:5])
    assert s.shape == (5, *CELL) and set(s.unique().tolist()) <= {-1.0, 1.0}


@pytest.mark.parametrize("system", ["ising", "alloy"])
def test_learned_prior_samples_are_spins(system):
    e, shape, c = build(system)
    s = e.sample_prior(conditions(5, *c))
    assert s.shape == (5, *shape) and set(s.unique().tolist()) <= {-1.0, 1.0}


def test_norm_c_maps_the_box_to_the_unit_square():
    e = alloy()
    c = torch.tensor([ALLOY_BOX[0], ALLOY_BOX[1], [0.0, 550.0]])
    assert torch.allclose(e.norm_c(c), torch.tensor([[-1.0, -1.0], [1.0, 1.0], [0.0, 0.0]]))


def test_ess_edge_cases():
    """Eq. (1): 1 for equal weights whatever the offset, 1/M for one dominant weight,
    finite where the unshifted exponentials overflow, 0 for no walkers."""
    for offset in (0.0, -500.0, 700.0):
        assert ess(torch.full((64,), offset)).item() == pytest.approx(1.0)
    A = torch.full((100,), -1e3)
    A[0] = 0.0
    assert ess(A).item() == pytest.approx(0.01, rel=1e-6)
    A = torch.full((32,), 900.0)
    A[:16] = 899.0
    value = ess(A).item()
    assert math.isfinite(value) and 0.0 < value <= 1.0
    assert ess(torch.zeros(0)).item() == 0.0
