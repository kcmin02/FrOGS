"""The symmetry group detected for each target and the symmetry penalty on the prior."""
import argparse
import glob
import itertools
import math
import os
from unittest import mock

import lightning as L
import numpy as np
import pytest
import torch

from frogs.config import load_config
from frogs.energy import AlloyEnergy, IsingEnergy
from frogs.energy.symmetry import SymmetryGroup, log_prior_gap, preserves_supercell
from frogs.nets import ARConvPrior
from frogs.paths import ROOT, data_file
from frogs.trainer import FrOGSModule

CONFIGS = sorted(glob.glob(os.path.join(ROOT, "configs", "**", "*.yml"), recursive=True))

ISING_BOX = ([-0.05, 1.5], [0.05, 3.0])
ALLOY_BOXES = {"cuau": ([-0.2, 200.0], [0.15, 1200.0]), "agpd": ([-0.4, 200.0], [0.4, 900.0])}
TABLES = {"cuau": ("cuau_cluster_tables_8x4x4.npz", (8, 4, 4)),
          "agpd": ("agpd_cluster_tables_L5.npz", (5, 5, 5))}
TINY = dict(prior_net=dict(num_layers=2, num_channels=4), num_channels=4, num_levels=1,
            free_energy_hidden=[16, 16], compile_prior=False, n_steps=4, n_anneal=4, n_mcmc=2,
            batch_size=4, loss_batch_size=6, buffer_size=2, buffer_cycle=3, seed=0)
SPECS = {
    "ising": dict(system="ising", lattice_shape=[4, 4], cond_low=ISING_BOX[0],
                  cond_high=ISING_BOX[1]),
    "alloy": dict(system="cuau_ce", lattice_shape=[2, 2, 4],
                  table_path="data/cuau_cluster_tables_2x2x4.npz", cond_low=[-0.2, 200.0],
                  cond_high=[0.15, 1200.0], pool_strides=[[1, 1, 2]], metropolis_rates=True,
                  loo_context=True),
}


def tiny_prior(shape, seed=0):
    torch.manual_seed(seed)
    prior = ARConvPrior(shape, num_layers=2, num_channels=4)
    with torch.no_grad():
        for film in (prior.film_initial, *prior.films):
            film.net[-1].weight.normal_(0, 0.1)
            film.net[-1].bias.normal_(0, 0.1)
    return prior


def constant_logit(prior, logit):
    with torch.no_grad():
        prior.head.weight.zero_()
        prior.head.bias.fill_(logit)
    return prior


def build(system, L=4, box=None, cls=IsingEnergy):
    if system == "ising":
        lo, hi = box or ISING_BOX
        return cls(L, tiny_prior((1, L, L)), cond_low=lo, cond_high=hi)
    name, shape = TABLES[system]
    lo, hi = box or ALLOY_BOXES[system]
    return AlloyEnergy(data_file(name), shape, prior=tiny_prior(shape), cond_low=lo, cond_high=hi)


def random_spins(n, shape):
    return torch.where(torch.rand(n, *shape) < 0.5, -1.0, 1.0)


def random_conditions(energy, n):
    lo, hi = torch.tensor(energy.cond_low), torch.tensor(energy.cond_high)
    return lo + (hi - lo) * torch.rand(n, 2)


def beta_H(energy, sigma, c):
    with torch.no_grad():
        return energy.Ut(sigma, torch.ones(sigma.shape[0]), c, need_dU=False)[0]


def elements(group, translations):
    rows = list(itertools.product(range(len(group.point)), translations))
    return tuple(torch.tensor(col) for col in zip(*rows))


def element_set(group, c=(0.03, 2.0)):
    q, r = elements(group, range(len(group.translation)))
    sigma = torch.arange(1.0, group.n_sites + 1, dtype=torch.float64).repeat(len(q), 1)
    out, g_c = group.apply((q, r), sigma, torch.tensor([c], dtype=torch.float64).repeat(len(q), 1))
    return {row.tobytes() for row in torch.cat([out, g_c], dim=1).numpy()}


def signed_permutations(dim):
    out = []
    for perm in itertools.permutations(range(dim)):
        for signs in itertools.product((1, -1), repeat=dim):
            M = np.zeros((dim, dim), dtype=np.int64)
            M[np.arange(dim), perm] = signs
            out.append(M)
    return out


def expected_elements(system, energy, c=(0.03, 2.0)):
    shape = energy.lattice_shape
    if system == "ising":
        ops, signs = signed_permutations(2), (1, -1)
    else:
        A = energy.ce.prim_vectors.T
        ops = [np.rint(np.linalg.solve(A, R @ A)).astype(np.int64) for R in signed_permutations(3)]
        ops, signs = [M for M in ops if preserves_supercell(M, shape)], (1,)
    dims = np.array(shape)
    sites = np.stack(np.meshgrid(*map(np.arange, shape), indexing="ij"), -1).reshape(-1, len(shape))
    out = set()
    for M, r, s in itertools.product(ops, sites, signs):
        perm = np.ravel_multi_index(((((sites + r) % dims) @ M.T) % dims).T, shape)
        out.add(np.concatenate([s * (perm + 1.0), [s * c[0], c[1]]]).tobytes())
    return out


class AnisotropicIsing(IsingEnergy):
    """Coupling J / 2 along the second axis."""

    def hamiltonian(self, sigma, h):
        bonds = (sigma * torch.roll(sigma, -1, dims=2)).flatten(1).sum(dim=1)
        return super().hamiltonian(sigma, h) + 0.5 * self.J * bonds


class BiasedIsing(IsingEnergy):
    """A fixed uniform field 0.1 on top of h."""

    def hamiltonian(self, sigma, h):
        return super().hamiltonian(sigma, h) - 0.1 * sigma.flatten(1).sum(dim=1)


class ChiralIsing(IsingEnergy):
    """Adds sum_n s_n s_{n+x+y} (s_{n+x} - s_{n+y}), odd under the spin flip and under x <-> y."""

    def hamiltonian(self, sigma, h):
        x, y, xy = (torch.roll(sigma, (-a, -b), dims=(1, 2)) for a, b in ((1, 0), (0, 1), (1, 1)))
        return super().hamiltonian(sigma, h) + 0.5 * (sigma * xy * (x - y)).flatten(1).sum(dim=1)


class PinnedIsing(IsingEnergy):
    """A fixed field 0.5 on site 0 only."""

    def hamiltonian(self, sigma, h):
        return super().hamiltonian(sigma, h) - 0.5 * sigma[:, 0, 0]


I2, SWAP = np.eye(2, dtype=np.int64), np.array([[0, 1], [1, 0]])


@pytest.mark.parametrize("system,L,size", [
    ("ising", 4, 256), ("ising", 5, 400), ("ising", 10, 1600), ("ising", 15, 3600),
    ("ising", 20, 6400), ("cuau", None, 1536), ("agpd", None, 6000)])
def test_detection_finds_the_expected_groups(system, L, size):
    energy = build(system, L)
    group = energy.symmetry_group()
    found = element_set(group)
    assert group.size == len(found) == size
    assert found == expected_elements(system, energy)


@pytest.mark.parametrize("L", [4, 5])
@pytest.mark.parametrize("cls,ops,translations", [
    (AnisotropicIsing, [(M, s) for M in (I2, -I2, np.diag([1, -1]), np.diag([-1, 1]))
                        for s in (1, -1)], True),
    (BiasedIsing, [(M, 1) for M in signed_permutations(2)], True),
    (ChiralIsing, [(I2, 1), (-SWAP, 1), (-I2, -1), (SWAP, -1)], True),
    (PinnedIsing, [(M, 1) for M in signed_permutations(2)], False),
], ids=["anisotropic", "biased", "chiral", "pinned"])
def test_detection_discovers_the_symmetries(L, cls, ops, translations):
    group = build("ising", L, cls=cls).symmetry_group()
    expected = SymmetryGroup((L, L), ops, field=0, translations=translations)
    assert group.size == expected.size and element_set(group) == element_set(expected)


def test_asymmetric_field_box_drops_the_flip():
    full = build("ising").symmetry_group()
    group = build("ising", box=([-0.05, 1.5], [0.1, 3.0])).symmetry_group()
    assert group.size == 128 and not group.flips
    assert torch.equal(group.point, full.point[full.sign > 0])


@pytest.mark.parametrize("system", ["cuau", "agpd"])
def test_alloy_species_swap_is_rejected_by_the_energy(system):
    lo, hi = ALLOY_BOXES[system]
    energy = build(system, box=([-hi[0], lo[1]], hi))
    torch.manual_seed(3)
    sigma, c = random_spins(64, energy.lattice_shape), random_conditions(energy, 64)
    swap = torch.tensor([-1.0, 1.0])
    assert (beta_H(energy, -sigma, c * swap) - beta_H(energy, sigma, c)).std() > 0.5
    group = energy.symmetry_group()
    assert not group.flips and element_set(group) == expected_elements(system, energy)

    odd = (energy.ce.all_inds < energy.n_sites).sum(dim=1) % 2 == 1
    energy.ce.all_weights[odd] = 0.0
    c = torch.tensor([[0.1, 500.0]]).repeat(64, 1)
    offset = beta_H(energy, -sigma, c * swap) - beta_H(energy, sigma, c)
    assert offset.abs().min() > 1 and torch.allclose(offset, offset[0].expand(64), rtol=1e-6)
    even = energy.symmetry_group()
    assert even.size == 2 * group.size and torch.equal(even.point[even.sign > 0], group.point)
    assert torch.equal(even.point[even.sign < 0], group.point)


def test_detection_is_deterministic_and_leaves_the_global_rng_alone():
    energy = build("ising")
    torch.manual_seed(3)
    expected = torch.rand(4)
    torch.manual_seed(3)
    first = energy.symmetry_group()
    assert torch.equal(torch.rand(4), expected)
    torch.manual_seed(4)
    second = energy.symmetry_group()
    assert torch.equal(first.point, second.point) and torch.equal(first.sign, second.sign)


@pytest.mark.parametrize("system,L", [("ising", 4), ("ising", 5), ("cuau", None), ("agpd", None)])
def test_group_is_closed_and_acts_by_permutations(system, L):
    group = build(system, L).symmetry_group()
    point, sign, trans = group.point.numpy(), group.sign.numpy(), group.translation.numpy()
    n = group.n_sites
    for perm in (*point, *trans):
        assert np.array_equal(np.sort(perm), np.arange(n))
    members = point[:, trans].reshape(-1, n)
    signs = np.repeat(sign, len(trans))
    keys = {(row.tobytes(), s) for row, s in zip(members, signs)}
    assert len(keys) == group.size and (np.arange(n).tobytes(), 1) in keys
    shape = group.lattice_shape
    units = [np.ravel_multi_index(np.eye(len(shape), dtype=np.int64)[k], shape)
             for k in range(len(shape))]
    for gen, s in (*zip(point, sign), *((trans[u], 1) for u in units)):
        assert all((row.tobytes(), t * s) in keys for row, t in zip(members[:, gen], signs))


def test_ising_flip_maps_h_to_minus_h():
    group = build("ising").symmetry_group()
    identity = (group.point == torch.arange(16)).all(dim=1)
    flip = int(torch.nonzero(identity & (group.sign < 0)))
    sigma, c = random_spins(3, (4, 4)), torch.tensor([[0.03, 2.0], [-0.01, 1.7], [0.0, 2.9]])
    zero = torch.zeros(3, dtype=torch.int64)
    out, c_out = group.apply((torch.full((3,), flip), zero), sigma, c)
    assert torch.equal(out, -sigma) and torch.equal(c_out, c * torch.tensor([-1.0, 1.0]))
    out, c_out = group.apply((zero, zero), sigma, c)
    assert torch.equal(out, sigma) and torch.equal(c_out, c)


@pytest.mark.usefixtures("float64")
@pytest.mark.parametrize("system", ["ising", "cuau", "agpd"])
def test_hamiltonian_is_invariant(system):
    torch.manual_seed(1)
    energy = build(system)
    group = energy.symmetry_group()
    shape = energy.lattice_shape
    g = elements(group, torch.randint(group.n_sites, (3,)).tolist())
    sigma = random_spins(len(g[0]), shape)
    c = random_conditions(energy, len(g[0]))
    g_sigma, g_c = group.apply(g, sigma, c)
    assert not torch.equal(g_sigma, sigma)
    assert torch.allclose(beta_H(energy, g_sigma, g_c), beta_H(energy, sigma, c), rtol=0, atol=1e-9)


@pytest.mark.usefixtures("float64")
def test_non_symmetries_change_the_alloy_energy():
    energy = build("cuau")
    wrong = SymmetryGroup(energy.lattice_shape, [(np.diag([-1, 1, 1]), 1)])
    sigma = random_spins(4, energy.lattice_shape)
    c = random_conditions(energy, 4)
    zero = torch.zeros(4, dtype=torch.int64)
    g_sigma, _ = wrong.apply((zero, zero), sigma, c)
    assert (beta_H(energy, g_sigma, c) - beta_H(energy, sigma, c)).abs().max() > 1e-3


@pytest.mark.usefixtures("float64")
@pytest.mark.parametrize("system", ["ising", "cuau", "agpd"])
def test_penalty_vanishes_only_for_an_invariant_prior(system):
    torch.manual_seed(2)
    energy = build(system)
    group = energy.symmetry_group()
    sigma, c = random_spins(64, energy.lattice_shape), random_conditions(energy, 64)
    generic = (log_prior_gap(energy, group, sigma, c) ** 2).mean()
    assert generic > 1e-3
    constant_logit(energy.prior, 0.0 if system == "ising" else 0.7)
    assert (log_prior_gap(energy, group, sigma, c) ** 2).mean() < 1e-20
    if system == "ising":
        constant_logit(energy.prior, 0.7)
        assert (log_prior_gap(energy, group, sigma, c) ** 2).mean() > 1e-2


@pytest.mark.usefixtures("float64")
def test_certificate_is_an_exact_kl():
    energy = build("ising", L=3)
    group = energy.symmetry_group()
    sigma = torch.tensor(list(itertools.product([-1.0, 1.0], repeat=9))).reshape(-1, 3, 3)
    B = sigma.shape[0]
    c = torch.tensor([[0.02, 2.0]]).expand(B, -1)
    with torch.no_grad():
        rho = torch.exp(energy.log_prior(sigma, c))
        kls = []
        for q, r in itertools.product(range(len(group.point)), range(9)):
            g = tuple(torch.full((B,), v, dtype=torch.int64) for v in (q, r))
            g_sigma, g_c = group.apply(g, sigma, c)
            assert torch.exp(energy.log_prior(g_sigma, g_c)).sum().item() == pytest.approx(1.0)
            with mock.patch.object(group, "sample", return_value=g):
                kls.append((rho * log_prior_gap(energy, group, sigma, c)).sum().item())
    assert min(kls) > -1e-12 and max(kls) > 1e-4 and kls[0] == pytest.approx(0.0, abs=1e-12)


def tiny_model(name, sym_penalty):
    L.seed_everything(0)
    return FrOGSModule(argparse.Namespace(**TINY, **SPECS[name], sym_penalty=sym_penalty))


@pytest.mark.parametrize("name", list(SPECS))
def test_penalty_adds_only_to_the_prior_gradient(name):
    grads, losses = {}, {}
    for lam in (0.0, 1.0):
        model = tiny_model(name, lam)
        k = 3
        ts = torch.linspace(0.0, k * model.dt, k + 1)
        sigmas, As, c = model._rollout(k, ts)
        torch.manual_seed(1)
        model.buffer = [(sigmas, As, k, ts, c)] * model.config.buffer_size
        loss = model.training_step(None, 0)
        loss.backward()
        losses[lam] = loss.item()
        grads[lam] = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    off, on = grads[0.0], grads[1.0]
    assert off.keys() == on.keys() and losses[1.0] > losses[0.0]
    assert all(torch.equal(off[n], on[n]) for n in off if not n.startswith("prior."))
    assert any(not torch.equal(off[n], on[n]) for n in off if n.startswith("prior."))


def fit(model, steps, tmp_path):
    trainer = L.Trainer(max_steps=steps, accelerator="cpu", devices=1, logger=False,
                        enable_checkpointing=False, enable_progress_bar=False,
                        enable_model_summary=False, default_root_dir=str(tmp_path))
    trainer.fit(model)
    return trainer


def test_zero_penalty_adds_no_calls(tmp_path):
    with mock.patch("frogs.energy.base.detect_symmetry", side_effect=AssertionError("called")):
        model = tiny_model("ising", 0.0)
    assert model.symmetry is None
    with mock.patch("frogs.trainer.log_prior_gap", side_effect=AssertionError("called")):
        trainer = fit(model, 2, tmp_path)
    assert "train_sym_penalty" not in trainer.callback_metrics


@pytest.mark.parametrize("name", list(SPECS))
def test_penalty_and_certificate_are_logged(name, tmp_path):
    m = fit(tiny_model(name, 1.0), 2, tmp_path).callback_metrics
    assert float(m["train_sym_penalty"]) > 0 and math.isfinite(float(m["val_sym_kl"]))
    kl = float(m["val_sym_kl"])
    assert float(m["val_sym_tv_bound"]) == pytest.approx(min(math.sqrt(max(kl, 0) / 2), 1))


@pytest.mark.parametrize("path", CONFIGS, ids=[os.path.relpath(p, ROOT) for p in CONFIGS])
def test_one_step_with_the_penalty_on_every_config(path, tmp_path):
    cfg = load_config(path, ["sym_penalty=1.0", "batch_size=2", "loss_batch_size=2",
                             "compile_prior=false", "seed=0"])
    L.seed_everything(0)
    model = FrOGSModule(cfg)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    m = fit(model, 1, tmp_path).callback_metrics
    penalty, kl = float(m["train_sym_penalty"]), float(m["val_sym_kl"])
    if model.prior is None:
        assert penalty == 0.0 and kl == 0.0
    else:
        assert 0.0 < penalty < 1e6 and math.isfinite(kl)
        assert any(not torch.equal(before[k], v) for k, v in model.state_dict().items()
                   if k.startswith("prior."))
    assert model._skips == 0
    assert all(torch.isfinite(v).all() for v in model.state_dict().values()
               if v.is_floating_point())
