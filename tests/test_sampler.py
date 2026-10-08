"""CTMC rollouts with Jarzynski log-weights and the locally equivariant rate matrix (§3, §A.7)."""
import numpy as np
import pytest
import torch

from frogs.config import normalize
from frogs.energy import make_energy
from frogs.nets import build_prior, build_transport
from frogs.nets.transport import CenterMaskedConv2d, CenterMaskedConv3d, StateDependentConv
from frogs.sampler import CTMCSampler

SPECS = {
    "alloy": dict(system="cuau_ce", lattice_shape=[2, 2, 4],
                  table_path="data/cuau_cluster_tables_2x2x4.npz",
                  cond_low=[-0.05, 200.0], cond_high=[0.05, 900.0], num_levels=1,
                  pool_strides=[[1, 1, 2]], metropolis_rates=True, loo_context=True),
    "alloy-uniform": dict(system="cuau_ce", lattice_shape=[2, 2, 4], prior="uniform",
                          table_path="data/cuau_cluster_tables_2x2x4.npz",
                          cond_low=[-0.05, 200.0], cond_high=[0.05, 900.0], num_levels=1,
                          pool_strides=[[1, 1, 2]], metropolis_rates=True),
    "ising": dict(system="ising", lattice_shape=[4, 4], cond_low=[-0.05, 1.5],
                  cond_high=[0.05, 3.0], num_levels=1),
}


def build(name):
    cfg = normalize(dict(SPECS[name], prior_net=dict(num_layers=2, num_channels=4),
                         num_channels=4))
    torch.manual_seed(0)
    prior = build_prior(cfg) if cfg.prior == "autoregressive" else None
    return make_energy(cfg, prior), build_transport(cfg), cfg


def conditions(cfg, n):
    lo, hi = torch.tensor(cfg.cond_low), torch.tensor(cfg.cond_high)
    return lo + (hi - lo) * torch.rand(n, 2)


def rollout(energy, transport, c, n_steps=4, n_mcmc=0, n_save=1, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    with torch.no_grad():
        sigma0 = energy.sample_prior(c)
        start = sigma0.clone()
        sampler = CTMCSampler(energy, torch.linspace(0.0, 1.0, n_steps + 1), transport,
                              n_mcmc=n_mcmc, n_save=n_save)
        sigmas, A = sampler.rollout(sigma0, c)
    return start, sigmas, A


@pytest.mark.parametrize("name", list(SPECS))
@pytest.mark.parametrize("n_mcmc", [0, 2])
def test_rollout_shapes_and_fp64_weights(name, n_mcmc):
    energy, transport, cfg = build(name)
    c = conditions(cfg, 5)
    start, sigmas, A = rollout(energy, transport, c, n_mcmc=n_mcmc, n_save=2)
    assert sigmas.shape == (3, 5, *cfg.lattice_shape) and A.shape == (3, 5)
    assert A.dtype == torch.float64 and torch.isfinite(A).all()
    assert torch.equal(A[0], torch.zeros(5)) and torch.equal(sigmas[0], start)
    assert set(sigmas.unique().tolist()) <= {-1.0, 1.0}
    assert not torch.equal(sigmas[-1], start)


@pytest.mark.parametrize("name", ["alloy", "ising"])
def test_without_moves_the_weight_is_the_endpoint_difference(name, float64):
    """With no transport and no Metropolis moves, A = U_0 - U_1 on any time grid."""
    energy, _, cfg = build(name)
    c = conditions(cfg, 6)
    start, sigmas, A = rollout(energy, None, c, n_steps=7)
    n = c.shape[0]
    with torch.no_grad():
        U0, _ = energy.Ut(start, torch.zeros(n), c, need_dU=False)
        U1, _ = energy.Ut(start, torch.ones(n), c, need_dU=False)
    assert torch.equal(sigmas[-1], start)
    assert torch.allclose(A[-1], U0 - U1, atol=1e-10)


@pytest.mark.parametrize("name", list(SPECS))
@pytest.mark.parametrize("n_mcmc", [0, 2])
def test_sharing_the_transport_output_within_a_step_is_bit_identical(monkeypatch, name, n_mcmc):
    energy, transport, cfg = build(name)
    c = conditions(cfg, 4)
    shared = rollout(energy, transport, c, n_mcmc=n_mcmc)
    states = torch.random.get_rng_state(), np.random.get_state()
    monkeypatch.setattr(CTMCSampler, "_reuse_q", lambda self, sigma: False)
    separate = rollout(energy, transport, c, n_mcmc=n_mcmc)
    for a, b in zip(shared, separate):
        assert torch.equal(a, b)
    assert torch.equal(torch.random.get_rng_state(), states[0])
    assert np.array_equal(np.random.get_state()[1], states[1][1])


@pytest.mark.parametrize("name", ["alloy", "ising"])
def test_rate_network_is_locally_equivariant(name, float64):
    """g(i | sigma) = -g(i | flip_i sigma) for any weights; removing the centre masks breaks it."""
    _, transport, cfg = build(name)
    sigma = torch.where(torch.rand(3, *cfg.lattice_shape) < 0.5, -1.0, 1.0)
    t, c = torch.rand(3), conditions(cfg, 3)

    def max_residual():
        with torch.no_grad():
            g = transport(sigma, t, c).reshape(3, -1)
            flat = sigma.reshape(3, -1)
            worst = 0.0
            for i in range(flat.shape[1]):
                flipped = flat.clone()
                flipped[:, i] *= -1
                g_flip = transport(flipped.reshape(sigma.shape), t, c).reshape(3, -1)[:, i]
                worst = max(worst, (g[:, i] + g_flip).abs().max().item())
        return worst / g.abs().max().item()

    assert max_residual() < 1e-12
    for m in transport.modules():
        if isinstance(m, (CenterMaskedConv2d, CenterMaskedConv3d)):
            m.mask.fill_(1.0)
        if isinstance(m, StateDependentConv):
            m.mid_idx = slice(0, 0)
    assert max_residual() > 1e-3
