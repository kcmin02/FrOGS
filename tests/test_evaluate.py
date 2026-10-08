"""Importance-sampling readout (Eq. 2, Eq. 3) on a tiny untrained model."""
import argparse
import math

import numpy as np
import pytest
import torch

from frogs.analysis.sweep_io import load_sweep, save_sweep, unpack_sigma
from frogs.evaluate import ISEvaluator, kappa2_log_mean_exp, raw_log_mean_exp
from frogs.trainer import FrOGSModule

CONFIG = dict(system="cuau_ce", lattice_shape=[2, 2, 4],
              table_path="data/cuau_cluster_tables_2x2x4.npz", cond_low=[-0.2, 200.0],
              cond_high=[0.15, 1200.0], prior_net=dict(num_layers=2, num_channels=4),
              num_channels=4, num_levels=1, pool_strides=[[1, 1, 2]], metropolis_rates=True,
              loo_context=True, free_energy_hidden=[16], compile_prior=False, n_steps=6)
DMU, T = [-0.05, 0.02], [300.0, 800.0]


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return FrOGSModule(argparse.Namespace(**CONFIG)).eval()


def run(model, **kwargs):
    ev = ISEvaluator(model, M=10, n_mcmc=kwargs.pop("n_mcmc", 1), seed=3, device="cpu",
                     transport=kwargs.pop("transport", None))
    return ev, ev.run(DMU, T, chunk=7, n_boot=20, store_A=True, store_sigma=True, **kwargs)


def test_sweep_schema_and_consistency(model, tmp_path):
    _, out = run(model)
    grid = (2, 2)
    for key in ("logZ", "logZ_se", "logZ_raw", "x", "x_se", "ess", "M"):
        assert out[key].shape == grid and np.isfinite(out[key]).all(), key
    assert out["A_samples"].shape == out["N_samples"].shape == (*grid, 10)
    assert out["sigma_packed"].shape == (*grid, 10, 2) and out["sigma_packed"].dtype == np.uint8
    assert np.all((out["ess"] > 0) & (out["ess"] <= 1 + 1e-12))
    spins = unpack_sigma(out["sigma_packed"], 16)
    np.testing.assert_array_equal(((spins + 1) // 2).sum(axis=-1), out["N_samples"])
    A = out["A_samples"]
    w = np.exp(A - A.max(axis=-1, keepdims=True))
    np.testing.assert_allclose(out["x"], (w * out["N_samples"]).sum(-1) / w.sum(-1) / 16)
    np.testing.assert_allclose(out["logZ_raw"],
                               np.log(np.exp(A - A.max(-1, keepdims=True)).mean(-1))
                               + A.max(-1))

    path = tmp_path / "sweep.npz"
    save_sweep(path, out, dict(N=16))
    back = load_sweep(path)
    assert back["meta"]["N"] == 16
    for key in ("logZ", "x", "ess", "A_samples", "sigma_packed", "dmu", "T"):
        np.testing.assert_array_equal(back[key], out[key])


def test_runs_are_reproducible(model):
    _, a = run(model)
    _, b = run(model)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])


def test_prior_only_readout_is_the_endpoint_difference(model):
    """Without transport or Metropolis moves the walkers stay put and A = U_0 - U_1."""
    ev, out = run(model, transport=False, n_mcmc=0)
    assert ev.n_steps == 1
    spins = torch.as_tensor(unpack_sigma(out["sigma_packed"], 16), dtype=torch.float32)
    c = torch.tensor([[d, t] for d in DMU for t in T]).repeat_interleave(10, dim=0)
    sigma = spins.reshape(-1, 2, 2, 4)
    with torch.no_grad():
        n = sigma.shape[0]
        U0, _ = model.energy.Ut(sigma, torch.zeros(n), c, need_dU=False)
        U1, _ = model.energy.Ut(sigma, torch.ones(n), c, need_dU=False)
    np.testing.assert_allclose(out["A_samples"].reshape(-1), (U0 - U1).double().numpy(),
                               rtol=1e-5, atol=1e-4)


def test_log_mean_exp_estimators():
    A = torch.full((50,), 3.0, dtype=torch.float64)
    assert raw_log_mean_exp(A) == pytest.approx(3.0)
    assert kappa2_log_mean_exp(A) == pytest.approx(3.0)
    A = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    assert kappa2_log_mean_exp(A) == pytest.approx(1.5 + 0.5 * A.var().item())
    assert raw_log_mean_exp(A) == pytest.approx(math.log(np.exp([0, 1, 2, 3]).mean()))
