"""Every shipped config loads and builds, and old checkpoint configs translate to the new keys."""
import argparse
import glob
import os

import numpy as np
import pytest
import torch

from frogs.config import DEFAULTS, SYSTEMS, load_config, normalize
from frogs.energy import DEFAULT_TABLES
from frogs.nets import build_transport
from frogs.paths import ROOT, resolve_data_file

CONFIGS = sorted(glob.glob(os.path.join(ROOT, "configs", "**", "*.yml"), recursive=True))
ids = [os.path.relpath(p, ROOT) for p in CONFIGS]

# The recorded config of the CuAu production checkpoint, minus machine paths.
LEGACY_CUAU = dict(
    F_hidden_sizes=[200, 200, 200, 200], L=8, accelerator="gpu", anneal_quadratic=True,
    arch="conv_unet3d", base_lr=0.0003, beta_0=0.0, beta_1=58.0, bs_for_loss_per_gpu=100,
    bs_per_gpu=50, buffer_cycle=100, ckpt_every=10000, ckpt_fname=None, cond_dim=2,
    cond_high=[0.15, 1200.0], cond_low=[-0.2, 200.0], cond_prior="ar_direct", couple_prior=True,
    cuau_table_path="/elsewhere/cuau_cluster_tables_8x4x4.npz", decoder_convs_per_scale=1,
    delta_t=0.008, encoder_convs_per_scale=1, grad_clip_norm=1000, in_channels=1, kernel_size=3,
    last_ckpt_every=100, lattice_shape=[8, 4, 4], log_every_local=1, lr_sched_gamma=0.995,
    lr_sched_step_size=1000, max_buffer_size=100, model_class="cuau_ce", mu_0=0.0, mu_1=0.0,
    n_anneal=60000, n_mcmc_per_net=10, num_channels=16, num_gpus=1, num_levels=2, num_nodes=1,
    num_workers=4, path="linear", phase_mix_ar_kwargs=argparse.Namespace(
        num_layers=4, num_channels=32, kernel_size=3),
    plot_every=10000, pool_strides=[[2, 1, 1], [2, 2, 2]], prior_lr=0.0001,
    reparam_metropolis=True, run_tag="cuau128", starting_k=1, target="cuau_ce",
    use_buffer=True, use_loo_context=True, wandb_project="cuau")


def test_configs_exist():
    assert CONFIGS


@pytest.mark.parametrize("path", CONFIGS, ids=ids)
def test_config_loads(path):
    cfg = load_config(path)
    assert cfg.system in SYSTEMS
    assert cfg.prior in ("autoregressive", "uniform")
    assert cfg.prior_backend in ("reference", "systems")
    assert len(cfg.lattice_shape) == (2 if cfg.system == "ising" else 3)
    assert len(cfg.cond_low) == len(cfg.cond_high) == 2
    assert all(lo < hi for lo, hi in zip(cfg.cond_low, cfg.cond_high))
    assert hasattr(cfg, "seed") and hasattr(cfg, "run_tag")
    if cfg.system != "ising":
        table = resolve_data_file(cfg.table_path, DEFAULT_TABLES[cfg.system])
        assert int(np.load(table)["n_sites"]) == int(np.prod(cfg.lattice_shape))


@pytest.mark.parametrize("path", CONFIGS, ids=ids)
def test_transport_builds_and_has_room_to_pool(path):
    """Lemma 4 (§A.7) needs every axis of the coarsest pyramid level to have length >= 2."""
    cfg = load_config(path)
    if not cfg.transport:
        return
    rates = build_transport(cfg)
    rates.network._pyramid(torch.zeros(1, 1, *cfg.lattice_shape))
    # The Metropolis-form rates of Eq. (7) exist for the 3D alloys only.
    assert rates.metropolis_rates == (bool(cfg.metropolis_rates) and cfg.system != "ising")


def test_legacy_checkpoint_config_translates():
    cfg = normalize(argparse.Namespace(**LEGACY_CUAU))
    assert cfg.system == "cuau_ce" and cfg.prior == "autoregressive"
    assert cfg.lattice_shape == [8, 4, 4] and cfg.n_steps == 125
    assert cfg.prior_net == dict(num_layers=4, num_channels=32, kernel_size=3)
    assert cfg.table_path.endswith("cuau_cluster_tables_8x4x4.npz")
    assert (cfg.batch_size, cfg.loss_batch_size, cfg.buffer_size, cfg.n_mcmc) == (50, 100, 100, 10)
    assert (cfg.lr, cfg.lr_gamma, cfg.lr_step, cfg.grad_clip) == (0.0003, 0.995, 1000, 1000)
    assert cfg.metropolis_rates and cfg.loo_context and cfg.transport
    assert cfg.encoder_convs == cfg.decoder_convs == 1
    assert cfg.free_energy_hidden == [200, 200, 200, 200]
    assert cfg.prior_backend == "reference"
    for old in ("model_class", "target", "cond_prior", "delta_t", "bs_per_gpu", "L", "beta_1",
                "phase_mix_ar_kwargs", "reparam_metropolis", "use_loo_context", "arch"):
        assert not hasattr(cfg, old), old
    assert vars(normalize(cfg)) == vars(cfg)


def test_legacy_switches_translate():
    cfg = normalize(dict(model_class="ising", L=10, cond_low=[-0.05, 1.5], cond_high=[0.05, 3.0],
                         no_transport=True, use_buffer=False,
                         prior_kwargs=dict(num_layers=3, compile_ar=False,
                                           local_flip_mode="triton")))
    assert cfg.system == "ising" and cfg.lattice_shape == [10, 10]
    assert cfg.transport is False and cfg.buffer_size == 0
    assert cfg.prior_backend == "systems" and cfg.compile_prior is False
    assert cfg.prior_net == dict(num_layers=3, num_channels=32, kernel_size=3)


def test_defaults_fill_missing_keys():
    cfg = normalize(dict(system="agpd_ce", lattice_shape=[5, 5, 5], cond_low=[-0.4, 200],
                         cond_high=[0.4, 1200]))
    for key, value in DEFAULTS.items():
        assert getattr(cfg, key) == value, key
    assert cfg.cond_low == [-0.4, 200.0]


@pytest.mark.parametrize("bad", [
    dict(system="potts"), dict(prior="ar_direct_mix"), dict(prior_backend="triton"),
    dict(system="ising", prior="uniform"), dict(sym_penalty=-1.0)])
def test_normalize_rejects_unknown_choices(bad):
    cfg = dict(system="cuau_ce", lattice_shape=[8, 4, 4], cond_low=[-0.2, 200],
               cond_high=[0.15, 1200])
    cfg.update(bad)
    if cfg["system"] == "ising":
        cfg["lattice_shape"] = [4, 4]
    with pytest.raises(ValueError):
        normalize(cfg)


def test_overrides_are_parsed_as_yaml():
    path = os.path.join(ROOT, "configs", "cuau_128.yml")
    cfg = load_config(path, ["seed=3", "run_tag=s3", "prior_net={num_channels: 16}", "n_mcmc=0"])
    assert cfg.seed == 3 and cfg.run_tag == "s3" and cfg.n_mcmc == 0
    assert cfg.prior_net == dict(num_layers=4, num_channels=16, kernel_size=3)
    with pytest.raises(ValueError):
        load_config(path, ["seed"])
