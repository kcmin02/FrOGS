"""Lightning training smoke tests, the annealing schedule and checkpoint loading."""
import argparse

import lightning as L
import pytest
import torch

from frogs.trainer import FrOGSModule, anneal_schedule

TINY = dict(prior_net=dict(num_layers=2, num_channels=4), num_channels=4, num_levels=1,
            free_energy_hidden=[16, 16], compile_prior=False, n_steps=4, n_anneal=4, n_mcmc=2,
            batch_size=4, loss_batch_size=6, buffer_size=2, buffer_cycle=3, seed=0)
CONFIGS = {
    "ising": dict(system="ising", lattice_shape=[4, 4], cond_low=[-0.05, 1.5],
                  cond_high=[0.05, 3.0]),
    "alloy": dict(system="cuau_ce", lattice_shape=[2, 2, 4],
                  table_path="data/cuau_cluster_tables_2x2x4.npz", cond_low=[-0.2, 200.0],
                  cond_high=[0.15, 1200.0], pool_strides=[[1, 1, 2]], metropolis_rates=True,
                  loo_context=True),
    "alloy-no-transport": dict(system="cuau_ce", lattice_shape=[2, 2, 4], transport=False,
                               table_path="data/cuau_cluster_tables_2x2x4.npz",
                               cond_low=[-0.2, 200.0], cond_high=[0.15, 1200.0]),
    "alloy-uniform-prior": dict(system="cuau_ce", lattice_shape=[2, 2, 4], prior="uniform",
                                table_path="data/cuau_cluster_tables_2x2x4.npz",
                                cond_low=[-0.2, 200.0], cond_high=[0.15, 1200.0],
                                pool_strides=[[1, 1, 2]], metropolis_rates=True),
}


def tiny_config(name):
    return argparse.Namespace(**TINY, **CONFIGS[name])


def fit(model, steps, tmp_path):
    trainer = L.Trainer(max_steps=steps, accelerator="cpu", devices=1, logger=False,
                        enable_checkpointing=False, enable_progress_bar=False,
                        enable_model_summary=False, default_root_dir=str(tmp_path))
    trainer.fit(model)
    return trainer


def test_anneal_schedule_ramps_from_k_min_to_k_max():
    ks = anneal_schedule(1, 125, 60_000)
    assert len(ks) == 60_000 and ks[0] == 1 and ks[-1] == 125
    assert all(a <= b for a, b in zip(ks, ks[1:])) and set(ks) == set(range(1, 126))
    counts = [ks.count(k) for k in (1, 60, 125)]
    assert counts[0] < counts[1] < counts[2]
    assert anneal_schedule(3, 3, 5) == [3] * 5
    with pytest.raises(ValueError):
        anneal_schedule(1, 10, 5)


@pytest.mark.parametrize("name", list(CONFIGS))
def test_two_training_steps(name, tmp_path):
    L.seed_everything(0)
    model = FrOGSModule(tiny_config(name))
    before = {k: v.clone() for k, v in model.state_dict().items()}
    trainer = fit(model, 2, tmp_path)
    assert trainer.global_step == 2
    after = model.state_dict()
    changed = [k for k in before if not torch.equal(before[k], after[k])]
    assert any(k.startswith("free_energy.") for k in changed)
    if model.transport is not None:
        assert any(k.startswith("transport.") for k in changed)
    if model.prior is not None:
        assert any(k.startswith("prior.") for k in changed)
    assert all(torch.isfinite(v).all() for v in after.values() if v.is_floating_point())
    assert model._skips == 0 and len(model.buffer) == 2


def test_checkpoint_roundtrip_and_legacy_prefixes(tmp_path):
    L.seed_everything(0)
    model = FrOGSModule(tiny_config("alloy"))
    trainer = fit(model, 1, tmp_path)
    path = tmp_path / "last.ckpt"
    trainer.save_checkpoint(path)
    torch.serialization.add_safe_globals([argparse.Namespace])
    loaded = FrOGSModule.load_from_checkpoint(path, map_location="cpu")
    assert vars(loaded.config) == vars(model.config)
    expected = model.state_dict()
    assert loaded.state_dict().keys() == expected.keys()
    assert all(torch.equal(v, expected[k]) for k, v in loaded.state_dict().items())

    # Checkpoints of the original code name the networks net., F_t_net. and _coupled_ar_direct.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    rename = {"transport.": "net.", "free_energy.": "F_t_net.", "prior.": "_coupled_ar_direct."}
    ckpt["state_dict"] = {next((old + k[len(new):] for new, old in rename.items()
                                if k.startswith(new)), k): v
                          for k, v in ckpt["state_dict"].items()}
    legacy = vars(ckpt["hyper_parameters"]["config"]).copy()
    legacy.update(model_class=legacy.pop("system"), cond_prior="ar_direct",
                  bs_per_gpu=legacy.pop("batch_size"), delta_t=1.0 / legacy.pop("n_steps"),
                  prior_kwargs=legacy.pop("prior_net"), use_loo_context=legacy.pop("loo_context"))
    ckpt["hyper_parameters"]["config"] = argparse.Namespace(**legacy)
    torch.save(ckpt, tmp_path / "legacy.ckpt")
    loaded = FrOGSModule.load_from_checkpoint(tmp_path / "legacy.ckpt", map_location="cpu")
    assert vars(loaded.config) == vars(model.config)
    assert all(torch.equal(v, expected[k]) for k, v in loaded.state_dict().items())


def test_non_finite_steps_are_skipped():
    model = FrOGSModule(tiny_config("ising"))
    params = list(model.free_energy.parameters())
    opt = torch.optim.SGD(params, lr=1.0)
    before = [p.detach().clone() for p in params]
    for p in params:
        p.grad = torch.full_like(p, float("nan"))
    model.optimizer_step(0, 0, opt, lambda: torch.tensor(float("inf")))
    assert model._skips == 1
    assert all(torch.equal(p, b) for p, b in zip(params, before))
    for p in params:
        p.grad = torch.ones_like(p)
    model.optimizer_step(0, 0, opt, lambda: torch.tensor(1.0))
    assert model._skips == 1
    assert not torch.equal(params[0], before[0])
