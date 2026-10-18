"""YAML run configuration.

A config is a flat YAML file (see `configs/`) loaded into an `argparse.Namespace`, which is also
what Lightning stores in a checkpoint. `normalize` fills defaults and translates the key names
used by older checkpoints.
"""
from __future__ import annotations

import argparse
import copy
import os

import yaml

SYSTEMS = ("ising", "agpd_ce", "cuau_ce")

DEFAULTS = dict(
    run_tag=None,
    seed=None,
    J=1.0,
    table_path=None,
    prior="autoregressive",
    prior_net=dict(num_layers=4, num_channels=32, kernel_size=3),
    prior_lr=1e-4,
    prior_backend="reference",
    compile_prior=True,
    transport=True,
    num_levels=2,
    pool_strides=None,
    kernel_size=3,
    num_channels=16,
    encoder_convs=1,
    decoder_convs=1,
    loo_context=False,
    metropolis_rates=False,
    full_grad=False,
    free_energy_hidden=[200, 200, 200, 200],
    lr=3e-4,
    lr_gamma=0.995,
    lr_step=1000,
    grad_clip=1000.0,
    max_steps=200_000,
    n_steps=125,
    starting_k=1,
    n_anneal=60_000,
    n_mcmc=10,
    batch_size=50,
    loss_batch_size=100,
    buffer_size=100,
    buffer_cycle=100,
    num_gpus=1,
    num_nodes=1,
    log_every=1,
    ckpt_every=10_000,
    last_ckpt_every=100,
    wandb_project="frogs",
)

_RENAMED = {
    "J_1": "J",
    "cond_prior": "prior",
    "prior_kwargs": "prior_net",
    "phase_mix_ar_kwargs": "prior_net",
    "agpd_table_path": "table_path",
    "cuau_table_path": "table_path",
    "encoder_convs_per_scale": "encoder_convs",
    "decoder_convs_per_scale": "decoder_convs",
    "use_loo_context": "loo_context",
    "reparam_metropolis": "metropolis_rates",
    "F_hidden_sizes": "free_energy_hidden",
    "base_lr": "lr",
    "lr_sched_gamma": "lr_gamma",
    "lr_sched_step_size": "lr_step",
    "grad_clip_norm": "grad_clip",
    "n_mcmc_per_net": "n_mcmc",
    "bs_per_gpu": "batch_size",
    "bs_for_loss_per_gpu": "loss_batch_size",
    "max_buffer_size": "buffer_size",
    "log_every_local": "log_every",
}
_PRIOR_NET_KEYS = ("num_layers", "num_channels", "kernel_size")


def _as_dict(x):
    if x is None:
        return {}
    return dict(x) if isinstance(x, dict) else dict(vars(x))


def _translate_legacy(cfg: dict) -> dict:
    out = dict(cfg)
    for old, new in _RENAMED.items():
        if old in out:
            value = out.pop(old)
            out.setdefault(new, value)
    model_class, target = out.pop("model_class", None), out.pop("target", None)
    system = model_class or target
    if system is not None:
        out.setdefault("system", system)
    if out.get("prior") == "ar_direct":
        out["prior"] = "autoregressive"
    if "lattice_shape" not in out and "L" in out:
        L = int(out["L"])
        out["lattice_shape"] = [L, L] if out.get("system") == "ising" else [L, L, L]
    if "delta_t" in out:
        out.setdefault("n_steps", int(round(1.0 / float(out.pop("delta_t")))))
    if "no_transport" in out:
        out.setdefault("transport", not bool(out.pop("no_transport")))
    if out.get("use_buffer") is False:
        out["buffer_size"] = 0
    prior_net = _as_dict(out.get("prior_net"))
    if prior_net.get("local_flip_mode") == "triton" or prior_net.get("triton_sampling"):
        out.setdefault("prior_backend", "systems")
    if "detach_corrector_dH" in out:
        out.setdefault("full_grad", not bool(out.pop("detach_corrector_dH")))
    if "compile_ar" in prior_net:
        out.setdefault("compile_prior", bool(prior_net["compile_ar"]))
    if prior_net:
        out["prior_net"] = {k: prior_net[k] for k in _PRIOR_NET_KEYS if k in prior_net}
    if out.get("prior") == "uniform" and (out.get("beta_0", 0.0) or out.get("mu_0", 0.0)):
        raise ValueError("the uniform-prior path assumes beta_0 = mu_0 = 0")
    for dead in ("L", "path", "J_0", "mu_0", "mu_1", "B_0", "B_1", "beta_0", "beta_1",
                 "in_channels", "arch", "cond_dim", "couple_prior", "use_buffer",
                 "anneal_quadratic", "warm_up", "accelerator", "num_workers", "plot_every",
                 "ce_dedupe", "training_fastpaths",
                 "reuse_step_quantities", "compile_qnet", "compile_qnet_mode",
                 "prior_ckpt", "ar_direct_ckpt", "ckpt_fname", "wandb_name", "results_dir"):
        out.pop(dead, None)
    return out


REQUIRED = ("system", "lattice_shape", "cond_low", "cond_high")


def normalize(config, strict=False) -> argparse.Namespace:
    """Return a new Namespace with defaults filled and legacy keys translated.

    Unknown keys raise with `strict` (YAML configs) and are dropped otherwise (old checkpoints).
    """
    cfg = _translate_legacy(_as_dict(config))
    unknown = sorted(set(cfg) - set(DEFAULTS) - set(REQUIRED))
    if unknown and strict:
        raise ValueError(f"unknown config keys: {unknown}")
    cfg = {k: v for k, v in cfg.items() if k not in unknown}
    table = cfg.get("table_path")
    if table and os.path.isabs(table) and not os.path.exists(table):
        cfg["table_path"] = os.path.join("data", os.path.basename(table))
    for key, value in DEFAULTS.items():
        if cfg.get(key) is None and value is not None:
            cfg[key] = copy.deepcopy(value)
        cfg.setdefault(key, None)
    prior_net = dict(DEFAULTS["prior_net"])
    prior_net.update(_as_dict(cfg["prior_net"]))
    cfg["prior_net"] = prior_net
    if cfg.get("system") not in SYSTEMS:
        raise ValueError(f"system must be one of {SYSTEMS}; got {cfg.get('system')!r}")
    if cfg["prior"] not in ("autoregressive", "uniform"):
        raise ValueError(f"prior must be 'autoregressive' or 'uniform'; got {cfg['prior']!r}")
    if cfg["prior_backend"] not in ("reference", "systems"):
        raise ValueError("prior_backend must be 'reference' or 'systems'")
    if cfg["system"] == "ising" and cfg["prior"] != "autoregressive":
        raise ValueError("the Ising model is implemented with the autoregressive prior only")
    cfg["lattice_shape"] = [int(s) for s in cfg["lattice_shape"]]
    cfg["cond_low"] = [float(v) for v in cfg["cond_low"]]
    cfg["cond_high"] = [float(v) for v in cfg["cond_high"]]
    return argparse.Namespace(**cfg)


def load_config(path: str, overrides=()) -> argparse.Namespace:
    """Load a YAML config; `overrides` are `key=value` strings parsed as YAML."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override must be key=value; got {item!r}")
        cfg[key.strip()] = yaml.safe_load(raw)
    return normalize(cfg, strict=True)


def run_name(config) -> str:
    return config.run_tag or config.system
