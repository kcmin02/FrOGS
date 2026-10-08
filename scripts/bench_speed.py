"""Time the prior backends: AR sampling, all-site flip scoring, a rollout and a loss/backward step.

    python scripts/bench_speed.py --ckpt <run>/ckpts/last.ckpt
    python scripts/bench_speed.py --config configs/cuau_128.yml --compile

Each operation runs with `prior.backend = "reference"` and then `"systems"` on the same model,
inputs and replay buffer; timings are warmed means over `--reps` CUDA-synchronized repetitions.
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch

from frogs.config import load_config, normalize
from frogs.sampler import CTMCSampler
from frogs.trainer import FrOGSModule

BACKENDS = ("reference", "systems")


def load_model(args):
    torch.serialization.add_safe_globals([argparse.Namespace])
    if args.ckpt:
        saved = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        cfg = normalize(saved["hyper_parameters"]["config"])
    else:
        cfg = load_config(args.config)
    if cfg.prior != "autoregressive":
        raise SystemExit("the prior backends need the autoregressive prior")
    cfg.batch_size = cfg.loss_batch_size = args.batch
    cfg.n_mcmc = args.n_mcmc
    cfg.buffer_size, cfg.buffer_cycle, cfg.log_every = 1, 10 ** 9, 10 ** 9
    if args.ckpt:
        model = FrOGSModule.load_from_checkpoint(args.ckpt, map_location="cpu", config=cfg)
    else:
        model = FrOGSModule(cfg)
    model = model.cuda()
    model.energy.to("cuda")
    if args.compile:
        model.prior.compile()  # as at the start of training
    model.ks = [args.steps]  # every training step uses an `--steps`-step trajectory
    return model


def timed(fn, warmup, reps, grad=False):
    with torch.set_grad_enabled(grad):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(reps):
            start = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
    return 1e3 * statistics.mean(times), 1e3 * (statistics.stdev(times) if reps > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt")
    src.add_argument("--config")
    ap.add_argument("--batch", type=int, default=16, help="walkers, and the loss batch")
    ap.add_argument("--steps", type=int, default=8, help="CTMC steps of the timed rollouts")
    ap.add_argument("--n-mcmc", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--compile", action="store_true", help="torch.compile the prior's logits")
    ap.add_argument("--seed", type=int, default=814)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("the systems backend needs a CUDA GPU")

    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    model = load_model(args)
    energy, prior = model.energy, model.prior
    lo, hi = torch.tensor(model.config.cond_low), torch.tensor(model.config.cond_high)
    c = (0.5 * (lo + hi)).cuda().expand(args.batch, -1).contiguous()
    cn = energy.norm_c(c)
    sampler = CTMCSampler(energy, torch.linspace(0.0, 1.0, args.steps + 1).cuda(),
                          model.transport, n_mcmc=args.n_mcmc)
    with torch.no_grad():
        sigma = energy.sample_prior(c)
    model.training_step(None, 0)  # fills the one-entry replay buffer

    def loss_backward():
        model.zero_grad(set_to_none=True)
        model.training_step(None, 0).backward()

    ops = {
        "AR sampling": (lambda: energy.sample_prior(c), False),
        "all-site flip scoring": (lambda: prior.delta_log_prob_all_sites(sigma, cn), False),
        f"rollout, {args.steps} steps, incl. sampling": (
            lambda: sampler.rollout(energy.sample_prior(c), c), False),
        "loss + backward (replayed trajectory)": (loss_backward, True),
    }
    results = {}
    for backend in BACKENDS:
        prior.backend = backend
        for name, (fn, grad) in ops.items():
            results[name, backend] = timed(fn, args.warmup, args.reps, grad)
    prior.backend = "reference"

    print(f"{torch.cuda.get_device_name()}, lattice {tuple(model.config.lattice_shape)}, "
          f"batch {args.batch}, n_mcmc {args.n_mcmc}, compiled prior: {args.compile}, "
          f"{args.warmup} warmup + {args.reps} timed repetitions")
    print(f"{'operation':40s} {'reference (ms)':>16s} {'systems (ms)':>16s} {'speedup':>8s}")
    for name in ops:
        (ref, ref_sd), (sys_, sys_sd) = results[name, "reference"], results[name, "systems"]
        print(f"{name:40s} {ref:9.2f} ± {ref_sd:4.1f} {sys_:9.2f} ± {sys_sd:4.1f} "
              f"{ref / sys_:7.2f}x")
    print(f"peak GPU memory allocated: {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB")


if __name__ == "__main__":
    main()
