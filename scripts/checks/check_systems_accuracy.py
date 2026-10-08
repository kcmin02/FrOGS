"""Accuracy of the fused prior backend (`prior_backend: systems`) against FP64, on trained weights.

    python scripts/checks/check_systems_accuracy.py --ckpt <run>/ckpts/last.ckpt [--shape 8 8 8]

At three normalized conditions, every sampling conditional is evaluated on one fixed reference
trajectory, and all-site flip scores on its final sample; both backends are compared with an FP64
reference. Also checks that the graph-replayed sampler reproduces the reference draws and CUDA RNG.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from unittest import mock

import torch

from frogs.nets import ARConvPrior
from frogs.trainer import FrOGSModule

CONDITIONS = ((-1.0, -1.0), (0.0, 0.0), (1.0, 1.0))


def load_prior(ckpt, shape=None):
    torch.serialization.add_safe_globals([argparse.Namespace])
    model = FrOGSModule.load_from_checkpoint(ckpt, map_location="cpu")
    if model.prior is None:
        raise SystemExit(f"{ckpt} has no autoregressive prior")
    net = model.config.prior_net
    prior = ARConvPrior(shape or model.prior.lattice_shape, num_layers=net["num_layers"],
                        num_channels=net["num_channels"], kernel_size=net["kernel_size"])
    prior.load_state_dict(model.prior.state_dict())
    return prior.cuda().eval()


@contextmanager
def bernoulli_probabilities(force=None):
    """Record every probability passed to torch.bernoulli; with `force` (B, N), return its bits."""
    probs, real = [], torch.bernoulli

    def draw(p):
        probs.append(p.clone())
        if force is None:
            return real(p)
        return (force[:, len(probs) - 1] + 1.0) * 0.5

    with mock.patch("torch.bernoulli", draw):
        yield probs


def max_error(actual, reference):
    return (actual.double() - reference.double()).abs().max().item()


@torch.no_grad()
def check(prior, c, seed):
    from frogs.nets.systems.triton_kernels import sample_triton
    batch = c.shape[0]
    prior64 = copy.deepcopy(prior).double()
    prior.backend = "reference"
    torch.manual_seed(seed)
    with bernoulli_probabilities() as recorded:
        sample = prior.sample(c, batch)
    rng = torch.cuda.get_rng_state()
    p_ref = torch.stack(recorded)
    bits = sample.flatten(1)
    with bernoulli_probabilities(bits) as recorded:
        sample_triton(prior, c, batch)
    p_sys = torch.stack(recorded)
    with bernoulli_probabilities(bits.double()) as recorded:
        prior64.sample(c.double(), batch)
    p_64 = torch.stack(recorded)

    flips_ref = prior.delta_log_prob_all_sites_reference(sample, c)
    flips_64 = prior64.delta_log_prob_all_sites_reference(sample.double(), c.double())
    prior.backend = "systems"
    flips_sys = prior.delta_log_prob_all_sites(sample, c)
    torch.manual_seed(seed)
    replayed = prior.sample(c, batch)
    rng_equal = torch.equal(torch.cuda.get_rng_state(), rng)
    prior.backend = "reference"
    return dict(prob_sys=max_error(p_sys, p_64), prob_ref=max_error(p_ref, p_64),
                flip_sys=max_error(flips_sys, flips_64), flip_ref=max_error(flips_ref, flips_64),
                sample_equal=torch.equal(replayed, sample), rng_equal=rng_equal)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--shape", type=int, nargs=3, default=None,
                    help="apply the trained weights on another lattice, e.g. 8 8 8")
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--seed", type=int, default=814)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("the systems backend check needs a CUDA GPU")

    prior = load_prior(args.ckpt, args.shape)
    print(f"lattice {prior.lattice_shape}, {prior.num_channels} channels, batch {args.batch}, "
          f"{torch.cuda.get_device_name()}")
    print(f"{'condition':>12} | {'max conditional-prob. error':^27} | {'max flip-score error':^23} "
          f"| graph sample | RNG")
    print(f"{'':>12} | {'systems':>12} {'reference':>14} | {'systems':>10} {'reference':>12} |")
    worst = dict(prob_sys=0.0, flip_sys=0.0)
    replay_ok = True
    for cond in CONDITIONS:
        c = torch.tensor(cond, device="cuda").expand(args.batch, -1).contiguous()
        r = check(prior, c, args.seed)
        print(f"{str(cond):>12} | {r['prob_sys']:12.2e} {r['prob_ref']:14.2e} | "
              f"{r['flip_sys']:10.2e} {r['flip_ref']:12.2e} | {str(r['sample_equal']):>12} | "
              f"{r['rng_equal']}")
        worst = {k: max(v, r[k]) for k, v in worst.items()}
        replay_ok &= r["sample_equal"] and r["rng_equal"]
    print(f"worst systems error vs FP64: conditional probability {worst['prob_sys']:.2e}, "
          f"flip score {worst['flip_sys']:.2e}; graph replay reproduces samples and RNG: "
          f"{replay_ok}")


if __name__ == "__main__":
    main()
