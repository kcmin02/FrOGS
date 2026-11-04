"""Mean over gamma of KL(rho_0(. | c) || rho_0(gamma . | gamma c)) on a grid of conditions.

    python scripts/checks/check_symmetry.py --ckpt <run>/ckpts/last.ckpt [--M 2000] [--n-mu 5]
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import torch

from frogs.energy.symmetry import log_prior_gap
from frogs.evaluate import load_model


def tv_bound(kl):
    return min(math.sqrt(max(kl, 0.0) / 2), 1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--M", type=int, default=2000, help="prior samples per condition")
    ap.add_argument("--n-mu", type=int, default=5, help="grid points along dmu (h for Ising)")
    ap.add_argument("--n-T", type=int, default=5)
    ap.add_argument("--mu-lo", type=float)
    ap.add_argument("--mu-hi", type=float)
    ap.add_argument("--T-lo", type=float)
    ap.add_argument("--T-hi", type=float)
    ap.add_argument("--chunk", type=int, default=1000, help="samples per batch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", choices=("reference", "systems"),
                    help="AR prior backend (default: the one the model was trained with)")
    ap.add_argument("--no-compile", action="store_true", help="run the prior eagerly")
    args = ap.parse_args()

    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.ckpt, device, compile=False, backend=args.backend)
    cfg, energy = model.config, model.energy
    if model.prior is None:
        raise SystemExit(f"{args.ckpt} has the uniform prior, which is exactly G-invariant")
    group = energy.symmetry_group(device)
    if not args.no_compile:
        model.prior.compile(grad=False)

    mus = np.linspace(cfg.cond_low[0] if args.mu_lo is None else args.mu_lo,
                      cfg.cond_high[0] if args.mu_hi is None else args.mu_hi, args.n_mu)
    Ts = np.linspace(cfg.cond_low[1] if args.T_lo is None else args.T_lo,
                     cfg.cond_high[1] if args.T_hi is None else args.T_hi, args.n_T)
    name = "h" if cfg.system == "ising" else "dmu"
    print(f"{cfg.system} {tuple(cfg.lattice_shape)}: |G| = {group.size}, M = {args.M}")

    torch.manual_seed(args.seed)
    chunk = min(args.chunk, args.M)
    kls = []
    for mu in mus:
        for T in Ts:
            gaps = []
            for start in range(0, args.M, chunk):
                n = min(chunk, args.M - start)
                c = torch.tensor([[mu, T]], dtype=torch.float32, device=device)
                c = c.expand(n, -1).contiguous()
                with torch.no_grad():
                    gaps.append(log_prior_gap(energy, group, energy.sample_prior(c), c).double())
            d = torch.cat(gaps)
            kl, se = float(d.mean()), float(d.std()) / math.sqrt(d.numel())
            kls.append(kl)
            print(f"  {name}={mu:+.4f}  T={T:.4g}  KL = {kl:.3e} +- {se:.1e}  "
                  f"TV <= {tv_bound(kl):.3e}", flush=True)
    kls = np.array(kls)
    print(f"over the grid: mean KL = {kls.mean():.3e}, max KL = {kls.max():.3e}, "
          f"max TV bound = {tv_bound(kls.max()):.3e}")


if __name__ == "__main__":
    main()
