"""Importance-sampling sweep of a trained model over a (dmu, T) grid (Eq. 2, Eq. 3).

    python scripts/sweep.py --ckpt results/cuau_ce/cuau128-1/ckpts/last.ckpt \
        --n-mu 36 --n-T 41 --M 5000 --store-A --store-sigma --tag cuau128

The grid defaults to the model's training box. Each grid point is written to its own shard as soon
as it finishes, so an interrupted sweep resumes where it stopped. A long sweep can be split over a
job array with --slice I N and reassembled with scripts/merge_sweep.py.
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch

from frogs.analysis.sweep_io import save_sweep
from frogs.evaluate import ISEvaluator, load_model
from frogs.paths import RESULTS_DIR, ensure_dir


POINT_KEYS = ("M", "logZ", "logZ_se", "logZ_raw", "x", "x_se", "ess")
SAMPLE_KEYS = ("A_samples", "N_samples")


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def source_sha256():
    """Hash of the frogs package sources, so shards from different code are never mixed."""
    import frogs
    root = os.path.dirname(frogs.__file__)
    h = hashlib.sha256()
    for base, _, files in sorted(os.walk(root)):
        for name in sorted(f for f in files if f.endswith(".py")):
            path = os.path.join(base, name)
            h.update(os.path.relpath(path, root).encode())
            with open(path, "rb") as f:
                h.update(f.read())
    return h.hexdigest()


def check_manifest(shard_dir, settings):
    """Record the sweep settings next to the shards and refuse to resume with different ones."""
    path = os.path.join(shard_dir, "manifest.json")
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
        if old != settings:
            diff = sorted(k for k in set(old) | set(settings) if old.get(k) != settings.get(k))
            raise SystemExit(f"{shard_dir} holds shards from different settings ({diff}); "
                             "use a new --tag or remove that directory")
    else:
        tmp = f"{path}.{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(settings, f, indent=1, sort_keys=True)
        os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n-mu", type=int, default=36)
    p.add_argument("--n-T", type=int, default=41)
    p.add_argument("--mu-lo", type=float)
    p.add_argument("--mu-hi", type=float)
    p.add_argument("--T-lo", type=float)
    p.add_argument("--T-hi", type=float)
    p.add_argument("--M", type=int, default=5000, help="walkers per grid point")
    p.add_argument("--n-mcmc", type=int, default=0, help="Metropolis moves per CTMC step")
    p.add_argument("--n-steps", type=int, help="CTMC steps (default: the training value)")
    p.add_argument("--chunk", type=int, default=1000, help="walkers per rollout batch")
    p.add_argument("--n-boot", type=int, default=500, help="bootstrap replicates for the SEs")
    p.add_argument("--store-A", action="store_true", help="keep per-walker log-weights and counts")
    p.add_argument("--store-sigma", action="store_true", help="keep bit-packed final configurations")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", choices=("reference", "systems"),
                   help="AR prior backend (default: the one the model was trained with)")
    p.add_argument("--no-compile", action="store_true", help="run the networks eagerly")
    p.add_argument("--slice", type=int, nargs=2, metavar=("I", "N"),
                   help="compute only the I-th of N blocks of dmu columns")
    p.add_argument("--tag", required=True)
    p.add_argument("--out-dir", default=os.path.join(RESULTS_DIR, "sweeps"))
    args = p.parse_args()

    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.ckpt, device, compile=not args.no_compile, backend=args.backend)
    cfg = model.config

    mu_lo = cfg.cond_low[0] if args.mu_lo is None else args.mu_lo
    mu_hi = cfg.cond_high[0] if args.mu_hi is None else args.mu_hi
    T_lo = cfg.cond_low[1] if args.T_lo is None else args.T_lo
    T_hi = cfg.cond_high[1] if args.T_hi is None else args.T_hi
    dmu_full = np.linspace(mu_lo, mu_hi, args.n_mu)
    T_grid = np.linspace(T_lo, T_hi, args.n_T)
    cols = np.arange(args.n_mu)
    out_name = args.tag
    if args.slice:
        i, n = args.slice
        cols = np.array_split(cols, n)[i]
        out_name = f"{args.tag}_slice{i:02d}of{n:02d}"

    ev = ISEvaluator(model, M=args.M, n_steps=args.n_steps, n_mcmc=args.n_mcmc, device=device)
    backend = model.prior.backend if model.prior is not None else None
    settings = dict(
        ckpt_sha256=file_sha256(args.ckpt), source_sha256=source_sha256(),
        dmu=dmu_full.tolist(), T=T_grid.tolist(), M=args.M, n_steps=ev.n_steps,
        n_mcmc=args.n_mcmc, chunk=args.chunk, n_boot=args.n_boot, seed=args.seed,
        store_A=args.store_A, store_sigma=args.store_sigma, backend=backend,
        compile=not args.no_compile, device=device, torch=torch.__version__,
    )
    out_dir = ensure_dir(args.out_dir)
    shard_dir = ensure_dir(os.path.join(out_dir, f"{args.tag}_shards"))
    check_manifest(shard_dir, settings)

    print(f"{cfg.system}: {len(cols)} x {args.n_T} points, M={args.M}, "
          f"n_steps={ev.n_steps}, n_mcmc={args.n_mcmc}, backend={backend}")
    t0 = time.time()
    shards = {}
    for gi in cols:
        for tj in range(args.n_T):
            path = os.path.join(shard_dir, f"pt_g{gi:03d}_t{tj:02d}.npz")
            if not os.path.exists(path):
                ev.seed = args.seed + int(gi) * 100_003 + tj  # independent of how the grid is sliced
                r = ev.run([dmu_full[gi]], [T_grid[tj]], chunk=args.chunk, n_boot=args.n_boot,
                           store_A=args.store_A, store_sigma=args.store_sigma)
                shard = {k: np.asarray(r[k]).reshape(-1)[0] for k in POINT_KEYS}
                for k in SAMPLE_KEYS + ("sigma_packed",):
                    if k in r:
                        shard[k] = r[k][0, 0]
                np.savez(path[:-4] + ".tmp.npz", **shard)
                os.replace(path[:-4] + ".tmp.npz", path)
                print(f"  dmu={dmu_full[gi]:+.4f} T={T_grid[tj]:.4g}  ess={shard['ess']:.3f}  "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)
            shards[gi, tj] = dict(np.load(path))

    out = {}
    for k in shards[cols[0], 0]:
        out[k] = np.stack([np.stack([shards[gi, tj][k] for tj in range(args.n_T)]) for gi in cols])
    out["dmu"], out["T"] = dmu_full[cols], T_grid
    meta = dict(settings, ckpt=os.path.abspath(args.ckpt), system=cfg.system,
                lattice_shape=list(cfg.lattice_shape), N=model.energy.n_sites, prior=cfg.prior,
                transport=model.transport is not None, n_mu=args.n_mu, n_T=args.n_T,
                columns=cols.tolist(), wall_sec=time.time() - t0,
                slice=list(args.slice) if args.slice else None)
    path = os.path.join(out_dir, f"{out_name}.npz")
    save_sweep(path, out, meta)
    print(f"wrote {path}")
    if args.slice:
        print(f"merge with: python scripts/merge_sweep.py --tag {args.tag} --out-dir {out_dir}")


if __name__ == "__main__":
    main()
