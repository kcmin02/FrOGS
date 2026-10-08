"""SEGAL baseline sweep for Fig. 3c/3d: importance weights of the released SEGAL network.

    python baselines/segal/sweep.py --system agpd --ckpt $SEGAL/AgPdCE_125_final_submission
    python baselines/segal/sweep.py --system cuau --ckpt $SEGAL/CuAu_128_Final_Submission
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from frogs.analysis.sweep_io import save_sweep
from frogs.constants import K_B
from frogs.energy import DEFAULT_TABLES, ClusterExpansion
from frogs.evaluate import kappa2_log_mean_exp, raw_log_mean_exp
from frogs.paths import RESULTS_DIR, ensure_dir, resolve_data_file
from frogs.sampler import ess
from model import SegalAR

# Architectures and field grids of the SEGAL notebooks, with T refined to 25 K steps.
# `ref_subtract`: the CuAu notebook subtracts the composition-weighted pure-element energies.
SYSTEMS = {
    "agpd": dict(table=DEFAULT_TABLES["agpd_ce"], lattice_shape=(5, 5, 5), num_layers=2,
                 activation="tanh", field=(-0.2, 0.2, 81), T=(200.0, 900.0, 29),
                 ref_subtract=False),
    "cuau": dict(table=DEFAULT_TABLES["cuau_ce"], lattice_shape=(8, 4, 4), num_layers=3,
                 activation="sigmoid", field=(-0.1, 0.075, 36), T=(200.0, 1200.0, 41),
                 ref_subtract=True),
}
CHUNK = 1000  # samples per AR batch; the random stream (and hence the paper sweeps) depends on it


def ase_to_grid(table_path, lattice_shape):
    """Permutation p with sigma_grid = sigma_ase[:, p]; SEGAL orders sites like the CLEASE atoms."""
    prim = np.load(table_path)["prim_coords"].astype(np.int64)
    _, Ly, Lz = lattice_shape
    return torch.as_tensor(np.argsort(prim[:, 0] * Ly * Lz + prim[:, 1] * Lz + prim[:, 2]))


@torch.no_grad()
def sample_condition(model, ce, perm, field, T, dmu, M, device):
    """A = -beta U_CE + beta dmu N_+ - log P_AR and N_+ for M samples at one (field, T)."""
    beta = 1.0 / (K_B * T)
    A, n_plus = [], []
    for start in range(0, M, CHUNK):
        b = min(CHUNK, M - start)
        Tc = torch.full((b,), T, device=device)
        fc = torch.full((b,), field, device=device)
        sigma = model.sample(Tc, fc)
        log_p = model.log_prob(sigma, Tc, fc).double()
        U = ce.energy(sigma[:, perm]).double()
        n = (sigma.sum(dim=1).double() + ce.n_sites) / 2.0
        A.append(-beta * U + beta * dmu * n - log_p)
        n_plus.append(n)
    return torch.cat(A).cpu().numpy(), torch.cat(n_plus).cpu().numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--system", required=True, choices=sorted(SYSTEMS))
    p.add_argument("--ckpt", required=True, help="released SEGAL weights (see README)")
    p.add_argument("--field", type=float, nargs="+", help="SEGAL field grid (default: paper)")
    p.add_argument("--T", type=float, nargs="+", help="temperatures in K (default: paper)")
    p.add_argument("--M", type=int, default=5000, help="samples per condition")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", help="default: results/sweeps/segal_<system>_<N>.npz")
    args = p.parse_args()

    spec = SYSTEMS[args.system]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    table = resolve_data_file(spec["table"])
    ce = ClusterExpansion(table, spec["lattice_shape"]).to(device)
    perm = ase_to_grid(table, spec["lattice_shape"]).to(device)
    N = ce.n_sites
    model = SegalAR.from_checkpoint(args.ckpt, spec["num_layers"], spec["activation"])
    if model.n_sites != N:
        raise SystemExit(f"{args.ckpt} has {model.n_sites} sites, the {args.system} cell {N}")
    model = model.to(device).eval()

    fields = np.asarray(args.field or np.linspace(*spec["field"]), dtype=np.float64)
    temps = np.asarray(args.T or np.linspace(*spec["T"]), dtype=np.float64)
    pure = torch.tensor([-1.0, 1.0], device=device)[:, None].expand(2, N)
    U_minus, U_plus = ce.energy(pure).tolist()
    # The reference subtraction is linear in N_+, i.e. a constant shift of dmu.
    dmu_offset = (U_plus - U_minus) / N if spec["ref_subtract"] else 0.0
    dmu = 2.0 * fields + dmu_offset
    print(f"SEGAL {args.system}: N={N}, {len(fields)} x {len(temps)} conditions, M={args.M}, "
          f"dmu = 2 field {dmu_offset * 1e3:+.3f} meV")

    shape = (len(fields), len(temps))
    A_all = np.zeros(shape + (args.M,))
    N_all = np.zeros(shape + (args.M,), dtype=np.float32)
    out = {k: np.zeros(shape) for k in ("logZ", "logZ_raw", "x", "ess")}
    torch.manual_seed(args.seed)
    t0 = time.time()
    for i, field in enumerate(fields):
        for j, T in enumerate(temps):
            A, n = sample_condition(model, ce, perm, float(field), float(T), float(dmu[i]),
                                    args.M, device)
            A_all[i, j], N_all[i, j] = A, n
            At = torch.as_tensor(A)
            out["logZ"][i, j] = kappa2_log_mean_exp(A)
            out["logZ_raw"][i, j] = raw_log_mean_exp(A)
            out["x"][i, j] = float((torch.softmax(At, dim=0) * torch.as_tensor(n)).sum()) / N
            out["ess"][i, j] = float(ess(At))
        print(f"  dmu={dmu[i]:+.4f}  ESS median {np.median(out['ess'][i]):.3f}  "
              f"x in [{out['x'][i].min():.3f}, {out['x'][i].max():.3f}]  "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)

    path = args.out or os.path.join(ensure_dir(os.path.join(RESULTS_DIR, "sweeps")),
                                    f"segal_{args.system}_{N}.npz")
    meta = dict(system=args.system, sampler="SEGAL", ckpt=os.path.basename(args.ckpt),
                table=os.path.basename(table), lattice_shape=list(spec["lattice_shape"]), N=N,
                M=args.M, seed=args.seed, field=fields.tolist(), dmu_offset=dmu_offset,
                U_minus=U_minus, U_plus=U_plus)
    save_sweep(path, dict(dmu=dmu, T=temps, M=np.full(shape, args.M), A_samples=A_all,
                          N_samples=N_all, **out), meta)
    print(f"ESS median {np.median(out['ess']):.3f}; saved {path}")


if __name__ == "__main__":
    main()
