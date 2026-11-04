"""Exactness of the learned-prior path: Z_0 = 1, U_1 = beta * H_c, telescoping log-weights.

Z_0 is summed over all 2^16 configurations of the 16-site CuAu cell; the 128-site cell
re-checks U_1 and the gradient coupling to the prior (§A.5). Exits non-zero if any check fails.
"""
from __future__ import annotations

import itertools
import sys

import torch

from frogs.constants import K_B
from frogs.energy import AlloyEnergy
from frogs.nets import ARConvPrior
from frogs.paths import data_file

COND_LOW, COND_HIGH = [-0.05339, 200.0], [0.104386, 900.0]


def build(table, shape):
    prior = ARConvPrior(shape, num_layers=3, num_channels=8, kernel_size=3)
    return AlloyEnergy(data_file(table), shape, prior=prior, cond_low=COND_LOW, cond_high=COND_HIGH)


def bare_target(energy, sigma, c):
    s = sigma.reshape(sigma.shape[0], -1)
    n_plus = ((s + 1.0) * 0.5).sum(dim=1)
    return (1.0 / (K_B * c[:, 1])) * (energy.ce.energy(s) - c[:, 0] * n_plus)


def report(name, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok


@torch.no_grad()
def check_enumerable():
    print("=== 16-site 2x2x4 cell (enumerated) ===")
    shape = (2, 2, 4)
    e = build("cuau_cluster_tables_2x2x4.npz", shape)
    cfgs = torch.tensor(list(itertools.product([-1.0, 1.0], repeat=16))).reshape(-1, *shape)
    B = cfgs.shape[0]
    c = torch.tensor([[0.02, 500.0]]).expand(B, -1).contiguous()
    U0, _ = e.Ut(cfgs, torch.zeros(B), c, need_dU=False)
    U1, _ = e.Ut(cfgs, torch.ones(B), c, need_dU=False)
    ok = True

    Z0 = torch.exp(-U0).sum().item()
    ok &= report("Z_0 = 1", abs(Z0 - 1.0) < 1e-6, f"Z_0 = {Z0:.10f}")
    err = (U1 - bare_target(e, cfgs, c)).abs().max().item()
    ok &= report("U_1 = beta (U_CE - dmu N_+)", err < 1e-7, f"max|err| = {err:.2e}")
    err = (U0 + e.prior.log_prob(cfgs, e.norm_c(c))).abs().max().item()
    ok &= report("U_0 = -log rho_0", err < 1e-7, f"max|err| = {err:.2e}")

    ts = torch.cat([torch.zeros(1), torch.sort(torch.rand(9)).values, torch.ones(1)])
    idx = torch.randperm(B)[:2048]
    x, cx = cfgs[idx], c[idx]
    n = x.shape[0]
    total = torch.zeros(n)
    for t0, t1 in zip(ts[:-1], ts[1:]):
        total += e.path_increment(x, t0.expand(n), (t1 - t0).expand(n), cx)
    U0x, _ = e.Ut(x, torch.zeros(n), cx, need_dU=False)
    U1x, _ = e.Ut(x, torch.ones(n), cx, need_dU=False)
    err = (total - (U1x - U0x)).abs().max().item()
    ok &= report("path increments telescope to U_1 - U_0", err < 1e-7, f"max|err| = {err:.2e}")

    t, h = torch.full((n,), 0.37), 1e-5
    dtU = e.dtUt(x, t, cx)
    fd = (e.Ut(x, t + h, cx, need_dU=False)[0] - e.Ut(x, t - h, cx, need_dU=False)[0]) / (2 * h)
    err = (dtU - fd).abs().max().item()
    drift = (dtU - e.dtUt(x, torch.full_like(t, 0.81), cx)).abs().max().item()
    ok &= report("dtUt = d/dt U_t, constant in t", err < 1e-4 and drift < 1e-9,
                 f"max|err vs FD| = {err:.2e}, t-drift = {drift:.2e}")

    U, dU = e.Ut(x, t, cx)
    dU, flat = dU.reshape(n, -1), x.reshape(n, -1)
    state = e.mcmc_state(flat, t, cx)
    for site in (0, 5, 15):
        flipped = flat.clone()
        flipped[:, site] = -flipped[:, site]
        direct = e.Ut(flipped.reshape(n, *shape), t, cx, need_dU=False)[0] - U
        mcmc = e.mcmc_delta(flat, t, cx, site, state)[0]
        err_field = (dU[:, site] - direct).abs().max().item()
        err_mcmc = (mcmc - direct).abs().max().item()
        ok &= report(f"flip energies, site {site}", max(err_field, err_mcmc) < 1e-6,
                     f"Ut field vs flip = {err_field:.2e}, Metropolis vs flip = {err_mcmc:.2e}")

    s = e.sample_prior(c[:16])
    values = sorted(set(torch.unique(s).tolist()))
    ok &= report("sample_prior", tuple(s.shape) == (16, *shape) and values == [-1.0, 1.0],
                 f"shape = {tuple(s.shape)}, values = {values}")
    return ok


def check_128_site_cell():
    print("=== 128-site 8x4x4 cell ===")
    e = build("cuau_cluster_tables_8x4x4.npz", (8, 4, 4))
    B = 12
    c = torch.stack([torch.empty(B).uniform_(COND_LOW[0], COND_HIGH[0]),
                     torch.empty(B).uniform_(COND_LOW[1], COND_HIGH[1])], dim=1)
    with torch.no_grad():
        x = e.sample_prior(c)
        U1, _ = e.Ut(x, torch.ones(B), c, need_dU=False)
    err = (U1 - bare_target(e, x, c)).abs().max().item()
    ok = report("U_1 = beta (U_CE - dmu N_+)", err < 1e-6, f"max|err| = {err:.2e}")

    (e.dtUt(x, torch.full((B,), 0.5), c) ** 2).mean().backward()
    params = list(e.prior.parameters())
    with_grad = [p for p in params if p.grad is not None]
    gnorm = sum(p.grad.norm().item() ** 2 for p in with_grad) ** 0.5
    ok &= report("gradients reach the prior through dtUt",
                 gnorm > 0 and len(with_grad) == len(params),
                 f"grad norm = {gnorm:.3e}, {len(with_grad)}/{len(params)} parameters")
    return ok


def main():
    torch.manual_seed(0)
    # Exact identities checked to ~1e-7 need fp64; tables take the default dtype at construction.
    torch.set_default_dtype(torch.float64)
    ok = check_enumerable()
    ok &= check_128_site_cell()
    print("\nALL PASS" if ok else "\nSOME FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
