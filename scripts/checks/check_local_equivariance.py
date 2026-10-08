"""Check g(i | sigma) = -g(i | flip_i sigma) on a trained transport network (§A.7).

    python scripts/checks/check_local_equivariance.py --ckpt <run>/ckpts/last.ckpt

Compares the trained network with a control whose centre masks are removed, and sweeps c over
the training box at fixed (sigma, t). Writes check_local_equivariance.{png,npz} to figures/.
"""
from __future__ import annotations

import argparse
import contextlib
import itertools
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm

from frogs.nets.transport import CenterMaskedConv2d, CenterMaskedConv3d, StateDependentConv
from frogs.paths import figure_path
from frogs.trainer import FrOGSModule

EPS32 = float(np.finfo(np.float32).eps)
TRAINED, CONTROL = "#2a78d6", "#eb6834"


def load_transport(ckpt, device):
    torch.serialization.add_safe_globals([argparse.Namespace])
    model = FrOGSModule.load_from_checkpoint(ckpt, map_location=device)
    if model.transport is None:
        raise SystemExit(f"{ckpt} has no transport network")
    cfg = model.config
    net = model.transport.network.to(device).eval()
    return net, tuple(cfg.lattice_shape), (cfg.cond_low, cfg.cond_high)


@contextlib.contextmanager
def centre_mask_removed(net):
    """Same trained weights, but every layer may read the spin at its own site."""
    masked = (CenterMaskedConv2d, CenterMaskedConv3d)
    masks = [(m, m.mask.clone()) for m in net.modules() if isinstance(m, masked)]
    convs = [(m, m.mid_idx) for m in net.modules() if isinstance(m, StateDependentConv)]
    for m, _ in masks:
        m.mask.fill_(1.0)
    for m, _ in convs:
        m.mid_idx = slice(0, 0)  # an empty slice turns the centre-tap zeroing into a no-op
    try:
        yield
    finally:
        for m, mask in masks:
            m.mask.copy_(mask)
        for m, mid in convs:
            m.mid_idx = mid


def face_neighbours(shape):
    sites = list(itertools.product(*[range(s) for s in shape]))
    flat = {p: i for i, p in enumerate(sites)}
    out = []
    for p in sites:
        nbr = {tuple((p[a] + d) % shape[a] if a == ax else p[a] for a in range(len(shape)))
               for ax in range(len(shape)) for d in (-1, 1)} - {p}
        out.append([flat[q] for q in nbr])
    return out


def random_point(shape, box, device, gen):
    x = 2 * torch.randint(0, 2, (1, *shape), generator=gen, device=device).float() - 1
    t = torch.rand(1, generator=gen, device=device)
    lo, hi = (torch.tensor(b, device=device) for b in box)
    c = lo + (hi - lo) * torch.rand(1, len(box[0]), generator=gen, device=device)
    return x, t, c


def flip_each_site(x):
    """(N, *shape): row s is x with site s flipped."""
    n = x[0].numel()
    xf = x.reshape(1, -1).repeat(n, 1)
    xf[torch.arange(n), torch.arange(n)] *= -1
    return xf.reshape(n, *x.shape[1:])


@torch.no_grad()
def residual_table(net, shape, box, n_configs, device, gen):
    """g(s | x), g(s | flip_s x), and the change of H = g / ((1 + t) x_s) at s and next to s."""
    nbrs = face_neighbours(shape)
    n = len(nbrs)
    out = {k: np.empty((n_configs, n)) for k in ("g", "g_flip", "dH_self", "dH_nbr")}
    for b in range(n_configs):
        x, t, c = random_point(shape, box, device, gen)
        xf = flip_each_site(x)
        g0 = net(x, t, c)[0].reshape(-1).double().cpu().numpy()
        gf = net(xf, t.repeat(n), c.repeat(n, 1)).reshape(n, -1).double().cpu().numpy()
        H0 = g0 / ((1.0 + float(t)) * x.reshape(-1).double().cpu().numpy())
        Hf = gf / ((1.0 + float(t)) * xf.reshape(n, -1).double().cpu().numpy())
        out["g"][b], out["g_flip"][b] = g0, np.diag(gf)
        out["dH_self"][b] = np.abs(np.diag(Hf) - H0)
        out["dH_nbr"][b] = [np.median(np.abs(Hf[s, nbrs[s]] - H0[nbrs[s]])) for s in range(n)]
    out["resid"] = out["g"] + out["g_flip"]
    return out


@torch.no_grad()
def condition_scan(net, shape, box, n_mu, n_T, device, gen):
    """Worst-site relative residual max|r| / max|g| on a (dmu, T) grid at one fixed (x, t)."""
    mus = np.linspace(box[0][0], box[1][0], n_mu)
    Ts = np.linspace(box[0][1], box[1][1], n_T)
    x, t, _ = random_point(shape, box, device, gen)
    xf = flip_each_site(x)
    n = xf.shape[0]
    R = np.zeros((n_T, n_mu))
    for i, T in enumerate(Ts):
        for j, mu in enumerate(mus):
            c = torch.tensor([[mu, T]], device=device).float()
            g0 = net(x, t, c)[0].reshape(-1).double().cpu().numpy()
            gf = net(xf, t.repeat(n), c.repeat(n, 1)).reshape(n, -1).double().cpu().numpy()
            R[i, j] = np.abs(g0 + np.diag(gf)).max() / (np.abs(g0).max() + 1e-30)
    return mus, Ts, R


def plot(trained, control, mus, Ts, R, out):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), constrained_layout=True)

    ax = axes[0]
    for d, color, label in ((control, CONTROL, "centre mask removed"),
                            (trained, TRAINED, "trained network")):
        sig, res = np.abs(d["g"]).ravel(), np.abs(d["resid"]).ravel()
        ax.scatter(sig, np.clip(res, 1e-12, None), s=3, alpha=0.25, color=color,
                   edgecolors="none", label=label, rasterized=True)
    sig = np.abs(trained["g"]).ravel()
    xs = np.array([max(sig[sig > 0].min(), 1e-4), sig.max()])
    ax.plot(xs, EPS32 * xs, color="0.2", lw=1.2, ls="--", label=r"$\epsilon_\mathrm{fp32}\,|g|$")
    ax.set(xscale="log", yscale="log", xlabel=r"$|g(i\,|\,\sigma)|$",
           ylabel=r"$|g(i\,|\,\sigma) + g(i\,|\,\mathrm{flip}_i\,\sigma)|$")
    ax.text(0.98, 0.08, "exact zeros drawn at $10^{-12}$", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=8, color="0.3")
    ax.set_title("(a) flux antisymmetry residual")
    for handle in ax.legend(loc="upper left", fontsize=8, markerscale=3).legend_handles:
        handle.set_alpha(1.0)

    ax = axes[1]
    bins = np.logspace(-11, 2, 70)
    ax.hist(np.clip(trained["dH_self"].ravel(), 1e-11, None), bins=bins, color=TRAINED,
            alpha=0.85, label="at the flipped site $i$")
    ax.hist(np.clip(trained["dH_nbr"].ravel(), 1e-11, None), bins=bins, color=CONTROL,
            alpha=0.7, label=r"at its neighbours $j \neq i$ (median)")
    ax.set(xscale="log", xlabel=r"$|H(\cdot\,|\,\sigma) - H(\cdot\,|\,\mathrm{flip}_i\,\sigma)|$",
           ylabel="site flips")
    ax.text(0.02, 0.72, "exact zeros\ndrawn at $10^{-11}$", transform=ax.transAxes, fontsize=8,
            color="0.3")
    ax.set_title(r"(b) readout $H = g / ((1+t)\,\sigma_i)$ ignores $\sigma_i$")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[2]
    im = ax.pcolormesh(mus, Ts, np.clip(R, 1e-12, None), norm=LogNorm(1e-8, 1e-5),
                       cmap="Blues", shading="auto")
    fig.colorbar(im, ax=ax, label=r"$\max_i |r_i| \,/\, \max_i |g_i|$")
    ax.set(xlabel=r"$\Delta\mu$ (eV)", ylabel="T (K)")
    ax.set_title("(c) over the training box, fixed $(\\sigma, t)$")

    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-configs", type=int, default=64)
    ap.add_argument("--n-mu", type=int, default=21)
    ap.add_argument("--n-T", type=int, default=21)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="figure path; the .npz is written next to it")
    args = ap.parse_args()

    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False  # TF32 convolutions lift the 2D residual to ~1e-4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.Generator(device=device).manual_seed(args.seed)
    net, shape, box = load_transport(args.ckpt, device)
    print(f"lattice {shape}, {sum(p.numel() for p in net.parameters())} parameters, "
          f"box {box}, device {device}")

    trained = residual_table(net, shape, box, args.n_configs, device, gen)
    with centre_mask_removed(net):
        control = residual_table(net, shape, box, args.n_configs, device, gen)
    mus, Ts, R = condition_scan(net, shape, box, args.n_mu, args.n_T, device, gen)

    rel = np.abs(trained["resid"]) / (np.abs(trained["g"]) + 1e-30)
    print(f"trained: max|r| = {np.abs(trained['resid']).max():.2e}, "
          f"median|r| = {np.median(np.abs(trained['resid'])):.2e}, "
          f"median|r|/|g| = {np.median(rel):.2e} (eps_fp32 = {EPS32:.2e})")
    print(f"control: max|r| = {np.abs(control['resid']).max():.2e}, "
          f"median|r| = {np.median(np.abs(control['resid'])):.2e}")
    print(f"readout change: median at flipped site = {np.median(trained['dH_self']):.2e}, "
          f"at neighbours = {np.median(trained['dH_nbr']):.2e}")
    print(f"condition scan: max relative residual over the box = {R.max():.2e}, "
          f"median = {np.median(R):.2e}")

    out = args.out or figure_path("check_local_equivariance.png")
    plot(trained, control, mus, Ts, R, out)
    arrays = {f"{name}_{k}": v for name, d in (("trained", trained), ("control", control))
              for k, v in d.items()}
    np.savez_compressed(os.path.splitext(out)[0] + ".npz", cond_dmu=mus, cond_T=Ts, cond_R=R,
                        **arrays)


if __name__ == "__main__":
    main()
