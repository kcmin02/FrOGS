"""Symmetry group of the target: the maps gamma that leave beta H_c invariant up to a constant."""
from __future__ import annotations

import itertools

import numpy as np
import torch


def lattice_point_ops(lattice_vectors):
    """Integer matrices with entries in {-1, 0, 1} that preserve the lattice metric."""
    A = np.asarray(lattice_vectors, dtype=np.float64)
    dim = len(A)
    G = A @ A.T
    M = np.array(list(itertools.product((-1, 0, 1), repeat=dim * dim)), dtype=np.int64)
    M = M.reshape(-1, dim, dim)
    M = M[np.abs(np.rint(np.linalg.det(M))) == 1]
    err = np.abs(np.einsum("nji,jk,nkl->nil", M, G, M) - G).max(axis=(1, 2))
    return list(M[err <= 1e-9 * np.abs(G).max()])


def preserves_supercell(M, lattice_shape):
    shape = np.asarray(lattice_shape, dtype=np.int64)
    return bool(np.all((np.asarray(M) * shape[None, :]) % shape[:, None] == 0))


def _site_coords(shape):
    grids = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    return np.stack(grids, axis=-1).reshape(-1, len(shape))


def _point_perm(M, shape):
    return np.ravel_multi_index(((_site_coords(shape) @ np.asarray(M).T) % shape).T, shape)


def _translation_perm(r, shape):
    return np.ravel_multi_index(((_site_coords(shape) + r) % shape).T, shape)


class SymmetryGroup:
    """(point operation, spin sign) pairs x translations; sign -1 also negates c[field]."""

    def __init__(self, lattice_shape, ops, field=None, translations=True):
        shape = tuple(int(s) for s in lattice_shape)
        if any(not preserves_supercell(M, shape) for M, _ in ops):
            raise ValueError("a point operation does not map the supercell onto itself")
        sites = _site_coords(shape) if translations else _site_coords(shape)[:1]
        pairs = np.unique(np.stack([np.append(_point_perm(M, shape), -s) for M, s in ops]), axis=0)
        self.lattice_shape = shape
        self.n_sites = int(np.prod(shape))
        self.point = torch.as_tensor(pairs[:, :-1], dtype=torch.int64)
        self.sign = torch.as_tensor(-pairs[:, -1], dtype=torch.int64)
        self.translation = torch.as_tensor(np.stack([_translation_perm(r, shape) for r in sites]),
                                           dtype=torch.int64)
        self.field = field
        self.flips = bool((self.sign < 0).any())

    @property
    def size(self) -> int:
        return len(self.point) * len(self.translation)

    def to(self, device):
        self.point = self.point.to(device)
        self.sign = self.sign.to(device)
        self.translation = self.translation.to(device)
        return self

    def sample(self, n, device=None):
        q = torch.randint(len(self.point), (n,), device=device)
        r = torch.randint(len(self.translation), (n,), device=device)
        return q, r

    def apply(self, g, sigma, c):
        if self.point.device != sigma.device:
            self.to(sigma.device)
        q, r = g
        idx = self.point[q].gather(1, self.translation[r])
        out = sigma.reshape(sigma.shape[0], -1).gather(1, idx).reshape(sigma.shape)
        if not self.flips:
            return out, c
        s = self.sign[q]
        out = out * s.view(-1, *(1,) * (sigma.dim() - 1)).to(out.dtype)
        if self.field is not None:
            c = c.clone()
            c[:, self.field] = c[:, self.field] * s.to(c.dtype)
        return out, c


def _beta_H(energy, sigma, c):
    return energy.Ut(sigma, sigma.new_ones(sigma.shape[0]), c, need_dU=False)[0].double()


@torch.no_grad()
def detect_symmetry(energy, device=None, n_configs=8, n_conds=4, rtol=1e-4, seed=0):
    """Point operations, spin flips and translations that leave beta * H_c invariant on the box."""
    if energy.cond_low is None:
        raise ValueError("symmetry detection samples conditions from the training box")
    shape = tuple(int(s) for s in energy.lattice_shape)
    field = energy.field_index
    lo, hi = np.asarray(energy.cond_low), np.asarray(energy.cond_high)
    signs = [1]
    if field is None or abs(lo[field] + hi[field]) <= 1e-9 * (hi[field] - lo[field]):
        signs.append(-1)

    gen = torch.Generator().manual_seed(seed)
    B, dtype = n_conds * n_configs, torch.get_default_dtype()
    frac = torch.rand(B, 1, generator=gen, dtype=torch.float64)
    sigma = torch.where(torch.rand(B, int(np.prod(shape)), generator=gen,
                                   dtype=torch.float64) < frac, 1.0, -1.0)
    c = torch.as_tensor(lo) + torch.as_tensor(hi - lo) * torch.rand(
        n_conds, len(lo), generator=gen, dtype=torch.float64)
    sigma = sigma.to(dtype=dtype, device=device)
    c = c.repeat_interleave(n_configs, dim=0).to(dtype=dtype, device=device)
    base = _beta_H(energy, sigma.reshape(-1, *shape), c).view(n_conds, n_configs)
    tol = rtol * base.std(dim=1)

    def invariant(perm, s):
        g_sigma = s * sigma[:, torch.as_tensor(perm, device=sigma.device)]
        g_c = c.clone()
        if s < 0 and field is not None:
            g_c[:, field] = -g_c[:, field]
        diff = _beta_H(energy, g_sigma.reshape(-1, *shape), g_c).view(n_conds, n_configs) - base
        return bool(((diff.max(dim=1).values - diff.min(dim=1).values) <= tol).all())

    units = np.eye(len(shape), dtype=np.int64)
    translations = all(invariant(_translation_perm(r, shape), 1) for r in units)
    ops = [(M, s) for M in lattice_point_ops(energy.lattice_vectors)
           if preserves_supercell(M, shape) for s in signs]
    ops = [(M, s) for M, s in ops if invariant(_point_perm(M, shape), s)]
    return SymmetryGroup(shape, ops, field, translations).to(device or "cpu")


def log_prior_gap(energy, group, sigma, c):
    """log rho_0(sigma | c) - log rho_0(gamma sigma | gamma c) for one uniform gamma per row."""
    if not energy.learned_prior:
        return sigma.new_zeros(sigma.shape[0])
    g_sigma, g_c = group.apply(group.sample(sigma.shape[0], sigma.device), sigma, c)
    return energy.log_prior(sigma, c) - energy.log_prior(g_sigma, g_c)
