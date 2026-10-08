"""2D Ising model on an L x L periodic square lattice (§5.1).

H(sigma) = -J sum_<ij> sigma_i sigma_j - h sum_i sigma_i, with k_B = 1 and c = (h, T).
The 3D autoregressive prior sees the lattice as the degenerate shape (1, L, L).
"""
from __future__ import annotations

import torch

from frogs.energy.base import PathEnergy

NN_OFFSETS = ((1, 0), (-1, 0), (0, 1), (0, -1))


class IsingEnergy(PathEnergy):
    """Learned-prior path U_t = t * H / T - (1 - t) * log rho_0."""

    def __init__(self, L, prior, J=1.0, cond_low=None, cond_high=None):
        super().__init__(prior, cond_low, cond_high)
        self.L = int(L)
        self.J = float(J)
        self.lattice_shape = (self.L, self.L)
        self.n_sites = self.L * self.L
        idx = torch.arange(self.n_sites).reshape(self.L, self.L)
        self.nn_idx = torch.stack(
            [torch.roll(idx, shifts=(-di, -dj), dims=(0, 1)).reshape(-1) for di, dj in NN_OFFSETS],
            dim=1)

    @property
    def prior_shape(self):
        return (1, self.L, self.L)

    def to(self, device):
        self.nn_idx = self.nn_idx.to(device)
        return super().to(device)

    @staticmethod
    def _neighbor_sum(sigma):
        s = torch.zeros_like(sigma)
        for di, dj in NN_OFFSETS:
            s = s + torch.roll(sigma, shifts=(-di, -dj), dims=(1, 2))
        return s

    def hamiltonian(self, sigma, h):
        """H(sigma) for a batch (B, L, L) -> (B,)."""
        bond = (sigma * self._neighbor_sum(sigma)).flatten(1).sum(dim=1) * 0.5
        return -self.J * bond - h * sigma.flatten(1).sum(dim=1)

    def _flip_dH_all(self, sigma, h):
        nn = self._neighbor_sum(sigma)
        return (2.0 * self.J * sigma * nn + 2.0 * h.view(-1, 1, 1) * sigma).flatten(1)

    def Ut(self, sigma, t, c, need_dU=True):
        h, beta = c[..., 0], 1.0 / c[..., 1]
        cn = self.norm_c(c)
        x = sigma.unsqueeze(1)
        S = self.prior.log_prob(x, cn)
        U = t * beta * self.hamiltonian(sigma, h) - (1.0 - t) * S
        if not need_dU:
            return U, None
        dH = self._flip_dH_all(sigma, h)
        dS = self.prior.delta_log_prob_all_sites(x, cn, base_log_prob=S)
        dU = (t * beta).unsqueeze(1) * dH - (1.0 - t).unsqueeze(1) * dS
        return U, dU.reshape(-1, *self.lattice_shape)

    def dtUt(self, sigma, t, c):
        h, beta = c[..., 0], 1.0 / c[..., 1]
        return beta * self.hamiltonian(sigma, h) + self.prior.log_prob(sigma.unsqueeze(1), self.norm_c(c))

    def path_increment(self, sigma, t, dt, c):
        return ((t + dt) - t) * self.dtUt(sigma, t, c)

    def mcmc_state(self, sigma_flat, t, c):
        return self.log_prior(sigma_flat, c)

    def mcmc_delta(self, sigma_flat, t, c, site, state):
        spins = sigma_flat[:, site]
        nnsum = sigma_flat[:, self.nn_idx[site]].sum(dim=-1)
        dH = 2.0 * self.J * spins * nnsum + 2.0 * c[..., 0] * spins
        flipped = sigma_flat.clone()
        flipped[:, site] = -flipped[:, site]
        lp_flip = self.log_prior(flipped, c)
        return t * (1.0 / c[..., 1]) * dH - (1.0 - t) * (lp_flip - state), lp_flip
