"""Semi-grand-canonical binary fcc alloy: H_c(sigma) = U_CE(sigma) - dmu * N_+(sigma) (Eq. 11).

N_+ counts sites with sigma = +1 (Ag in AgPd, Au in CuAu); c = (dmu [eV], T [K]).
"""
from __future__ import annotations

import torch
from torch.func import vmap

from frogs.constants import K_B
from frogs.energy.base import PathEnergy
from frogs.energy.cluster_expansion import ClusterExpansion


class AlloyEnergy(PathEnergy):
    """Annealing path for an fcc cluster expansion.

    With a learned prior: U_t = t * beta * H_c - (1 - t) * log rho_0.
    With the uniform prior (the LEAPS ablation): U_t = beta(t) * (U_CE - mu(t) * N_+) with
    beta(t) = t * beta, mu(t) = t * dmu, so rho_0 is uniform on {-1, +1}^N.
    """

    field_index = 0  # dmu

    def __init__(self, table_path, lattice_shape, prior=None, cond_low=None, cond_high=None):
        super().__init__(prior, cond_low, cond_high)
        self.ce = ClusterExpansion(table_path, lattice_shape)
        self.lattice_shape = self.ce.lattice_shape
        self.lattice_vectors = self.ce.prim_vectors
        self.n_sites = self.ce.n_sites

    def to(self, device):
        self.ce.to(device)
        return super().to(device)

    # -- uniform-prior path ----------------------------------------------------------------
    def path_coeffs(self, t, c):
        """(alpha, b) with U_t = alpha * U_CE - b * N_+ on the uniform-prior path."""
        beta = t * (1.0 / (K_B * c[..., 1]))
        return beta, beta * (t * c[..., 0])

    def _single_Ut_uniform(self, cfg, t, c):
        alpha, b = self.path_coeffs(t, c)
        s = cfg.reshape(-1)
        U = alpha * self.ce.energy_single(s) - b * ((s + 1.0) * 0.5).sum()
        dU = alpha * self.ce.flip_deltas_single(s) + b * s
        return U, dU.reshape(*self.lattice_shape)

    def _dtUt_uniform(self, sigma, t, c):
        s = sigma.reshape(sigma.shape[0], -1)
        U_CE = vmap(self.ce.energy_single)(s)
        n_plus = ((s + 1.0) * 0.5).sum(dim=1)
        beta, mu = t * (1.0 / (K_B * c[..., 1])), t * c[..., 0]
        d_beta, d_mu = 1.0 / (K_B * c[..., 1]), c[..., 0]
        return d_beta * U_CE - (d_beta * mu + beta * d_mu) * n_plus

    # -- path interface --------------------------------------------------------------------
    def _target(self, sigma, c):
        s = sigma.reshape(sigma.shape[0], -1)
        beta = 1.0 / (K_B * c[..., 1])
        return s, beta, ((s + 1.0) * 0.5).sum(dim=1), self.ce.energy(s)

    def Ut(self, sigma, t, c, need_dU=True):
        """(U_t, per-site flip differences Delta_i U_t of shape (B, *lattice_shape))."""
        if not self.learned_prior:
            return vmap(self._single_Ut_uniform, randomness="different")(sigma, t, c)
        s, beta, n_plus, U_CE = self._target(sigma, c)
        alpha, field = t * beta, t * beta * c[..., 0]
        cn = self.norm_c(c)
        S = self.prior.log_prob(sigma, cn)
        U = alpha * U_CE - field * n_plus - (1.0 - t) * S
        if not need_dU:
            return U, None
        dU_CE = vmap(self.ce.flip_deltas_single)(s)
        dS = self.prior.delta_log_prob_all_sites(sigma, cn, base_log_prob=S)
        dU = alpha.unsqueeze(1) * dU_CE + field.unsqueeze(1) * s - (1.0 - t).unsqueeze(1) * dS
        return U, dU.reshape(sigma.shape[0], *self.lattice_shape)

    def dtUt(self, sigma, t, c):
        """d U_t / dt, shape (B,)."""
        if not self.learned_prior:
            return self._dtUt_uniform(sigma, t, c)
        _, beta, n_plus, U_CE = self._target(sigma, c)
        return beta * (U_CE - c[..., 0] * n_plus) + self.prior.log_prob(sigma, self.norm_c(c))

    def path_increment(self, sigma, t, dt, c):
        """U_{t+dt}(sigma) - U_t(sigma), the path part of the log-weight increment."""
        if not self.learned_prior:
            return dt * self._dtUt_uniform(sigma, t, c)
        return ((t + dt) - t) * self.dtUt(sigma, t, c)

    def step_terms(self, sigma, t, dt, c):
        """(U_t, Delta U_t, path increment) sharing the CE energy and the base log-prob."""
        s, beta, n_plus, U_CE = self._target(sigma, c)
        alpha, field = t * beta, t * beta * c[..., 0]
        cn = self.norm_c(c)
        S = self.prior.log_prob(sigma, cn)
        U = alpha * U_CE - field * n_plus - (1.0 - t) * S
        dU_CE = vmap(self.ce.flip_deltas_single)(s)
        dS = self.prior.delta_log_prob_all_sites(sigma, cn, base_log_prob=S)
        dU = alpha.unsqueeze(1) * dU_CE + field.unsqueeze(1) * s - (1.0 - t).unsqueeze(1) * dS
        increment = ((t + dt) - t) * (beta * (U_CE - c[..., 0] * n_plus) + S)
        return U, dU.reshape(sigma.shape[0], *self.lattice_shape), increment

    # -- single-site Metropolis ------------------------------------------------------------
    def mcmc_state(self, sigma_flat, t, c):
        """Per-step state for `mcmc_delta`: path coefficients, or log rho_0 of the current sigma."""
        if not self.learned_prior:
            return self.path_coeffs(t, c)
        return self.log_prior(sigma_flat, c)

    def mcmc_delta(self, sigma_flat, t, c, site, state):
        """(Delta U_t for flipping `site`, log rho_0 of the flipped sigma or None)."""
        dU_CE = self.ce.flip_delta_at(sigma_flat, site)
        spins = sigma_flat[:, site]
        if not self.learned_prior:
            alpha, b = state
            return alpha * dU_CE - b * (-spins), None
        beta = 1.0 / (K_B * c[..., 1])
        alpha, field = t * beta, t * beta * c[..., 0]
        flipped = sigma_flat.clone()
        flipped[:, site] = -flipped[:, site]
        lp_flip = self.log_prior(flipped, c)
        return alpha * dU_CE + field * spins - (1.0 - t) * (lp_flip - state), lp_flip
