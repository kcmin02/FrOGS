"""CTMC transport with discrete Jarzynski log-weights (§3, §A.2).

Walkers born from rho_0 are moved by the learned rate matrix along U_t and carry a log-weight
A = int_0^1 K_t rho_t(X_t) dt (Eq. 16), so that E[e^A] = Z_1 / Z_0 for any rate matrix (Eq. 17).
Optional single-site Metropolis moves preserve rho_t after each transport step.
"""
from __future__ import annotations

import numpy as np
import torch


def ess(A: torch.Tensor) -> torch.Tensor:
    """Normalized effective sample size (sum e^A)^2 / (M sum e^2A), Eq. (1)."""
    if A.numel() == 0:
        return A.new_zeros(())
    w = A - A.max()
    return torch.mean(torch.exp(w)) ** 2 / torch.mean(torch.exp(2 * w))


class CTMCSampler:
    """Euler discretization of the transport on the grid `ts` (t_0 = 0, ..., t_k = 1)."""

    def __init__(self, energy, ts, transport=None, n_mcmc=0, n_save=1):
        self.energy = energy
        self.ts = ts
        self.transport = transport
        self.n_mcmc = int(n_mcmc)
        n_step = len(ts) - 1
        if n_step % n_save:
            raise ValueError(f"{n_step} steps are not divisible by n_save={n_save}")
        self.n_save = n_save
        self.save_every = n_step // n_save

    def _reuse_q(self, sigma):
        # Compiled CUDA graphs are not bitwise stable across the two calls, so keep them separate.
        return not (sigma.is_cuda and self.transport.compiled)

    def _step(self, sigma, A, t, dt, c):
        """Advance A and sigma by one step; A is evaluated before sigma is mutated in place."""
        E, Q = self.energy, self.transport
        if Q is None:
            return sigma, A - E.path_increment(sigma, t, dt, c)
        with torch.no_grad():
            if E.learned_prior and Q.metropolis_rates:
                _, dU, increment = E.step_terms(sigma, t, dt, c)
            else:
                _, dU = E.Ut(sigma, t, c)
                increment = None
            g = Q(sigma, t, c) if self._reuse_q(sigma) else None
            corrector = Q.corrector(sigma, t, dU, c, g=g)
        if increment is None:
            increment = E.path_increment(sigma, t, dt, c)
        A = A - increment - dt * corrector
        with torch.no_grad():
            sigma = Q.step(sigma, t, dt, c, dU=dU, g=g)
        return sigma, A

    def _metropolis(self, sigma, t, c):
        """`n_mcmc` single-site Metropolis moves at fixed (t, c), one random site per move."""
        if self.n_mcmc == 0:
            return sigma
        E = self.energy
        shape = sigma.shape[1:]
        flat = sigma.view(sigma.shape[0], -1)
        state = E.mcmc_state(flat, t, c)
        for _ in range(self.n_mcmc):
            idx = [int(np.random.randint(0, n, ())) for n in shape]
            site = int(np.ravel_multi_index(idx, tuple(shape)))
            spins = flat[:, site]
            dU, lp_flip = E.mcmc_delta(flat, t, c, site, state)
            accept = torch.rand(flat.shape[0], device=flat.device) < torch.exp(-dU).clamp(max=1.0)
            flat[:, site] = torch.where(accept, -spins, spins)
            if lp_flip is not None:
                state = torch.where(accept, lp_flip, state)
        return sigma

    def rollout(self, sigma0, c):
        """Run from t = 0 to 1. Returns sigmas (n_save + 1, B, ...) and fp64 log-weights (n_save + 1, B)."""
        B, device = sigma0.shape[0], sigma0.device
        sigmas = torch.zeros((self.n_save + 1, *sigma0.shape)).to(sigma0)
        As = torch.zeros((self.n_save + 1, B), dtype=torch.float64, device=device)
        A = torch.zeros(B, dtype=torch.float64, device=device)
        sigma, c = sigma0, c.to(device)
        sigmas[0] = sigma0.detach()
        save = 1
        for i in range(len(self.ts) - 1):
            dt = (self.ts[i + 1] - self.ts[i]).repeat(B).to(device)
            t = self.ts[i].repeat(B).to(device)
            sigma, A = self._step(sigma, A, t, dt, c)
            sigma = self._metropolis(sigma, t, c)
            if (i + 1) % self.save_every == 0:
                sigmas[save] = sigma.detach()
                As[save] = A.detach()
                save += 1
            sigma, A = sigma.detach(), A.detach()
        sigmas[-1] = sigma.detach()
        As[-1] = A.detach()
        return sigmas, As
