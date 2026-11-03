"""Shared pieces of the annealing path U_t(sigma | c) between the prior and the target (§A.5)."""
from __future__ import annotations

import math

import numpy as np
import torch

from frogs.energy.symmetry import detect_symmetry


class PathEnergy:
    """Base class: condition normalization and the learned prior rho_0(sigma | c).

    With a learned prior the path is U_t = t * U_1 - (1 - t) * log rho_0, with U_1 = beta * H_c,
    so Z_0 = 1 and the log-weight increment U_{t+dt} - U_t is exact (Eq. 33).
    """

    lattice_shape: tuple
    lattice_vectors: np.ndarray
    n_sites: int
    field_index = None

    def __init__(self, prior, cond_low, cond_high):
        self.prior = prior
        self.cond_low = None if cond_low is None else [float(v) for v in cond_low]
        self.cond_high = None if cond_high is None else [float(v) for v in cond_high]

    @property
    def learned_prior(self) -> bool:
        return self.prior is not None

    @property
    def prior_shape(self):
        return self.lattice_shape

    def norm_c(self, c):
        """Rescale c = (dmu, T) to [-1, 1] over the training box (§A.8)."""
        if self.cond_low is None:
            return c
        lo = torch.as_tensor(self.cond_low, dtype=c.dtype, device=c.device)
        hi = torch.as_tensor(self.cond_high, dtype=c.dtype, device=c.device)
        return 2.0 * (c - lo) / (hi - lo) - 1.0

    def log_prior(self, sigma, c):
        return self.prior.log_prob(sigma.reshape(-1, *self.prior_shape), self.norm_c(c))

    def log_Z0(self) -> float:
        return 0.0 if self.learned_prior else self.n_sites * math.log(2.0)

    def sample_prior(self, c):
        """sigma ~ rho_0(. | c), shape (B, *lattice_shape)."""
        if self.learned_prior:
            sigma = self.prior.sample(self.norm_c(c), c.shape[0])
            return sigma.reshape(c.shape[0], *self.lattice_shape)
        sigma = 2 * torch.randint(0, 2, (c.shape[0], *self.lattice_shape), device=c.device) - 1
        return sigma.to(torch.get_default_dtype())

    def symmetry_group(self, device=None):
        return detect_symmetry(self, device)

    def to(self, device):
        if self.prior is not None:
            self.prior.to(device)
        return self
