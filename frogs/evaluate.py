"""Importance-sampling readout of a trained model on a grid of conditions (Eq. 2, Eq. 3).

At each c = (dmu, T): log Z(c) = log Z_0 + log E[e^A], the composition <N_+>/N, and the ESS.
log E[e^A] is reported as mean(A) + var(A)/2 (`logZ`) and as logsumexp(A) - log M (`logZ_raw`).
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch

from frogs.sampler import CTMCSampler, ess
from frogs.trainer import FrOGSModule


def load_model(path, device=None, compile=True, backend=None):
    """Load a trained model (current or older checkpoint format) for evaluation."""
    torch.serialization.add_safe_globals([argparse.Namespace])
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = FrOGSModule.load_from_checkpoint(path, map_location=device).to(device).eval()
    model.energy.to(device)
    if model.prior is not None:
        if backend:
            model.prior.backend = backend
        if compile:
            model.prior.compile(grad=False)
    if compile and model.transport is not None:
        model.transport.compile()
    return model


def kappa2_log_mean_exp(A):
    A = torch.as_tensor(A, dtype=torch.float64)
    var = float(A.var(unbiased=True)) if A.numel() > 1 else 0.0
    return float(A.mean()) + 0.5 * var


def raw_log_mean_exp(A):
    A = torch.as_tensor(A, dtype=torch.float64)
    return float(torch.logsumexp(A, dim=0) - math.log(A.numel()))


def bootstrap_se(A, n_plus, n_sites, n_boot, rng):
    """Bootstrap SE of (logZ, x), resampling walkers."""
    M = A.size
    if M < 2 or n_boot <= 0:
        return 0.0, 0.0
    idx = rng.integers(0, M, size=(n_boot, M))
    Ab, Nb = A[idx], n_plus[idx]
    logZ = Ab.mean(axis=1) + 0.5 * Ab.var(axis=1, ddof=1)
    w = np.exp(Ab - Ab.max(axis=1, keepdims=True))
    x = (w * Nb).sum(axis=1) / w.sum(axis=1) / n_sites
    return float(logZ.std(ddof=1)), float(x.std(ddof=1))


class ISEvaluator:
    """Roll M walkers per condition through the trained CTMC and read out Eq. (2)-(3)."""

    def __init__(self, model, M=1000, n_steps=None, n_mcmc=0, seed=0, device=None,
                 transport=None):
        self.model = model
        self.energy = model.energy
        self.M = int(M)
        self.n_mcmc = int(n_mcmc)
        self.seed = int(seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if transport is None:
            transport = model.transport is not None
        if transport and model.transport is None:
            raise ValueError("this checkpoint has no transport network")
        self.transport = model.transport if transport else None
        self.n_steps = int(n_steps or model.config.n_steps)
        if self.transport is None and self.n_mcmc == 0 and self.energy.learned_prior:
            # Nothing moves sigma, so A = U_0 - U_1 exactly on any time grid.
            self.n_steps = 1
        self.n_sites = self.energy.n_sites
        self.ts = torch.linspace(0.0, 1.0, self.n_steps + 1).to(self.device)

    def run(self, dmu, T, chunk=1000, n_boot=500, store_A=False, store_sigma=False,
            verbose=False) -> dict:
        """Sweep the outer product of `dmu` and `T`; returns arrays shaped (n_mu, n_T[, M])."""
        dmu, T = np.asarray(dmu, dtype=np.float64), np.asarray(T, dtype=np.float64)
        DMU, TT = np.meshgrid(dmu, T, indexing="ij")
        points = np.stack([DMU.ravel(), TT.ravel()], axis=1)
        n_pts, M = len(points), self.M
        conds = np.repeat(points, M, axis=0)
        total = len(conds)
        chunk = max(1, min(int(chunk), total))

        A_all = np.empty(total)
        n_plus = np.empty(total)
        occ_all = np.empty((total, self.n_sites), dtype=np.uint8) if store_sigma else None

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        rng = np.random.default_rng(self.seed + 1)
        sampler = CTMCSampler(self.energy, self.ts, self.transport, n_mcmc=self.n_mcmc, n_save=1)
        t0 = time.time()
        for start in range(0, total, chunk):
            end = min(start + chunk, total)
            c = torch.as_tensor(conds[start:end], dtype=torch.get_default_dtype(),
                                device=self.device)
            with torch.no_grad():
                sigmas, As = sampler.rollout(self.energy.sample_prior(c), c)
            occ = ((sigmas[-1] + 1.0) * 0.5).reshape(end - start, -1)
            A_all[start:end] = As[-1].double().cpu().numpy()
            n_plus[start:end] = occ.sum(dim=1).double().cpu().numpy()
            if store_sigma:
                occ_all[start:end] = occ.to(torch.uint8).cpu().numpy()
            if verbose:
                print(f"  walkers {end}/{total}  {(time.time() - t0) / 60:.1f} min", flush=True)

        log_Z0 = self.energy.log_Z0()
        out = {k: np.zeros(n_pts) for k in ("logZ", "logZ_se", "logZ_raw", "x", "x_se", "ess")}
        for p in range(n_pts):
            A, n = A_all[p * M:(p + 1) * M], n_plus[p * M:(p + 1) * M]
            At = torch.as_tensor(A)
            out["logZ"][p] = log_Z0 + kappa2_log_mean_exp(A)
            out["logZ_raw"][p] = log_Z0 + raw_log_mean_exp(A)
            w = torch.softmax(At, dim=0)
            out["x"][p] = float((w * torch.as_tensor(n)).sum() / self.n_sites)
            out["ess"][p] = float(ess(At))
            out["logZ_se"][p], out["x_se"][p] = bootstrap_se(A, n, self.n_sites, n_boot, rng)

        shape = (len(dmu), len(T))
        result = {k: v.reshape(shape) for k, v in out.items()}
        result.update(dmu=dmu, T=T, M=np.full(shape, M, dtype=np.int64))
        if store_A:
            result["A_samples"] = A_all.reshape(*shape, M)
            result["N_samples"] = n_plus.reshape(*shape, M)
        if store_sigma:
            result["sigma_packed"] = np.packbits(occ_all, axis=-1).reshape(*shape, M, -1)
        return result
