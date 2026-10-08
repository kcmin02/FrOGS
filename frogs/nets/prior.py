"""Autoregressive prior rho_0(sigma | c) = prod_s p(sigma_s | sigma_<s, c) (§4.1, Fig. 8).

A PixelCNN-style stack of causally masked Conv3d layers over the raster order (dim 0 slowest).
The first layer excludes the centre voxel (type A), deeper layers include it (type B), padding is
zero, and FiLM/SiLU/head are pointwise, so Z_0 = 1 exactly for any weights.

`backend = "systems"` routes no-grad sampling and all-site flip scoring through fused Triton
kernels replayed as CUDA graphs (`frogs.nets.systems`); this changes FP32 rounding only.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from frogs.nets.film import FiLM


def causal_mask(kernel_size: int, mask_type: str) -> torch.Tensor:
    """(K, K, K) boolean mask keeping raster-earlier taps (and the centre for type B)."""
    c = kernel_size // 2
    keep = torch.zeros(kernel_size, kernel_size, kernel_size, dtype=torch.bool)
    for a in range(kernel_size):
        for b in range(kernel_size):
            for d in range(kernel_size):
                off = (a - c, b - c, d - c)
                keep[a, b, d] = off < (0, 0, 0) or (mask_type == "B" and off == (0, 0, 0))
    return keep


class MaskedConv3d(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, mask_type):
        super().__init__(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=True)
        mask = torch.zeros_like(self.weight)
        mask[:, :, causal_mask(kernel_size, mask_type)] = 1.0
        self.register_buffer("mask", mask)

    def forward(self, x):
        return F.conv3d(x, self.weight * self.mask, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


class ARConvPrior(nn.Module):
    """Causal masked-Conv3d prior over sigma in {-1, +1}^(Lx, Ly, Lz)."""

    def __init__(self, lattice_shape, cond_dim: int = 2, num_layers: int = 4,
                 num_channels: int = 32, kernel_size: int = 3):
        super().__init__()
        self.lattice_shape = tuple(int(s) for s in lattice_shape)
        self.N = self.lattice_shape[0] * self.lattice_shape[1] * self.lattice_shape[2]
        self.num_channels = num_channels
        self.input_conv = MaskedConv3d(1, num_channels, kernel_size, "A")
        self.convs = nn.ModuleList(MaskedConv3d(num_channels, num_channels, kernel_size, "B")
                                   for _ in range(num_layers - 1))
        self.act = nn.SiLU()
        self.film_initial = FiLM(cond_dim, num_channels, 3)
        self.films = nn.ModuleList(FiLM(cond_dim, num_channels, 3) for _ in range(num_layers - 1))
        self.head = nn.Conv3d(num_channels, 1, kernel_size=1)
        self.backend = "reference"
        self._compiled_logits = None
        self._compile_grad = True

    def compile(self, mode="reduce-overhead", grad=True):
        """torch.compile the logits body; `grad=False` keeps autograd-connected calls eager."""
        self._compile_grad = grad
        if self._compiled_logits is None:
            self._compiled_logits = torch.compile(self._logits, mode=mode, dynamic=False,
                                                  fullgraph=True)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_compiled_logits"] = None
        state.pop("_graph_cache", None)
        state.pop("_triton_metadata", None)
        return state

    def _logits(self, sigma, c):
        x = sigma.unsqueeze(1).to(self.head.weight.dtype)
        h = self.act(self.film_initial(self.input_conv(x), c))
        for conv, film in zip(self.convs, self.films):
            h = self.act(film(conv(h), c))
        return self.head(h).squeeze(1)

    def logits(self, sigma, c):
        """Per-site logits of p(sigma_s = +1 | sigma_<s, c), shape (B, Lx, Ly, Lz)."""
        f = self._compiled_logits
        if f is None or (not self._compile_grad and torch.is_grad_enabled()):
            return self._logits(sigma, c)
        return f(sigma, c).clone()  # CUDA-graph outputs are overwritten on replay

    def log_prob(self, sigma, c):
        logit = self.logits(sigma, c)
        return (-F.softplus(-sigma * logit)).flatten(1).sum(dim=1)

    def _systems_active(self, sigma):
        return (self.backend == "systems" and not torch.is_grad_enabled() and sigma.is_cuda
                and sigma.dtype == torch.float32)

    @torch.no_grad()
    def sample(self, c, n_samples: int):
        """Exact ancestral sampling in raster order, (n_samples, Lx, Ly, Lz)."""
        device, dtype = self.head.weight.device, self.head.weight.dtype
        c = torch.as_tensor(c, dtype=dtype, device=device)
        if c.ndim == 1:
            c = c.unsqueeze(0)
        c = c.expand(n_samples, -1).contiguous() if c.shape[0] == 1 else c
        if self.backend == "systems":
            from frogs.nets.systems import cuda_graphs, sampling
            op = sampling.sample_cached
            if device.type == "cuda":
                from frogs.nets.systems.triton_kernels import sample_triton
                op = sample_triton
            return cuda_graphs.replay(self, op.__name__, lambda cc, n: op(self, cc, n), c, n_samples)
        Lx, Ly, Lz = self.lattice_shape
        sigma = torch.zeros(n_samples, Lx, Ly, Lz, dtype=dtype, device=device)
        for i in range(Lx):
            for j in range(Ly):
                for k in range(Lz):
                    logit = self.logits(sigma, c)[:, i, j, k]
                    sigma[:, i, j, k] = 2.0 * torch.bernoulli(torch.sigmoid(logit)) - 1.0
        return sigma

    def delta_log_prob_all_sites(self, sigma, c, base_log_prob=None):
        """log rho_0(flip_s sigma) - log rho_0(sigma) for every site s, shape (B, N)."""
        if self._systems_active(sigma):
            from frogs.nets.systems import cuda_graphs, triton_kernels
            if triton_kernels.supported(self, sigma):
                op = lambda ss, cc: triton_kernels.delta_log_prob_all_sites(self, ss, cc)
                return cuda_graphs.replay(self, "triton_flips", op, sigma, c)
        return self.delta_log_prob_all_sites_reference(sigma, c, base_log_prob)

    def delta_log_prob_all_sites_reference(self, sigma, c, base_log_prob=None):
        """Evaluates all N single-flip neighbours in one (B * N)-batch."""
        B, N = sigma.shape[0], self.N
        base = self.log_prob(sigma, c) if base_log_prob is None else base_log_prob
        rep = sigma.reshape(B, N).unsqueeze(1).repeat(1, N, 1)
        diag = torch.arange(N, device=sigma.device)
        rep[:, diag, diag] = -rep[:, diag, diag]
        rep = rep.reshape(B * N, *self.lattice_shape)
        c_rep = c.unsqueeze(1).repeat(1, N, 1).reshape(B * N, -1)
        return self.log_prob(rep, c_rep).reshape(B, N) - base.unsqueeze(1)


def build_prior(config) -> ARConvPrior:
    """The prior for `config`; a 2D L x L lattice is treated as the 3D shape (1, L, L)."""
    shape = list(config.lattice_shape)
    shape = [1] + shape if len(shape) == 2 else shape
    net = config.prior_net
    prior = ARConvPrior(shape, num_layers=int(net["num_layers"]),
                        num_channels=int(net["num_channels"]), kernel_size=int(net["kernel_size"]))
    prior.backend = config.prior_backend
    return prior
