"""Locally equivariant U-Net rate matrix Q_theta (§4.3, Fig. 1b; proof in §A.7).

The network emits one scalar g_t(i | sigma) per site with g(i | sigma) = -g(i | flip_i sigma):
every layer that reads the lattice excludes the site's own spin (centre-masked kernels),
pooling partitions the lattice, upsampling is nearest-neighbour and cross-scale mixing is
pointwise. The same code serves the 2D Ising lattice and 3D fcc supercells.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from frogs.nets.film import FiLM


def _conv_cls(dim):
    return nn.Conv2d if dim == 2 else nn.Conv3d


def unfold_circular(x, kernel_size):
    """K^d neighbourhood of every site under periodic boundaries: (B, C, *S) -> (B, C, K^d, *S)."""
    pad = kernel_size // 2
    if x.dim() == 4:
        xp = F.pad(x, (pad,) * 4, mode="circular")
        out = F.unfold(xp, kernel_size=kernel_size, padding=0)
        return out.reshape(x.shape[0], x.shape[1], kernel_size ** 2, x.shape[2], x.shape[3])
    xp = F.pad(x, (pad,) * 6, mode="circular")
    D, H, W = x.shape[2:]
    patches = [xp[:, :, i:i + D, j:j + H, k:k + W]
               for i in range(kernel_size) for j in range(kernel_size) for k in range(kernel_size)]
    return torch.stack(patches, dim=2)


class _CenterMasked:
    """Zero-padded convolution whose centre tap is fixed to zero."""

    def _init_mask(self, dim):
        mask = torch.ones_like(self.weight)
        mask[(...,) + (self.kernel_size[0] // 2,) * dim] = 0
        self.register_buffer("mask", mask)

    def forward(self, x):
        conv = F.conv2d if x.dim() == 4 else F.conv3d
        return conv(x, self.weight * self.mask, self.bias, self.stride, self.padding,
                    self.dilation, self.groups)


class CenterMaskedConv2d(_CenterMasked, nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size,
                           padding=kernel_size // 2, bias=False)
        self._init_mask(2)


class CenterMaskedConv3d(_CenterMasked, nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size):
        nn.Conv3d.__init__(self, in_channels, out_channels, kernel_size,
                           padding=kernel_size // 2, bias=False)
        self._init_mask(3)


def CenterMaskedConv(dim, in_channels, out_channels, kernel_size):
    cls = CenterMaskedConv2d if dim == 2 else CenterMaskedConv3d
    return cls(in_channels, out_channels, kernel_size)


class TimeEmbedding(nn.Module):
    """t -> one additive weight per kernel tap."""

    def __init__(self, n_taps, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                                 nn.Linear(hidden_dim, n_taps))

    def forward(self, t):
        return self.mlp(t)


class StateDependentConv(nn.Module):
    """Kernel built pointwise from the previous features and t, applied to the raw lattice
    with the centre tap zeroed, so the output at site i never reads sigma_i."""

    def __init__(self, dim, kernel_size, channels, prev_channels):
        super().__init__()
        n_taps = kernel_size ** dim
        scale = torch.sqrt(torch.tensor(n_taps)) if dim == 2 else n_taps ** 0.5
        self.A = nn.Parameter(0.002 * torch.rand(channels, n_taps, prev_channels) / scale)
        self.b1 = nn.Parameter(0.002 * torch.rand(channels, n_taps))
        self.b2 = nn.Parameter(0.002 * torch.rand(channels, n_taps))
        self.dim = dim
        self.kernel_size = kernel_size
        self.mid_idx = n_taps // 2
        self.mlp = TimeEmbedding(n_taps)

    def forward(self, x, t, prev):
        sp = "jk" if self.dim == 2 else "dhw"
        x_unfold = unfold_circular(x, self.kernel_size)
        A_prev = torch.einsum(f"bc{sp},lic->bli{sp}", prev, self.A)
        tail = (None,) * self.dim
        A_b = F.silu(A_prev + self.b1[(None, slice(None), slice(None)) + tail])
        t_emb = self.mlp(t.reshape(-1, 1))
        k_x = (A_b + self.b2[(None, slice(None), slice(None)) + tail]
               + t_emb[(slice(None), None, slice(None)) + tail])
        k_x[:, :, self.mid_idx] = 0.0
        return (k_x * x_unfold).sum(dim=2)


def leave_one_out_mean(x):
    """Mean of every spin other than i, which is invariant to flipping sigma_i."""
    dims = tuple(range(2, x.dim()))
    n = 1
    for d in dims:
        n *= x.shape[d]
    return (x.sum(dim=dims, keepdim=True) - x) / (n - 1)


class EncoderBlock(nn.Module):
    def __init__(self, dim, channels, kernel_size, n_convs, cond_dim, loo_context=False):
        super().__init__()
        self.loo_context = loo_context
        prev = channels + (1 if loo_context else 0)
        self.initial_conv = CenterMaskedConv(dim, 1, channels, kernel_size)
        self.state_dep_convs = nn.ModuleList(
            StateDependentConv(dim, kernel_size, channels, prev) for _ in range(n_convs))
        self.act = nn.SiLU()
        self.film_initial = FiLM(cond_dim, channels, dim)
        self.films = nn.ModuleList(FiLM(cond_dim, channels, dim) for _ in range(n_convs))

    def forward(self, x_scaled, x_raw, t, c):
        loo = leave_one_out_mean(x_raw) if self.loo_context else None
        f = self.film_initial(self.initial_conv(x_scaled), c)
        for conv, film in zip(self.state_dep_convs, self.films):
            prev = self.act(f)
            if loo is not None:
                prev = torch.cat([prev, loo], dim=1)
            f = film(conv(x_raw, t, prev), c)
        return f


class DecoderBlock(nn.Module):
    def __init__(self, dim, channels, kernel_size, n_convs, cond_dim):
        super().__init__()
        self.combine = _conv_cls(dim)(2 * channels, channels, kernel_size=1, bias=True)
        self.state_dep_convs = nn.ModuleList(
            StateDependentConv(dim, kernel_size, channels, channels) for _ in range(n_convs))
        self.act = nn.SiLU()
        self.film_combine = FiLM(cond_dim, channels, dim)
        self.films = nn.ModuleList(FiLM(cond_dim, channels, dim) for _ in range(n_convs))

    def forward(self, f_up, f_skip, x_raw, t, c):
        f = self.film_combine(self.combine(torch.cat([f_up, f_skip], dim=1)), c)
        for conv, film in zip(self.state_dep_convs, self.films):
            f = film(conv(x_raw, t, self.act(f)), c)
        return f


class LocEquivUNet(nn.Module):
    """Pyramid of `num_levels + 1` encoder scales and `num_levels` decoder scales."""

    def __init__(self, dim, num_levels, channels, kernel_size, encoder_convs, decoder_convs,
                 cond_dim=2, cond_low=None, cond_high=None, pool_strides=None, loo_context=False):
        super().__init__()
        self.dim = dim
        self.num_levels = num_levels
        if pool_strides is None:
            pool_strides = [2] * num_levels
        if len(pool_strides) != num_levels:
            raise ValueError("pool_strides needs one entry per level")
        self.pool_strides = [s if isinstance(s, int) else tuple(int(a) for a in s)
                             for s in pool_strides]
        self.encoder = nn.ModuleList(
            EncoderBlock(dim, channels, kernel_size, encoder_convs, cond_dim,
                         loo_context=loo_context and lvl == 0)
            for lvl in range(num_levels + 1))
        self.decoder = nn.ModuleList(
            DecoderBlock(dim, channels, kernel_size, decoder_convs, cond_dim)
            for _ in range(num_levels))
        self.linear = nn.Parameter(torch.randn(channels) / (channels ** 0.5))
        self._normalize_cond = cond_low is not None
        if self._normalize_cond:
            lo = torch.as_tensor(cond_low, dtype=torch.get_default_dtype())
            hi = torch.as_tensor(cond_high, dtype=torch.get_default_dtype())
            self.register_buffer("_cond_mid", 0.5 * (lo + hi))
            self.register_buffer("_cond_half", 0.5 * (hi - lo))

    def _pool(self, x, stride):
        pool = F.avg_pool2d if self.dim == 2 else F.avg_pool3d
        return pool(x, kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False)

    def _pyramid(self, x):
        levels = [x]
        for stride in self.pool_strides:
            levels.append(self._pool(levels[-1], stride))
        if min(levels[-1].shape[2:]) < 2:
            raise ValueError("every axis of the coarsest pyramid level must have length >= 2 "
                             "(§A.7); reduce num_levels or use anisotropic pool_strides")
        return levels

    def forward(self, x, t, c):
        if self._normalize_cond:
            c = (c - self._cond_mid) / self._cond_half
        x_raw = x.unsqueeze(1)
        x_scaled = (1 + t.view(-1, *(1,) * (self.dim + 1))) * x_raw
        raw, scaled = self._pyramid(x_raw), self._pyramid(x_scaled)
        skips = [self.encoder[l](scaled[l], raw[l], t, c) for l in range(self.num_levels + 1)]
        f = skips[-1]
        for l in range(self.num_levels - 1, -1, -1):
            f_up = F.interpolate(f, size=skips[l].shape[2:], mode="nearest")
            f = self.decoder[l](f_up, skips[l], raw[l], t, c)
        out = f * x_scaled
        return torch.sum(self.linear.view(1, -1, *(1,) * self.dim) * out, dim=1)


class RateMatrix(nn.Module):
    """Single-flip CTMC rates from the U-Net output g.

    With `metropolis_rates` the rates are [g]_+ * M with M = exp(-[Delta U_t]_+), whose
    corrector is the bounded -sum_i g_i M_i; otherwise the LEAPS rates [g]_+.
    """

    def __init__(self, network: LocEquivUNet, metropolis_rates: bool = False):
        super().__init__()
        self.network = network
        self.metropolis_rates = bool(metropolis_rates)
        self.dims = tuple(range(1, network.dim + 1))
        self._compiled = None

    def compile(self, mode="reduce-overhead"):
        if self._compiled is None:
            self._compiled = torch.compile(self.network.forward, mode=mode, dynamic=False,
                                           fullgraph=True)

    @property
    def compiled(self):
        return self._compiled is not None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_compiled"] = None
        return state

    def forward(self, sigma, t, c):
        if self._compiled is None:
            return self.network(sigma, t, c)
        return self._compiled(sigma, t, c).clone()  # CUDA-graph outputs are overwritten on replay

    def step(self, sigma, t, dt, c, dU=None, g=None):
        """One Euler step of the CTMC; flips sites of `sigma` in place and returns it."""
        if g is None:
            g = self.forward(sigma, t, c)
        rates = torch.relu(g)
        if self.metropolis_rates:
            rates = rates * torch.exp(-torch.relu(dU))
        p = torch.clip(dt.view(-1, *(1,) * len(self.dims)) * rates, min=0, max=1)
        flip = torch.rand_like(p) < p
        sigma[flip] = -sigma[flip]
        return sigma

    def corrector(self, sigma, t, dU, c, g=None):
        """sum over moves of the rate-matrix term of K_t rho_t, shape (B,)."""
        if g is None:
            g = self.forward(sigma, t, c)
        if self.metropolis_rates:
            return -(g * torch.exp(-torch.relu(dU))).sum(dim=self.dims)
        stay = -torch.relu(g).sum(dim=self.dims)
        return stay + (torch.relu(-g) * torch.exp(-dU)).sum(dim=self.dims)


def build_transport(config) -> RateMatrix:
    dim = len(config.lattice_shape)
    alloy = dim == 3
    net = LocEquivUNet(
        dim, config.num_levels, config.num_channels, config.kernel_size,
        config.encoder_convs, config.decoder_convs,
        cond_low=config.cond_low if alloy else None,
        cond_high=config.cond_high if alloy else None,
        pool_strides=config.pool_strides,
        loo_context=bool(config.loo_context) and alloy,
    )
    return RateMatrix(net, metropolis_rates=bool(config.metropolis_rates) and alloy)
