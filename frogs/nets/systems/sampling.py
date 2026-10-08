"""Ancestral sampling of the AR prior with per-site activation caches (CPU fallback of the backend)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


class CachedSampler:
    """Visit sites in raster order, computing each site's features from cached predecessors."""

    def __init__(self, prior, c, n_samples: int):
        self.prior = prior
        self.n_samples = int(n_samples)
        self.shape = tuple(prior.lattice_shape)
        self.dtype = prior.head.weight.dtype
        self.device = prior.head.weight.device
        c = torch.as_tensor(c, dtype=self.dtype, device=self.device)
        if c.ndim == 1:
            c = c.unsqueeze(0)
        self.c = c.expand(self.n_samples, -1).contiguous() if c.shape[0] == 1 else c
        self.convs = [prior.input_conv] + list(prior.convs)
        self.weights = [conv.weight * conv.mask for conv in self.convs]
        self.affines = [self._film_affine(f) for f in [prior.film_initial] + list(prior.films)]
        self.pad = tuple(int(p) for p in self.convs[0].padding)
        padded = tuple(n + 2 * p for n, p in zip(self.shape, self.pad))
        self.sigma_padded = torch.zeros(self.n_samples, 1, *padded, dtype=self.dtype,
                                        device=self.device)
        self.activations = [torch.zeros(self.n_samples, conv.out_channels, *padded,
                                        dtype=self.dtype, device=self.device)
                            for conv in self.convs]
        self.site = 0

    def _film_affine(self, film):
        gamma_raw, beta = film.net(self.c).chunk(2, dim=-1)
        return 1.0 + gamma_raw, beta

    def _coords(self):
        ly, lz = self.shape[1], self.shape[2]
        i, rem = divmod(self.site, ly * lz)
        return (i,) + divmod(rem, lz)

    def _local_conv(self, source, layer, i, j, k):
        kx, ky, kz = self.convs[layer].kernel_size
        patch = source[:, :, i:i + kx, j:j + ky, k:k + kz].contiguous()
        return F.conv3d(patch, self.weights[layer], self.convs[layer].bias).flatten(1)

    def next_logits(self):
        i, j, k = self._coords()
        pi, pj, pk = i + self.pad[0], j + self.pad[1], k + self.pad[2]
        source = self.sigma_padded
        for layer in range(len(self.convs)):
            gamma, beta = self.affines[layer]
            h = F.silu(gamma * self._local_conv(source, layer, i, j, k) + beta)
            self.activations[layer][:, :, pi, pj, pk] = h
            source = self.activations[layer]
        return F.conv3d(h[:, :, None, None, None], self.prior.head.weight,
                        self.prior.head.bias).flatten()

    def commit(self, spins):
        i, j, k = self._coords()
        self.sigma_padded[:, 0, i + self.pad[0], j + self.pad[1], k + self.pad[2]] = spins
        self.site += 1

    def result(self):
        px, py, pz = self.pad
        lx, ly, lz = self.shape
        return self.sigma_padded[:, 0, px:px + lx, py:py + ly, pz:pz + lz].contiguous()


@torch.no_grad()
def sample_cached(prior, c, n_samples: int) -> torch.Tensor:
    """Same raster order and one Bernoulli draw per site as `ARConvPrior.sample`."""
    sampler = CachedSampler(prior, c, n_samples)
    for _ in range(prior.N):
        bit = torch.bernoulli(torch.sigmoid(sampler.next_logits()))
        sampler.commit(2.0 * bit - 1.0)
    return sampler.result()
