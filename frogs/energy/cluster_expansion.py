"""Binary cluster-expansion energy U_CE(sigma) evaluated from precomputed index tables (§A.1).

U_CE(sigma) = N * [eci_0 + sum_c eci_c / M_c * sum_m prod_j sigma[idx_c[m, j]]], sigma in {-1, +1}^N.
The tables in `data/` hold, for every cluster orbit, the site indices of each occurrence on the
supercell. Sites are ordered by their primitive-basis coordinates (§A.9).
"""
from __future__ import annotations

import os

import numpy as np
import torch

K_MAX = 4  # largest cluster in the shipped expansions
FCC_PRIMITIVE = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])


def _reduce_row_mod2(row, n_sites):
    """Drop sites that occur an even number of times (sigma_i^2 = 1); pad with the dummy index."""
    mult = {}
    for x in row:
        s = int(x)
        if s < n_sites:
            mult[s] = mult.get(s, 0) + 1
    kept = [s for s, m in mult.items() if m % 2 == 1]
    return kept + [n_sites] * (K_MAX - len(kept))


class ClusterExpansion:
    """Packed occurrence tables plus per-site tables for single-flip energy differences.

    Rows shorter than K_MAX are padded with the dummy index N, where sigma is extended by +1.
    """

    def __init__(self, table_path: str, lattice_shape):
        lattice_shape = tuple(int(x) for x in lattice_shape)
        if not os.path.exists(table_path):
            raise FileNotFoundError(f"cluster table not found: {table_path}")
        data = dict(np.load(table_path, allow_pickle=True))
        Lx, Ly, Lz = lattice_shape
        n_sites = int(data["n_sites"])
        if n_sites != Lx * Ly * Lz:
            raise ValueError(f"table has {n_sites} sites, lattice_shape {lattice_shape}")
        if "lattice_shape" in data and tuple(int(x) for x in data["lattice_shape"]) != lattice_shape:
            raise ValueError(f"table lattice {tuple(data['lattice_shape'])} != {lattice_shape}")
        self.lattice_shape = lattice_shape
        self.n_sites = n_sites
        cell = data.get("ase_cell", FCC_PRIMITIVE * np.array(lattice_shape)[:, None])
        self.prim_vectors = np.asarray(cell, dtype=np.float64) / np.array(lattice_shape)[:, None]

        prim = data["prim_coords"]
        to_grid = (prim[:, 0] * Ly * Lz + prim[:, 1] * Lz + prim[:, 2]).astype(np.int64)
        eci_keys = list(data["eci_keys"])
        eci_values = data["eci_values"].astype(np.float64)

        rows, weights, eci_0 = [], [], 0.0
        for key, eci in zip(eci_keys, eci_values):
            size = int(data[f"cluster_{key}_size"])
            if size == 0:
                eci_0 = float(eci)
                continue
            inds = to_grid[data[f"cluster_{key}_indices"].astype(np.int64)]
            m = inds.shape[0]
            if size < K_MAX:
                inds = np.concatenate([inds, np.full((m, K_MAX - size), n_sites, np.int64)], axis=1)
            rows.append(np.array([_reduce_row_mod2(r, n_sites) for r in inds], dtype=np.int64))
            weights.append(np.full(m, float(eci) / m, dtype=np.float64))
        all_inds = np.concatenate(rows, axis=0)
        all_w = np.concatenate(weights, axis=0)

        # Each occurrence is stored once per member site; merge duplicate rows (products commute).
        uniq, inv = np.unique(np.sort(all_inds, axis=1), axis=0, return_inverse=True)
        uniq_w = np.zeros(uniq.shape[0], dtype=np.float64)
        np.add.at(uniq_w, np.asarray(inv).reshape(-1), all_w)
        all_inds, all_w = uniq, uniq_w

        site_rows = [[] for _ in range(n_sites)]
        site_w = [[] for _ in range(n_sites)]
        for row, w in zip(all_inds, all_w):
            for s in row:
                if s < n_sites:
                    site_rows[s].append(row)
                    site_w[s].append(float(w))
        n_per_site = len(site_rows[0])
        if any(len(r) != n_per_site for r in site_rows):
            raise ValueError("cluster table is not translation invariant")

        dt = torch.get_default_dtype()
        self.all_inds = torch.as_tensor(all_inds, dtype=torch.int64)
        self.all_weights = torch.as_tensor(all_w, dtype=dt)
        self.eci_0 = torch.as_tensor(eci_0, dtype=dt)
        self.site_inds = torch.as_tensor(np.stack([np.stack(r) for r in site_rows]), dtype=torch.int64)
        self.site_weights = torch.as_tensor(np.array(site_w, dtype=np.float64), dtype=dt)
        self.site_inds_flat = self.site_inds.reshape(-1).contiguous()

    def to(self, device):
        for name in ("all_inds", "all_weights", "eci_0", "site_inds", "site_weights",
                     "site_inds_flat"):
            setattr(self, name, getattr(self, name).to(device))
        return self

    def energy(self, sigma):
        """U_CE for a batch, (B, N) -> (B,)."""
        ext = torch.cat([sigma, sigma.new_ones(sigma.shape[0], 1)], dim=1)
        prods = ext[:, self.all_inds].prod(dim=2)
        return self.n_sites * (self.eci_0 + (prods * self.all_weights).sum(dim=1))

    def energy_single(self, sigma):
        """U_CE for one configuration, (N,) -> (); vmap-compatible."""
        ext = torch.cat([sigma, sigma.new_ones(1)])
        prods = ext[self.all_inds].prod(dim=1)
        return self.n_sites * (self.eci_0 + (self.all_weights * prods).sum())

    def flip_deltas_single(self, sigma):
        """U_CE(flip_i sigma) - U_CE(sigma) for every site i, (N,) -> (N,); vmap-compatible."""
        ext = torch.cat([sigma, sigma.new_ones(1)])
        gathered = ext.index_select(0, self.site_inds_flat).view(self.n_sites, -1, K_MAX)
        prods = gathered.prod(dim=2)
        return self.n_sites * (-2.0) * (prods * self.site_weights).sum(dim=1)

    def flip_delta_at(self, sigma, site: int):
        """U_CE(flip_site sigma) - U_CE(sigma) for one site across a batch, (B, N) -> (B,)."""
        ext = torch.cat([sigma, sigma.new_ones(sigma.shape[0], 1)], dim=1)
        prods = ext[:, self.site_inds[site]].prod(dim=2)
        return self.n_sites * (-2.0) * (prods * self.site_weights[site][None, :]).sum(dim=1)
