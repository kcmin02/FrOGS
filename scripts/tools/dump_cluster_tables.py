"""Regenerate a cluster table in data/ from the ECI files in data/reference/ with CLEASE (§A.1, §A.9).

    python scripts/tools/dump_cluster_tables.py --system cuau --size 8 4 4
    python scripts/tools/dump_cluster_tables.py --system agpd --size 5 5 5

Needs CLEASE and ASE, which nothing else imports. Checks the table against the CLEASE energy.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile

import numpy as np
import torch

from frogs.energy import ClusterExpansion
from frogs.paths import REFERENCE_DIR, data_file

# Settings of the SEGAL notebooks; `z` is the atomic number of sigma = +1, -1.
SYSTEMS = {
    "cuau": dict(eci="ECI_CuAu_Final_Submission.json", elements=["Au", "Cu"],
                 z=(79, 29), a=3.8, max_cluster_dia=[6.0, 4.5, 4.5], size=(8, 4, 4)),
    "agpd": dict(eci="agpd_dft_eci_final_submission.json", elements=["Ag", "Pd"],
                 z=(47, 46), a=4.09, max_cluster_dia=[8.0, 6.5, 5.5], size=(5, 5, 5)),
}


def key_order(key):
    """c0, c1_0, then clusters by size and diameter index."""
    if key == "c0":
        return (0, 0, 0, key)
    parts = key.split("_")
    return (int(parts[0][1]), key.count("_"), int(parts[1].lstrip("d") or 0), key)


def cluster_name(key):
    """ECI key -> CLEASE cluster name, e.g. 'c2_d0001_0_00' -> 'c2_d0001_0'."""
    return "_".join(key.split("_")[:-1])


def default_out(system, size):
    lx, ly, lz = size
    suffix = f"L{lx}" if lx == ly == lz else f"{lx}x{ly}x{lz}"
    return data_file(f"{system}_cluster_tables_{suffix}.npz")


def build_tables(system, size, db_name):
    from clease import CEBulk, Concentration
    from clease.calculator import attach_calculator

    spec = SYSTEMS[system]
    with open(os.path.join(REFERENCE_DIR, spec["eci"])) as f:
        eci = json.load(f)
    settings = CEBulk(crystalstructure="fcc", a=spec["a"], size=list(size),
                      concentration=Concentration(basis_elements=[spec["elements"]]),
                      db_name=db_name, max_cluster_size=4,
                      max_cluster_dia=spec["max_cluster_dia"])
    atoms = attach_calculator(settings, atoms=settings.atoms.copy(), eci=eci)
    n = len(atoms)

    pos = atoms.get_positions()
    cell = np.asarray(atoms.cell.array)
    frac = pos @ np.linalg.inv(cell)
    frac -= np.floor(frac + 1e-9)
    prim = (frac * np.array(size, dtype=float)).round().astype(np.int32) % np.array(size)
    grid = np.empty(size, dtype=np.int32)
    grid[tuple(prim.T)] = np.arange(n)
    # T[r, j]: the site that j moves to under the lattice translation taking site 0 to site r.
    moved = (prim[None, :, :] + prim[:, None, :] - prim[0]) % np.array(size)
    T = grid[moved[..., 0], moved[..., 1], moved[..., 2]].astype(np.int32)

    keys = sorted(eci, key=key_order)
    tables = {
        "eci_keys": np.array(keys),
        "eci_values": np.array([eci[k] for k in keys], dtype=np.float64),
        "n_sites": np.array(n, dtype=np.int32),
        "lattice_shape": np.array(size, dtype=np.int32),
        "ase_positions": pos,
        "ase_cell": cell,
        "translation_matrix": T,
        "prim_coords": prim.astype(np.int32),
    }
    clusters = {c.name: c for c in settings.cluster_list}
    for key in keys:
        if key == "c0":
            size_k, inds = 0, np.empty((1, 0), dtype=np.int32)
        else:
            if key == "c1_0":
                ref = np.arange(n, dtype=np.int32)[:, None]
            else:
                ref = np.asarray(clusters[cluster_name(key)].indices, dtype=np.int32)
            # CLEASE lists each orbit relative to site 0; translate it to every site.
            size_k, inds = ref.shape[1], T[:, ref].reshape(-1, ref.shape[1])
        tables[f"cluster_{key}_size"] = np.array(size_k, dtype=np.int32)
        tables[f"cluster_{key}_indices"] = inds
    return tables, atoms


def check_against_clease(path, tables, atoms, system):
    """Max |U_CE - U_CLEASE| (eV) over the two pure cells and three random configurations."""
    n = int(tables["n_sites"])
    z_plus, z_minus = SYSTEMS[system]["z"]
    rng = np.random.default_rng(0)
    configs = [-np.ones(n), np.ones(n)] + [rng.choice([-1.0, 1.0], size=n) for _ in range(3)]
    torch.set_default_dtype(torch.float64)
    ce = ClusterExpansion(path, tuple(int(s) for s in tables["lattice_shape"]))
    prim, shape = tables["prim_coords"], tables["lattice_shape"]
    to_grid = (prim[:, 0] * shape[1] + prim[:, 1]) * shape[2] + prim[:, 2]
    err = 0.0
    for sigma in configs:
        atoms.numbers = np.where(sigma > 0, z_plus, z_minus)
        grid_sigma = torch.zeros(1, n)
        grid_sigma[0, to_grid] = torch.as_tensor(sigma)
        err = max(err, abs(atoms.get_potential_energy() - float(ce.energy(grid_sigma)[0])))
    return err


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--system", choices=sorted(SYSTEMS), required=True)
    ap.add_argument("--size", type=int, nargs=3,
                    help="supercell in primitive fcc cells (default: the cell in configs/)")
    ap.add_argument("--out", help="output .npz (default: the matching file in data/)")
    args = ap.parse_args()

    size = tuple(args.size or SYSTEMS[args.system]["size"])
    out = args.out or default_out(args.system, size)
    with tempfile.TemporaryDirectory() as tmp:
        tables, atoms = build_tables(args.system, size, os.path.join(tmp, "clease.db"))
        np.savez_compressed(out, **tables)
        err = check_against_clease(out, tables, atoms, args.system)
    n_occ = sum(len(v) for k, v in tables.items() if k.endswith("_indices") and v.size)
    print(f"wrote {out}: {int(tables['n_sites'])} sites, {len(tables['eci_keys'])} ECIs, "
          f"{n_occ} cluster occurrences")
    print(f"max |U_CE - U_CLEASE| over 5 configurations: {err:.2e} eV "
          f"({'PASS' if err < 1e-8 else 'FAIL'})")
    if err >= 1e-8:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
