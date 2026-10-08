# Data

| file | cell | sites | used by |
| --- | --- | --- | --- |
| `cuau_cluster_tables_8x4x4.npz` | 8x4x4 fcc | 128 | `configs/cuau_128.yml` |
| `agpd_cluster_tables_L5.npz` | 5x5x5 fcc | 125 | `configs/agpd_125.yml` |
| `cuau_cluster_tables_2x2x4.npz` | 2x2x4 fcc | 16 | `scripts/checks/check_exactness.py` (exhaustive enumeration) |

Each table holds the effective cluster interactions (`eci_keys`, `eci_values`), the supercell
(`n_sites`, `prim_coords`: primitive-basis coordinates of every site, §A.9), and
per cluster orbit the site indices of every occurrence (`cluster_<name>_indices`, `cluster_<name>_size`).
They are derived from the ECI files in `reference/` and can be regenerated with
`scripts/tools/dump_cluster_tables.py` (requires CLEASE and ASE).

`reference/` holds the cluster expansions and the metadynamics reference free energies F(x, T) of
Damewood et al. (2022), used only for comparison in Fig. 3 (see `reference/NOTICE`).

The 2D Ising model needs no data. Set `FROGS_DATA_DIR` to keep the data elsewhere.
