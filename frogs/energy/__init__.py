"""Target energies and annealing paths: 2D Ising, AgPd (125 sites) and CuAu (128 sites)."""
from __future__ import annotations

from frogs.constants import K_B
from frogs.energy.alloy import AlloyEnergy
from frogs.energy.cluster_expansion import ClusterExpansion
from frogs.energy.ising import IsingEnergy
from frogs.paths import resolve_data_file

__all__ = ["AlloyEnergy", "ClusterExpansion", "IsingEnergy", "K_B", "make_energy"]

DEFAULT_TABLES = {
    "agpd_ce": "data/agpd_cluster_tables_L5.npz",
    "cuau_ce": "data/cuau_cluster_tables_8x4x4.npz",
}


def make_energy(config, prior=None):
    """Build the energy named by `config.system`; `prior` is the AR network or None (uniform)."""
    if config.system == "ising":
        L = config.lattice_shape[0]
        return IsingEnergy(L, prior, J=config.J, cond_low=config.cond_low,
                           cond_high=config.cond_high)
    table = resolve_data_file(config.table_path, DEFAULT_TABLES[config.system])
    return AlloyEnergy(table, config.lattice_shape, prior=prior,
                       cond_low=config.cond_low, cond_high=config.cond_high)
