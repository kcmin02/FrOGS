"""Table 4, FrOGS column: the conditions of lowest ESS in a sweep.

    python scripts/tables/tab4_lowest_ess.py results/sweeps/cuau128.npz
"""
import argparse

import numpy as np

from frogs.analysis.sweep_io import load_sweep


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sweep")
    p.add_argument("--n", type=int, default=8)
    args = p.parse_args()

    s = load_sweep(args.sweep, keys=("ess", "dmu", "T"))
    ess = s["ess"]
    order = np.argsort(np.where(np.isfinite(ess), ess, np.inf), axis=None)[:args.n]
    print(f"{'dmu [eV]':>9s} {'T [K]':>7s} {'ESS':>8s}")
    for flat in order:
        i, j = np.unravel_index(flat, ess.shape)
        print(f"{s['dmu'][i]:+9.3f} {s['T'][j]:7.0f} {ess[i, j]:8.4f}")


if __name__ == "__main__":
    main()
