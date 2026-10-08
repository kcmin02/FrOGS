"""§A.13: ordered-compound count of the §A.4 constructor over 2268 threshold settings per sweep.

    python scripts/tables/gate_sensitivity.py --arm FrOGS=<cuau>.npz --arm SEGAL=<segal_cuau>.npz
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
from collections import Counter

from frogs.analysis.phase_diagram import (PERSIST_RULES, SWEEP_KEYS, certify, ordered_compounds,
                                          site_count, sweep_gaps)
from frogs.analysis.sweep_io import load_sweep
from frogs.paths import RESULTS_DIR, ensure_dir

W_FLOOR_GRID = (0.0, 2.0, 3.0, 5.0, 8.0, 10.0, 20.0)
W_MIN_GRID = tuple(range(2, 11))
N_SIGMA_GRID = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
ABSENCE_GRID = (True, False)


def scan(sweep):
    """One record per setting: thresholds, number of rungs and the ordered compounds."""
    n_sites = site_count(sweep)
    records = []
    for w_floor in W_FLOOR_GRID:
        gaps, _ = sweep_gaps(sweep, n_sites, w_floor)
        for w_min, n_sigma, persist, absence in itertools.product(
                W_MIN_GRID, N_SIGMA_GRID, PERSIST_RULES, ABSENCE_GRID):
            rungs, _ = certify(gaps, n_sites, w_min, n_sigma, persist, absence)
            comps = ordered_compounds(rungs, n_sites)
            records.append(dict(w_floor=w_floor, w_min=w_min, n_sigma=n_sigma, persist=persist,
                                measured_absence=absence,
                                n_rungs=sum(len(v) for v in rungs.values()),
                                n_compounds=len(comps),
                                compounds=" ".join(f"{x:.4f}" for x, _ in comps)))
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=SWEEP")
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "gate_sensitivity.csv"))
    args = ap.parse_args()

    rows = []
    for spec in args.arm:
        name, path = spec.split("=", 1)
        records = scan(load_sweep(path, keys=SWEEP_KEYS))
        rows += [dict(arm=name, **r) for r in records]
        counts = Counter(r["n_compounds"] for r in records)
        xs = Counter(round(float(x), 2) for r in records for x in r["compounds"].split())
        print(f"{name}: {len(records)} settings; ordered compounds -> settings: "
              + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
        print(f"  compositions (settings resolving them): "
              + ", ".join(f"x={x:.2f} ({v})" for x, v in sorted(xs.items())))

    ensure_dir(os.path.dirname(os.path.abspath(args.out)))
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
