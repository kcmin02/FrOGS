"""Tables 2 and 3: training-time statistics over the final optimizer steps of each run.

    python scripts/tables/training_stats.py results/cuau_ce/tab2-cuau-ar32-mcmc10-1/metrics.csv ...

Reports the median and 99th percentile of `train_loss` and of `A_corrected_std` (the per-step spread
of A_1 + F(1, c) - F(0, c)) over the last `--last` steps; non-finite entries are excluded and counted.
"""
import argparse
import csv
import os

import numpy as np


def read_metric(path, name):
    steps, values = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            if row.get(name) not in (None, ""):
                steps.append(int(row["step"]))
                values.append(float(row[name]))
    return np.asarray(steps), np.asarray(values)


def tail_stats(path, name, last):
    steps, values = read_metric(path, name)
    keep = steps > steps.max() - last
    v = values[keep]
    finite = np.isfinite(v)
    return np.median(v[finite]), np.percentile(v[finite], 99), int((~finite).sum()), int(keep.sum())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("metrics", nargs="+", help="metrics.csv files written by scripts/train.py --offline")
    p.add_argument("--last", type=int, default=10_000, help="number of final optimizer steps")
    args = p.parse_args()

    print(f"{'run':40s} {'loss med':>9s} {'loss p99':>10s} {'std[A] med':>11s} {'std[A] p99':>11s}")
    for path in args.metrics:
        run = os.path.basename(os.path.dirname(os.path.abspath(path)))
        l_med, l_p99, l_bad, n = tail_stats(path, "train_loss", args.last)
        a_med, a_p99, _, _ = tail_stats(path, "A_corrected_std", args.last)
        note = f"  ({l_bad}/{n} non-finite losses excluded)" if l_bad else ""
        print(f"{run:40s} {l_med:9.2f} {l_p99:10.3g} {a_med:11.2f} {a_p99:11.3g}{note}")


if __name__ == "__main__":
    main()
