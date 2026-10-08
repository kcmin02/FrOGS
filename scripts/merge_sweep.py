"""Reassemble the slices written by `scripts/sweep.py --slice I N` into one sweep file.

    python scripts/merge_sweep.py --tag cuau128
"""
import argparse
import glob
import os
import re

import numpy as np

from frogs.analysis.sweep_io import load_sweep, save_sweep
from frogs.paths import RESULTS_DIR

SAME = ("ckpt_sha256", "source_sha256", "dmu", "T", "M", "n_steps", "n_mcmc", "chunk",
        "n_boot", "seed", "store_A", "store_sigma", "backend", "compile", "N")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", required=True)
    p.add_argument("--out-dir", default=os.path.join(RESULTS_DIR, "sweeps"))
    p.add_argument("--allow-missing", action="store_true",
                   help="merge an incomplete set of slices; missing columns are NaN")
    args = p.parse_args()

    paths = sorted(glob.glob(os.path.join(args.out_dir, f"{args.tag}_slice*of*.npz")))
    if not paths:
        raise SystemExit(f"no slices matching {args.tag}_slice*of*.npz in {args.out_dir}")
    parts = [load_sweep(p) for p in paths]
    first = parts[0]["meta"]
    n_slices = int(re.search(r"of(\d+)\.npz$", paths[0]).group(1))
    for path, part in zip(paths, parts):
        bad = [k for k in SAME if part["meta"].get(k) != first.get(k)]
        if bad:
            raise SystemExit(f"{path} differs from {paths[0]} in {bad}")
    found = sorted(part["meta"]["slice"][0] for part in parts)
    missing = sorted(set(range(n_slices)) - set(found))
    if missing and not args.allow_missing:
        raise SystemExit(f"missing slices {missing}; rerun them or pass --allow-missing")

    dmu, T = np.asarray(first["dmu"]), np.asarray(first["T"])
    shape = (len(dmu), len(T))
    out = {}
    for part in parts:
        cols = np.asarray(part["meta"]["columns"])
        for k, v in part.items():
            if k in ("meta", "dmu", "T"):
                continue
            if k not in out:
                fill = 0 if v.dtype.kind in "iu" else np.nan
                out[k] = np.full(shape + v.shape[2:], fill, dtype=v.dtype)
            out[k][cols] = v
    out["dmu"], out["T"] = dmu, T
    meta = {k: v for k, v in first.items() if k not in ("slice", "columns", "wall_sec")}
    meta.update(columns=list(range(len(dmu))), wall_sec=sum(p["meta"]["wall_sec"] for p in parts),
                partial=bool(missing), missing_slices=missing)
    path = os.path.join(args.out_dir, f"{args.tag}.npz")
    save_sweep(path, out, meta)
    print(f"merged {len(paths)} slices into {path}" + (f" (missing {missing})" if missing else ""))


if __name__ == "__main__":
    main()
