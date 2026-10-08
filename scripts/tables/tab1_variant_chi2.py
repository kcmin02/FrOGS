"""Table 1: chi^2/dof of the degenerate-variant histograms against the flat 1/g expectation.

    python scripts/tables/tab1_variant_chi2.py --sweep results/sweeps/cuau128.npz \
        --seeds results/sweeps/cuau128-s2.npz results/sweeps/cuau128-s3.npz results/sweeps/cuau128-s4.npz

The first sweep gives the unweighted and weighted rows; each `--seeds` sweep adds its weighted
value. Sweeps must carry `A_samples` and `sigma_packed` (scripts/sweep.py --store-A --store-sigma).
"""
from __future__ import annotations

import argparse

from frogs.analysis.sweep_io import load_sweep
from frogs.analysis.variants import MIN_GATED, PHASES, SWEEP_KEYS, chi2_table, variant_histograms
from frogs.paths import figure_path


def table(path, min_gated):
    return chi2_table(variant_histograms(load_sweep(path, keys=SWEEP_KEYS)), min_gated)


def to_latex(rows, seeds):
    head = " & ".join(rf"{ph.label} ($g={ph.g}$)" for ph in PHASES)
    lines = [r"\begin{tabular}{lccc}", r"  \toprule",
             rf"  $\chi^2/\mathrm{{dof}}$ vs $1/g$ & {head} \\", r"  \midrule"]
    fmt = lambda v, key: f"{v:.1f}" if key == "unweighted" or v >= 10 else f"{v:.2f}"
    for key, name in (("unweighted", "FrOGS unweighted"), ("weighted", "FrOGS weighted")):
        lines.append(rf"  {name} & " + " & ".join(fmt(rows[ph.name][key], key) for ph in PHASES)
                     + r" \\")
    if seeds:
        lines.append(r"  \midrule")
        vals = " & ".join("/".join(fmt(s[ph.name]["weighted"], "weighted") for s in seeds)
                          for ph in PHASES)
        lines.append(rf"  FrOGS weighted, other seeds & {vals} \\")
    lines.append(r"  \midrule")
    lines.append(r"  perfect balance (null median) & "
                 + " & ".join(f"{rows[ph.name]['null']:.2f}" for ph in PHASES) + r" \\")
    lines += [r"  \bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sweep", required=True, help="CuAu sweep with A_samples and sigma_packed")
    ap.add_argument("--seeds", nargs="*", default=[], help="sweeps of independently trained models")
    ap.add_argument("--min-gated", type=int, default=MIN_GATED)
    ap.add_argument("--out", default=None, help="LaTeX tabular (default: figures/tab1_variant_chi2.tex)")
    args = ap.parse_args()

    rows = table(args.sweep, args.min_gated)
    seeds = [table(p, args.min_gated) for p in args.seeds]

    print(f"{'chi2/dof vs 1/g':32s}" + "".join(f"{ph.name + f' (g={ph.g})':>16s}" for ph in PHASES))
    for key, name, fmt in (("unweighted", "FrOGS unweighted", "{:16.1f}"),
                           ("weighted", "FrOGS weighted", "{:16.2f}"),
                           ("null", "perfect balance (null median)", "{:16.2f}"),
                           ("points", "gated sweep points", "{:16d}"),
                           ("n_eff", "median n_eff", "{:16.0f}"),
                           ("min_occupied", "min variants occupied", "{:16d}")):
        print(f"{name:32s}" + "".join(fmt.format(rows[ph.name][key]) for ph in PHASES))
    for path, s in zip(args.seeds, seeds):
        print(f"{'FrOGS weighted':32s}" + "".join(f"{s[ph.name]['weighted']:16.2f}" for ph in PHASES)
              + f"   ({path})")

    out = args.out or figure_path("tab1_variant_chi2.tex")
    with open(out, "w") as f:
        f.write(to_latex(rows, seeds))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
