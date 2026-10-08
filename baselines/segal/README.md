# SEGAL baseline (Fig. 3c, 3d)

SEGAL (Damewood, Schwalbe-Koda and Gómez-Bombarelli, *npj Comput. Mater.* **8**, 61 (2022)) is an
autoregressive network over the lattice sites, conditioned on the temperature and a field coupled to
sum(sigma). We do not retrain it. `model.py` re-implements the network so that the released weights
load unchanged, and `sweep.py` draws M = 5000 samples per condition and writes their importance
weights as a sweep file, which the phase diagram constructor (§A.4) reads like a FrOGS sweep.

## Checkpoints

The weights are not redistributed here. Clone the upstream repository (MIT License; we used commit
`d2f2ad4`):

    git clone https://github.com/learningmatter-mit/Segal
    SEGAL=Segal/Revision_Submission/Phase_Diagram

`$SEGAL/AgPdCE_125_final_submission` is the 125-site AgPd model and
`$SEGAL/CuAu_128_Final_Submission` the 128-site CuAu model.

## Reproducing Fig. 3c and 3d

    python baselines/segal/sweep.py --system agpd --ckpt $SEGAL/AgPdCE_125_final_submission
    python baselines/segal/sweep.py --system cuau --ckpt $SEGAL/CuAu_128_Final_Submission

    python scripts/figures/fig3_phase_diagram.py --sweep results/sweeps/segal_agpd_125.npz --label SEGAL
    python scripts/figures/fig3_phase_diagram.py --sweep results/sweeps/segal_cuau_128.npz --label SEGAL

Each sweep takes about ten minutes on one GPU (seed 0). The grids are SEGAL's field grids, 81 points
on [-0.2, 0.2] for AgPd and 36 on [-0.1, 0.075] for CuAu (dmu steps of 0.01 eV), with T in 25 K
steps over 200-900 K and 200-1200 K. `--field`, `--T` and `--M` select other grids.

## Conventions

* dmu = 2 field for AgPd. The CuAu notebook subtracts the composition-weighted pure-element
  energies, which shifts it to dmu = 2 field + (U_Au - U_Cu)/N = 2 field - 5.33 meV.
* A = -beta U_CE + beta dmu N_+ - log P_AR, with U_CE from `frogs.energy.ClusterExpansion` and the
  cluster tables in `data/`. P_AR is normalized, so log Z = log mean e^A (`logZ_raw`; `logZ` is the
  second-cumulant estimate, as in the FrOGS sweeps).
* SEGAL orders sites like the atoms of the CLEASE supercell; `sweep.py` permutes them to the
  primitive-coordinate order of the cluster tables before evaluating the energy.
