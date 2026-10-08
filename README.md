# FrOGS

Code for **FrOGS: Discrete Neural Sampler for Independent Alloy Configurations Across Chemical
Conditions** (K. Min, E. Hofgard, T. Smidt; [arXiv:2609.02948](https://arxiv.org/abs/2609.02948)).

FrOGS couples an autoregressive prior ρ₀(σ | c) to a learned continuous-time Markov chain that
transports it to the Boltzmann distribution of a lattice model, and trains both under one loss.
One model covers a whole region of chemical conditions c = (Δμ, T). It draws i.i.d.
configurations with importance weights e^A, and E[e^A] = Z(c) gives the semi-grand-canonical free
energy at every condition on a common absolute scale. The transport builds on
[LEAPS](https://arxiv.org/abs/2502.10843).

## Installation

```bash
conda create -n frogs python=3.10 -y && conda activate frogs
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[dev]"            # add ".[wandb]" to log to Weights & Biases
pytest                             # ~150 tests, about 30 s; Triton tests run only on a GPU
```

Tested with Python 3.9, PyTorch 2.8.0 (CUDA 12.8), Lightning 2.6.0, NumPy 1.23.5, SciPy 1.13 and
Matplotlib 3.9. Training and sweeps need a GPU. The fused prior kernels
([docs/performance.md](docs/performance.md)) need an Ampere-or-newer GPU and the Triton that ships
with PyTorch; elsewhere the plain PyTorch path is used.

## Layout

```
frogs/
  energy/          2D Ising and fcc cluster-expansion energies, annealing paths
  nets/            autoregressive prior, locally equivariant U-Net rate matrix, FiLM, F-network
  nets/systems/    fused Triton kernels and CUDA-graph replay for the prior
  sampler.py       CTMC transport and discrete Jarzynski log-weights
  trainer.py       joint training (Lightning module)
  evaluate.py      importance-sampling readout on a grid of conditions
  analysis/        phase-diagram constructor (§A.4), variant statistics, Ising exact results
configs/           one YAML per trained model (ablations/, hparams/)
scripts/           train.py, sweep.py, merge_sweep.py, figures/, tables/, checks/, tools/
baselines/segal/   the SEGAL baseline of Fig. 3c-d
data/              cluster-expansion tables and reference data (data/README.md)
tests/
```

## Reproducing the paper

Each result is a training run, an importance-sampling sweep of the trained model, and a figure or
table script that reads the sweep. Runs go to `results/`, figures to `figures/` (override with
`FROGS_RESULTS_DIR` / `FROGS_FIGURES_DIR`).

### 1. Train

```bash
python scripts/train.py --config configs/cuau_128.yml       # CuAu, 128 sites
python scripts/train.py --config configs/agpd_125.yml       # AgPd, 125 sites
python scripts/train.py --config configs/ising2d_L20.yml    # 2D Ising (also ising2d_L10, _L15)
```

A run is 200k steps (about 8 h for CuAu on one RTX A5000) and writes
`results/<system>/<run_tag>-<n>/ckpts/last.ckpt`; continue an interrupted run with
`--resume <ckpt>`. Logs go to Weights & Biases if it is installed, otherwise (or with `--offline`)
to `metrics.csv` in the run directory. The other trained models:

```bash
python scripts/train.py --config configs/ablations/cuau_128_no_transport.yml   # Fig. 6a
python scripts/train.py --config configs/ablations/cuau_128_uniform_prior.yml  # Fig. 6b, 7
for s in 2 3 4; do                                                             # Fig. 10
  python scripts/train.py --config configs/cuau_128.yml --set seed=$s run_tag=cuau128-s$s
done
for c in configs/hparams/*.yml; do                                             # §A.11 tables
  python scripts/train.py --config $c --offline
done
```

### 2. Sweep the conditions

```bash
python scripts/sweep.py --ckpt results/cuau_ce/cuau128-1/ckpts/last.ckpt \
    --n-mu 36 --n-T 41 --M 5000 --store-A --store-sigma --tag cuau128
python scripts/sweep.py --ckpt results/agpd_ce/agpd125-1/ckpts/last.ckpt \
    --n-mu 81 --n-T 29 --M 5000 --store-A --store-sigma --tag agpd125
```

The grid spans the model's training box at 10 meV × 25 K, with no Metropolis moves at evaluation
(`--n-mcmc 0`, the default). Sweep the ablation and seed models with the CuAu command and their own
`--tag`. A sweep costs roughly a minute per grid point on one A100. Split it over a job array with
`--slice $SLURM_ARRAY_TASK_ID 36` and reassemble it with `python scripts/merge_sweep.py --tag cuau128`;
an interrupted sweep resumes from its per-point shards. `--backend systems` selects the fused prior
kernels.

The SEGAL sweeps of Fig. 3c-d use the released SEGAL weights; see
[baselines/segal/README.md](baselines/segal/README.md).

### 3. Figures and tables

| Result | Command |
| --- | --- |
| Fig. 2 | `python scripts/figures/fig2_ising.py --ckpt results/ising/ising-L20-1/ckpts/last.ckpt` |
| Fig. 3a-b | `python scripts/figures/fig3_phase_diagram.py --sweep results/sweeps/agpd125.npz` (and `cuau128.npz`) |
| Fig. 3c-d | `python scripts/figures/fig3_phase_diagram.py --sweep results/sweeps/segal_cuau_128.npz --label SEGAL` (and `segal_agpd_125.npz`) |
| Fig. 4 | `python scripts/figures/fig4_ess_map.py --sweep results/sweeps/agpd125.npz results/sweeps/cuau128.npz` |
| Fig. 6 | `python scripts/figures/fig4_ess_map.py --sweep <no-transport>.npz <uniform-prior>.npz --labels "autoregressive only" "transport only"` |
| Fig. 7 | `python scripts/figures/fig7_variant_coverage.py --sweep <uniform-prior>.npz` |
| Fig. 9 | `python scripts/figures/fig9_ising_multiscale.py --results results/ising/ising_L{10,15,20}.npz` |
| Fig. 10 | `python scripts/figures/fig10_seed_panels.py --panel "seed 1=results/sweeps/cuau128.npz" --panel "seed 2=..." ...` |
| Table 1 (§5.3) | `python scripts/tables/tab1_variant_chi2.py --sweep results/sweeps/cuau128.npz --seeds results/sweeps/cuau128-s{2,3,4}.npz` |
| §A.7 checks | `python scripts/checks/check_local_equivariance.py --ckpt <ckpt>` |
| §A.11 tables | `python scripts/tables/training_stats.py results/*/tab*/metrics.csv` |
| §A.13 | `python scripts/tables/gate_sensitivity.py --arm FrOGS=results/sweeps/cuau128.npz --arm SEGAL=results/sweeps/segal_cuau_128.npz` |
| §A.14 (FrOGS ESS column) | `python scripts/tables/tab4_lowest_ess.py results/sweeps/cuau128.npz` |

`fig2_ising.py` rolls out its own walkers (M = 5000 at 14 temperatures, with and without the
transport; 10-15 GPU-hours at L = 20) and saves them to `results/ising/ising_L<L>.npz`, which
`--results` replots and Fig. 9 reads. The phase-diagram constructor of §A.4,
`frogs/analysis/phase_diagram.py`, discovers the number and composition of the phases from the
samples alone.

### Checks

```bash
python scripts/checks/check_exactness.py                   # Z_0 = 1 by enumerating a 16-site cell
python scripts/checks/check_local_equivariance.py --ckpt <ckpt>
python scripts/checks/check_systems_accuracy.py --ckpt <ckpt>
```

## Notes

All alloy results are the thermodynamics of the cluster expansions of Damewood et al. (2022)
(`data/reference/`), which differ from the experimental phase diagrams. The Metropolis-form rates
of Eq. (7) are used for the alloys; the Ising models use the LEAPS rates of Eq. (4). The
conventional Metropolis measurements of §A.14 are not part of this repository.

## Citation

```bibtex
@article{min2026frogs,
  title   = {{FrOGS}: Discrete Neural Sampler for Independent Alloy Configurations Across Chemical Conditions},
  author  = {Min, Kyucheol and Hofgard, Elyssa and Smidt, Tess},
  journal = {arXiv preprint arXiv:2609.02948},
  year    = {2026}
}
```

## License

MIT (see `LICENSE`). Third-party code and data are attributed in `NOTICE`.
