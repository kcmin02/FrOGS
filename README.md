# Official implementation of FrOGS

This is the official implementation of [FrOGS: Discrete Neural Sampler for Independent Alloy
Configurations Across Chemical Conditions](https://arxiv.org/abs/2609.02948), a work by Kyucheol Min,
Elyssa Hofgard, and Tess Smidt.

FrOGS couples an autoregressive prior to a learned continuous-time Markov chain built on
[LEAPS](https://github.com/malbergo/leaps). One trained model samples the semi-grand-canonical
ensemble over a whole range of chemical conditions (Δμ, T) and returns free energies on a common
absolute scale.

## Setup environment

```bash
conda env create -f environment.yml
conda activate frogs
pytest
```

Training and sweeps need a GPU. The fused prior kernels (`prior_backend: systems`, see
[docs/performance.md](docs/performance.md)) run on Ampere-or-newer GPUs; elsewhere the prior uses
plain PyTorch.

## Train models

Each config in `configs/` trains one model of the paper; runs are written to `results/`.

```bash
python scripts/train.py --config configs/cuau_128.yml      # CuAu, 128 sites
python scripts/train.py --config configs/agpd_125.yml      # AgPd, 125 sites
python scripts/train.py --config configs/ising2d_L20.yml   # 2D Ising, L = 20 (also L10, L15)
```

The ablations (Fig. 6, 7) are in `configs/ablations/`, the hyperparameter arms of §A.11 in
`configs/hparams/`, and the extra CuAu seeds of Fig. 10 are run with
`--set seed=2 run_tag=cuau128-s2` (and 3, 4). Continue an interrupted run with `--resume <ckpt>`.

## Sweep chemical conditions

```bash
python scripts/sweep.py --ckpt results/cuau_ce/cuau128-1/ckpts/last.ckpt \
    --n-mu 36 --n-T 41 --M 5000 --store-A --store-sigma --tag cuau128
python scripts/sweep.py --ckpt results/agpd_ce/agpd125-1/ckpts/last.ckpt \
    --n-mu 81 --n-T 29 --M 5000 --store-A --store-sigma --tag agpd125
```

This writes `results/sweeps/<tag>.npz`. Large sweeps can be split with `--slice I N` and joined
with `scripts/merge_sweep.py --tag <tag>`. The SEGAL baseline sweeps are described in
[baselines/segal](baselines/segal/README.md).

## Reproduce figures and tables

- Fig. 2, Fig. 9: `scripts/figures/fig2_ising.py --ckpt <ising ckpt>` for L = 10, 15 and 20, then `scripts/figures/fig9_ising_multiscale.py`
- Fig. 3: `scripts/figures/fig3_phase_diagram.py --sweep results/sweeps/cuau128.npz` (add `--label SEGAL` for SEGAL sweeps)
- Fig. 4, Fig. 6: `scripts/figures/fig4_ess_map.py --sweep <sweep> ...`
- Fig. 7: `scripts/figures/fig7_variant_coverage.py --sweep <uniform-prior sweep>`
- Fig. 10: `scripts/figures/fig10_seed_panels.py --panel "seed 1=<sweep>" ...`
- Table 1: `scripts/tables/tab1_variant_chi2.py --sweep results/sweeps/cuau128.npz`
- §A.11 tables: `scripts/tables/training_stats.py results/*/hparams-*/metrics.csv` (train the
  `configs/hparams/` runs with `--offline`, which writes `metrics.csv`)
- §A.13: `scripts/tables/gate_sensitivity.py --arm FrOGS=<sweep> --arm SEGAL=<segal sweep>`
- §A.7 and other checks: `scripts/checks/`

The conventional Metropolis measurements of §A.14 are not part of this repository.

## Run experiments for a different energy model

A new system needs its Hamiltonian H_c(σ). Use `frogs/energy/alloy.py` (cluster expansions) and
`frogs/energy/ising.py` as templates, register the class in `SYSTEMS` (`frogs/config.py`) and
`make_energy` (`frogs/energy/__init__.py`), and use `scripts/tools/dump_cluster_tables.py` to build
the cluster table of a new cell.

## Analysis of trained models

```python
from frogs.evaluate import ISEvaluator, load_model

model = load_model("results/cuau_ce/cuau128-1/ckpts/last.ckpt")
out = ISEvaluator(model, M=1000).run(dmu=[0.0], T=[500.0])
print(out["logZ"], out["x"], out["ess"])   # log Z, composition and ESS at (dmu, T)
```

## Citation

```bibtex
@article{min2026frogs,
  title   = {{FrOGS}: Discrete Neural Sampler for Independent Alloy Configurations Across Chemical Conditions},
  author  = {Min, Kyucheol and Hofgard, Elyssa and Smidt, Tess},
  journal = {arXiv preprint arXiv:2609.02948},
  year    = {2026}
}
```

All alloy results are the thermodynamics of the cluster expansions of Damewood et al. (2022),
which are redistributed in `data/reference/` (see `NOTICE`). The code is released under the MIT
License.
