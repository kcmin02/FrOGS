# Systems backend for the autoregressive prior

`prior_backend: systems` runs the no-grad operations of the prior rho_0(sigma | c) through fused
Triton kernels replayed as CUDA graphs (`frogs/nets/systems/`): ancestral sampling, and the
all-site flip scores log rho_0(flip_i sigma | c) - log rho_0(sigma | c) that enter Delta U_t at
every CTMC step. Calls that carry gradients (log rho_0 in the training loss), CPU tensors and
non-FP32 inputs keep the reference PyTorch path. It needs an Ampere-or-newer GPU and at most 64
prior channels; fused sampling also needs the 3x3x3 kernel.

## Enabling it

* Training: `prior_backend: systems` in the config (the shipped alloy configs set it).
* Sweeps: `python scripts/sweep.py --ckpt <run>/ckpts/last.ckpt --backend systems ...`
  (without `--backend` a sweep uses the backend the model was trained with).

## What it changes

FP32 rounding only. Model, weights, walkers, CTMC steps and Metropolis moves are unchanged; the
sampler still makes one `torch.bernoulli` call per site in raster order, and graph capture leaves
the CUDA RNG state untouched. Matrix products use Triton's FP32-accurate `tf32x3` mode and the
log-weights still accumulate in FP64. Results are not bitwise identical to the reference backend.

## Accuracy

Maximum absolute error against an FP64 copy of the prior, CuAu production checkpoint (8x4x4,
64 channels), batch 3, normalized conditions (-1, -1), (0, 0), (1, 1):

```bash
python scripts/checks/check_systems_accuracy.py --ckpt <run>/ckpts/last.ckpt [--shape 8 8 8]
```

| lattice | quantity | systems | reference |
| --- | --- | ---: | ---: |
| 8x4x4 | conditional probability, every site of a sampled trajectory | 2.4e-7 | 1.9e-4 |
| 8x4x4 | all-site flip score | 6.6e-6 | 1.8e-3 |
| 8x8x8 | conditional probability | 4.1e-7 | 2.1e-4 |
| 8x8x8 | all-site flip score | 1.1e-5 | 3.9e-3 |

The reference error comes from cuDNN's default TF32 convolutions; with
`torch.backends.cudnn.allow_tf32 = False` it drops to 4.1e-7 / 1.6e-5 on 8x4x4. At every
condition the graph-replayed sampler reproduced the reference samples and CUDA RNG state.

## Speed

```bash
python scripts/bench_speed.py --ckpt <run>/ckpts/last.ckpt --compile
```

CuAu production checkpoint, batch 16, no Metropolis moves, compiled prior, mean of 7 repetitions
after 3 warm-ups, one RTX A5000 shared with another job (timings are indicative):

| operation | reference (ms) | systems (ms) | speedup |
| --- | ---: | ---: | ---: |
| AR sampling | 44.3 | 20.2 | 2.2x |
| all-site flip scoring | 13.2 | 7.1 | 1.9x |
| rollout, 8 CTMC steps, including sampling | 233.1 | 160.0 | 1.5x |
| loss + backward on a replayed trajectory | 61.4 | 51.5 | 1.2x |

Peak allocated GPU memory was 0.21 GiB. The loss step uses the backend only for its no-grad
Delta U_t, so it gains little at 128 sites; the reference flip scorer evaluates B x N flipped
lattices, so the gain in scoring and rollouts grows with the cell size.
