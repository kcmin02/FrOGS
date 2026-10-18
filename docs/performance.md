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
32 channels), batch 3, normalized conditions (-1, -1), (0, 0), (1, 1):

```bash
python scripts/checks/check_systems_accuracy.py --ckpt <run>/ckpts/last.ckpt [--shape 8 8 8]
```

| lattice | quantity | systems | reference |
| --- | --- | ---: | ---: |
| 8x4x4 | conditional probability, every site of a sampled trajectory | 1.7e-7 | 1.3e-4 |
| 8x4x4 | all-site flip score | 5.6e-6 | 3.6e-3 |
| 8x8x8 | conditional probability | 2.3e-7 | 3.8e-4 |
| 8x8x8 | all-site flip score | 7.5e-6 | 5.9e-3 |

The reference error comes from cuDNN's default TF32 convolutions; with
`torch.backends.cudnn.allow_tf32 = False` it drops to 2.3e-7 / 1.2e-5 on 8x4x4. At every
condition the graph-replayed sampler reproduced the reference samples and CUDA RNG state.

## Speed

```bash
python scripts/bench_speed.py --ckpt <run>/ckpts/last.ckpt --compile
```

CuAu production checkpoint, batch 16, no Metropolis moves, compiled prior, mean of 7 repetitions
after 3 warm-ups, one RTX A5000 shared with another job (timings are indicative):

| operation | reference (ms) | systems (ms) | speedup |
| --- | ---: | ---: | ---: |
| AR sampling | 62.1 | 19.1 | 3.3x |
| all-site flip scoring | 11.8 | 5.3 | 2.2x |
| rollout, 8 CTMC steps, including sampling | 278.9 | 214.2 | 1.3x |
| loss + backward on a replayed trajectory | 67.4 | 71.6 | 0.9x |

Repeated runs on the shared GPU varied by up to a factor 1.7 in absolute time. Peak allocated GPU
memory was 0.11 GiB. The loss step uses the backend only for its no-grad Delta U_t, so it gains
nothing at 128 sites; the reference flip scorer evaluates B x N flipped lattices, so the gain in
scoring and rollouts grows with the cell size.
