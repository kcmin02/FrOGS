"""The autoregressive prior: flip scores, the cached sampler and the fused systems backend."""
import copy

import pytest
import torch

from frogs.nets import ARConvPrior
from frogs.nets.systems.sampling import CachedSampler, sample_cached

requires_fused_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8,
    reason="the fused kernels need an Ampere-or-newer CUDA GPU")


def make_prior(shape=(2, 3, 2), channels=5, layers=3, kernel_size=3, device="cpu"):
    torch.manual_seed(8921)
    prior = ARConvPrior(shape, num_layers=layers, num_channels=channels, kernel_size=kernel_size)
    with torch.no_grad():  # FiLM starts as the identity; give it a real c-dependence
        for film in (prior.film_initial, *prior.films):
            film.net[-1].weight.normal_(0, 0.08)
            film.net[-1].bias.normal_(0, 0.08)
    return prior.to(device)


def spins(n, shape, device="cpu", dtype=torch.float32):
    return (2 * torch.randint(0, 2, (n, *shape), device=device) - 1).to(dtype)


@torch.no_grad()
def test_flip_scores_match_explicit_flips():
    prior = make_prior().double()
    s, c = spins(3, prior.lattice_shape, dtype=torch.float64), torch.randn(3, 2).double()
    scores = prior.delta_log_prob_all_sites(s, c)
    base = prior.log_prob(s, c)
    for site in range(prior.N):
        flipped = s.reshape(3, -1).clone()
        flipped[:, site] *= -1
        direct = prior.log_prob(flipped.reshape(s.shape), c) - base
        assert torch.allclose(scores[:, site], direct, atol=1e-12)


@torch.no_grad()
def test_cached_site_logits_match_the_full_forward():
    prior = make_prior()
    c = torch.tensor([[0.1, -0.4], [0.7, 0.2], [-0.3, 0.9]])
    target = spins(3, prior.lattice_shape)
    partial = torch.zeros_like(target)
    cache = CachedSampler(prior, c, 3)
    for site in range(prior.N):
        i, rem = divmod(site, prior.lattice_shape[1] * prior.lattice_shape[2])
        j, k = divmod(rem, prior.lattice_shape[2])
        assert torch.allclose(cache.next_logits(), prior.logits(partial, c)[:, i, j, k],
                              rtol=2e-6, atol=1e-7)
        cache.commit(target[:, i, j, k])
        partial[:, i, j, k] = target[:, i, j, k]
    assert torch.equal(cache.result(), target)


@torch.no_grad()
def test_cached_sampler_reproduces_reference_draws_and_rng():
    prior = make_prior()
    c = torch.tensor([0.23, -0.71])
    torch.manual_seed(99173)
    reference = prior.sample(c, 4)
    rng = torch.random.get_rng_state()
    torch.manual_seed(99173)
    cached = sample_cached(prior, c, 4)
    assert torch.equal(cached, reference)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert cached.is_contiguous()


def test_systems_backend_falls_back_to_the_reference_on_cpu():
    """CPU and autograd calls must give the reference result, gradients included."""
    prior = make_prior().double()
    s = spins(2, prior.lattice_shape, dtype=torch.float64)
    c = torch.randn(2, 2, dtype=torch.float64, requires_grad=True)
    reference = prior.delta_log_prob_all_sites_reference(s, c)
    prior.backend = "systems"
    actual = prior.delta_log_prob_all_sites(s, c)
    assert torch.equal(actual, reference)
    params = (c, *prior.parameters())
    for a, b in zip(torch.autograd.grad(actual.square().sum(), params),
                    torch.autograd.grad(reference.square().sum(), params)):
        assert torch.equal(a, b)

    prior = make_prior()
    before = {k: v.clone() for k, v in prior.state_dict().items()}
    torch.manual_seed(44)
    reference = prior.sample([0.2, -0.3], 2)
    prior.backend = "systems"
    torch.manual_seed(44)
    assert torch.equal(prior.sample([0.2, -0.3], 2), reference)
    after = prior.state_dict()
    assert after.keys() == before.keys()
    assert all(torch.equal(after[k], before[k]) for k in before)


@requires_fused_gpu
@pytest.mark.parametrize("shape,channels,layers,kernel_size", [
    ((1, 1, 1), 3, 1, 3), ((2, 3, 2), 5, 1, 3), ((1, 4, 3), 32, 4, 3), ((3, 2, 3), 5, 3, 3),
    ((2, 2, 3), 64, 2, 3), ((2, 2, 3), 5, 2, 1), ((2, 2, 3), 5, 2, 5)])
@torch.no_grad()
def test_fused_flip_scores_match_fp64(shape, channels, layers, kernel_size):
    prior = make_prior(shape, channels, layers, kernel_size, device="cuda")
    prior.backend = "systems"
    s, c = spins(3, shape, device="cuda"), torch.randn(3, 2, device="cuda")
    actual = prior.delta_log_prob_all_sites(s, c)
    reference = copy.deepcopy(prior).double().delta_log_prob_all_sites_reference(
        s.double(), c.double())
    torch.testing.assert_close(actual.double(), reference, rtol=3e-4, atol=2e-5)


@requires_fused_gpu
@pytest.mark.parametrize("shape,channels,layers", [
    ((1, 3, 2), 5, 1), ((2, 3, 2), 32, 4), ((2, 2, 3), 64, 2)])
@torch.no_grad()
def test_fused_sampler_matches_every_conditional_and_the_rng(monkeypatch, shape, channels, layers):
    from frogs.nets.systems.triton_kernels import sample_triton
    prior = make_prior(shape, channels, layers, device="cuda")
    c = torch.randn(3, 2, device="cuda")
    recorded, bernoulli = [], torch.bernoulli

    def record(p):
        recorded.append(p.clone())
        return bernoulli(p)

    monkeypatch.setattr(torch, "bernoulli", record)
    torch.manual_seed(789)
    reference = prior.sample(c, 3)
    rng = torch.cuda.get_rng_state()
    probs = torch.stack(recorded)
    recorded.clear()
    torch.manual_seed(789)
    actual = sample_triton(prior, c, 3)
    assert torch.equal(torch.cuda.get_rng_state(), rng)
    assert torch.equal(actual, reference)
    torch.testing.assert_close(torch.stack(recorded), probs, rtol=2e-5, atol=2e-6)


@requires_fused_gpu
@torch.no_grad()
def test_graph_replay_preserves_rng_and_follows_live_inputs_and_weights():
    from frogs.nets.systems import triton_kernels
    prior = make_prior(device="cuda")
    prior.backend = "systems"
    c = torch.randn(3, 2, device="cuda")
    torch.manual_seed(912)
    reference = triton_kernels.sample_triton(prior, c, 3)
    rng = torch.cuda.get_rng_state()
    torch.manual_seed(912)
    sample = prior.sample(c, 3)
    assert torch.equal(torch.cuda.get_rng_state(), rng)
    assert torch.equal(sample, reference)
    held_sample, score = sample.clone(), prior.delta_log_prob_all_sites(sample, c)
    held_score = score.clone()
    probe = sample.clone()
    for _ in range(2):
        probe.neg_()
        c.add_(0.3)
        prior.head.weight.add_(0.2)
        prior.input_conv.weight.mul_(1.1)
        expected = triton_kernels.delta_log_prob_all_sites(prior, probe, c)
        assert torch.equal(prior.delta_log_prob_all_sites(probe, c), expected)
        torch.manual_seed(814)
        expected = triton_kernels.sample_triton(prior, c, 3)
        rng = torch.cuda.get_rng_state()
        torch.manual_seed(814)
        assert torch.equal(prior.sample(c, 3), expected)
        assert torch.equal(torch.cuda.get_rng_state(), rng)
    assert torch.equal(score, held_score) and torch.equal(sample, held_sample)
    assert "_graph_cache" not in copy.deepcopy(prior).__dict__


@requires_fused_gpu
@torch.no_grad()
def test_graph_cache_is_bounded_and_fp64_uses_the_reference():
    from frogs.nets.systems.cuda_graphs import MAX_GRAPHS
    prior = make_prior(device="cuda")
    prior.backend = "systems"
    for batch in range(1, MAX_GRAPHS + 3):
        s = torch.ones(batch, *prior.lattice_shape, device="cuda")
        c = torch.randn(batch, 2, device="cuda")
        prior.delta_log_prob_all_sites(s, c)
    assert len(prior._graph_cache) == MAX_GRAPHS
    prior.double()
    s, c = s.double(), c.double()
    assert torch.equal(prior.delta_log_prob_all_sites(s, c),
                       prior.delta_log_prob_all_sites_reference(s, c))


@requires_fused_gpu
@torch.no_grad()
def test_graph_replay_at_training_batch_sizes():
    from frogs.nets.systems import triton_kernels
    prior = make_prior((8, 8, 8), 32, 4, device="cuda")
    prior.backend = "systems"
    for batch in (50, 100, 50, 100):
        s, c = spins(batch, (8, 8, 8), device="cuda"), torch.randn(batch, 2, device="cuda")
        expected = triton_kernels.delta_log_prob_all_sites(prior, s, c)
        for _ in range(2):
            assert torch.equal(prior.delta_log_prob_all_sites(s, c), expected)
