import itertools

import numpy as np
import pytest

from frogs.analysis import variants as V


def _variant(spins, phase):
    gated, variant = V.classify(spins)[phase]
    return bool(gated[0]), int(variant[0])


def _translations():
    return itertools.product(*[range(s) for s in V.LATTICE])


def _roll(spins, t):
    return np.roll(spins.reshape(V.LATTICE), t, axis=(0, 1, 2)).reshape(-1)


def _l10_configs():
    return [sign * V.perfect_l10(o) for o in range(3) for sign in (1, -1)]


def test_perfect_l10_fills_six_variants():
    seen = set()
    for k, s in enumerate(_l10_configs()):
        gated, v = _variant(s, "CuAu")
        assert gated and v // 2 == k // 2
        seen.add(v)
        assert not V.classify(s)["Cu3Au"][0][0] and not V.classify(s)["CuAu3"][0][0]
    assert seen == set(range(6))


@pytest.mark.parametrize("phase,minority", [("Cu3Au", 1), ("CuAu3", -1)])
def test_perfect_l12_maps_to_its_sublattice(phase, minority):
    for s in range(4):
        spins = V.perfect_l12(s, minority)
        assert _variant(spins, phase) == (True, s)
        assert not V.classify(spins)["CuAu"][0][0]


def test_perfect_order_parameters():
    for s in _l10_configs():
        S = np.abs(V.structure_factors(s)) ** 2 / V.N_SITES
        assert np.isclose(S.max(), V.S_PERFECT_L10)
    for s in range(4):
        S = np.abs(V.structure_factors(V.perfect_l12(s))) ** 2 / V.N_SITES
        assert np.allclose(S, V.S_PERFECT_L12 / 3)


@pytest.mark.parametrize("phase,configs", [
    ("CuAu", _l10_configs()),
    ("Cu3Au", [V.perfect_l12(s, 1) for s in range(4)]),
    ("CuAu3", [V.perfect_l12(s, -1) for s in range(4)]),
])
def test_translations_permute_variants(phase, configs):
    base = [_variant(s, phase)[1] for s in configs]
    orbit = set()
    for t in _translations():
        image = [_variant(_roll(s, t), phase) for s in configs]
        assert all(g for g, _ in image)
        image = [v for _, v in image]
        assert sorted(image) == sorted(base)
        if phase == "CuAu":
            assert [v // 2 for v in image] == [v // 2 for v in base]
        orbit.add(image[0])
    # translations reach every anti-phase domain (L1_0) or sublattice (L1_2) of a variant
    assert orbit == ({base[0], base[1]} if phase == "CuAu" else set(base))


def test_defects_keep_variant_and_disorder_is_not_gated():
    rng = np.random.default_rng(0)
    s = V.perfect_l10(1).copy()
    au, cu = np.flatnonzero(s > 0), np.flatnonzero(s < 0)
    flip = np.concatenate([rng.choice(au, 3, replace=False), rng.choice(cu, 3, replace=False)])
    s[flip] *= -1
    assert _variant(s, "CuAu") == _variant(V.perfect_l10(1), "CuAu")
    disordered = rng.permutation(np.repeat([1.0, -1.0], V.N_SITES // 2))
    assert not any(g[0] for g, _ in V.classify(disordered).values())


def test_chi2_statistics():
    assert V.chi2_per_dof(np.array([7, 7, 7, 7])) == 0.0
    counts = np.array([[10, 20, 30, 40], [25, 25, 25, 25]], dtype=float)
    chi2, n_eff = V.weighted_chi2_per_dof(counts / 200.0, np.full(2, 100 / 200.0 ** 2))
    assert np.allclose(n_eff, 100.0)
    assert np.allclose(chi2, V.chi2_per_dof(counts))
    assert np.isnan(V.chi2_per_dof(np.zeros(6)))
    assert round(V.null_median(4), 2) == 0.79 and round(V.null_median(6), 2) == 0.87


def test_variant_histograms_on_a_packed_sweep():
    l10 = _l10_configs()
    l12 = [V.perfect_l12(s, 1) for s in range(4)]
    spins = np.stack([np.stack(l10), np.stack(l10[:2] + l12)])     # (n_T=2, M=6, N)
    packed = np.packbits((spins > 0).astype(np.uint8), axis=-1)[None]
    A = np.zeros((1, 2, 6))
    A[0, 1, 2] = np.log(3.0)
    sweep = dict(A_samples=A, N_samples=(spins > 0).sum(-1)[None].astype(float),
                 sigma_packed=packed, meta={"N": V.N_SITES})
    stats = V.variant_histograms(sweep)
    assert stats["CuAu"].counts[0, 0].tolist() == [1] * 6
    assert stats["Cu3Au"].counts[0, 1].tolist() == [1] * 4
    assert np.allclose(stats["Cu3Au"].wsum[0, 1], [3 / 8, 1 / 8, 1 / 8, 1 / 8])
    assert stats["CuAu"].n_gated[0, 1] == 2
    points = V.deepest_ordered_points(stats, np.array([300.0, 200.0]))
    assert tuple(points["CuAu"]) == (0, 0) and tuple(points["Cu3Au"]) == (0, 1)
