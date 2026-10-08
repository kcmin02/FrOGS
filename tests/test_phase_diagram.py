"""Unit tests of the §A.4 phase-diagram constructor on small synthetic inputs."""
import numpy as np
import pytest

from frogs.analysis.phase_diagram import (
    Gap, PhaseDiagram, apply_persistence, certify, construct_phase_diagram, find_lobes,
    fixed_composition_free_energy, gate_gaps, hull_gaps, lower_hull, ordered_compounds, site_count)
from frogs.constants import K_B

N_SITES = 16


def double_well(x, depth=0.003125):
    """Symmetric double well with minima 0 at x = 1/4 and 3/4 and f(1/2) = depth."""
    return depth * 256 * (x - 0.25) ** 2 * (x - 0.75) ** 2


def synthetic_sweep(f_rows, T, dmu, n_sites=N_SITES, k=20):
    """Sweep whose samples reproduce f(x, T) exactly: k walkers per composition per condition."""
    n = np.repeat(np.arange(n_sites + 1), k).astype(float)
    A = np.zeros((len(dmu), len(T), n.size))
    for j, (f, t) in enumerate(zip(f_rows, T)):
        beta = 1.0 / (K_B * t)
        ln_z = -beta * n_sites * f
        for i, mu in enumerate(dmu):
            A[i, j] = np.repeat(ln_z + beta * mu * np.arange(n_sites + 1)
                                + np.log(n_sites + 1), k)
    N = np.broadcast_to(n, A.shape).copy()
    return dict(dmu=np.asarray(dmu, float), T=np.asarray(T, float), A_samples=A, N_samples=N,
                meta=dict(N=n_sites))


def test_lower_hull_drops_points_on_or_above_the_chain():
    x = np.arange(7) / 8
    y = (x - 0.375) ** 2
    assert lower_hull(x, y) == list(range(7))
    y[4] += 0.1
    assert lower_hull(x, y) == [0, 1, 2, 3, 5, 6]
    assert lower_hull(np.arange(4.0), np.zeros(4)) == [0, 3]


def test_hull_gaps_double_well():
    x = np.arange(N_SITES + 1) / N_SITES
    f, s = double_well(x), 1e-4
    (gap,) = hull_gaps(f, np.full_like(f, s), N_SITES)
    assert (gap.x_lo, gap.x_hi, gap.n_missing) == (0.25, 0.75, 0)
    assert gap.depth == pytest.approx(0.003125)
    assert gap.se == pytest.approx(s * np.sqrt(1.5))

    f[5:12] = np.nan
    (gap,) = hull_gaps(f, np.full_like(f, s), N_SITES)
    assert (gap.x_lo, gap.x_hi, gap.depth, gap.n_missing) == (0.25, 0.75, np.inf, 7)


def test_gate_gaps_width_depth_and_measured_absence():
    narrow = Gap(0.0, 3 / 16, 1.0, 0.01, 0)
    shallow = Gap(0.25, 0.5, 0.02, 0.01, 0)
    deep = Gap(0.5, 0.75, 0.04, 0.01, 0)
    unsampled = Gap(0.75, 1.0, np.inf, 0.0, 3)
    gaps = [narrow, shallow, deep, unsampled]
    assert gate_gaps(gaps, N_SITES) == ([deep, unsampled], 0)
    assert gate_gaps(gaps, N_SITES, measured_absence=False) == ([deep], 1)
    assert gate_gaps(gaps, N_SITES, w_min=3, n_sigma=1) == (gaps, 0)


def test_persistence_rules():
    certified = {0: [Gap(0.2, 0.5, 1, 0.1, 0)], 1: []}
    cand = [Gap(0.3, 0.4, 1, 0.1, 0), Gap(0.6, 0.8, 1, 0.1, 0)]
    assert apply_persistence(cand, certified, 2, "below") == []
    assert apply_persistence(cand, certified, 2, "below_gap1") == cand[:1]
    assert apply_persistence(cand, certified, 2, "off") == cand
    assert apply_persistence(cand, {}, 0, "below") == cand
    with pytest.raises(ValueError):
        apply_persistence(cand, certified, 2, "above")


def test_persistence_is_a_chain():
    deep, shallow = Gap(0.25, 0.75, 1.0, 0.1, 0), Gap(0.25, 0.75, 0.1, 0.1, 0)
    rows = [[deep], [shallow], [deep], [deep]]
    below, _ = certify(rows, N_SITES, persist="below")
    gap1, _ = certify(rows, N_SITES, persist="below_gap1")
    assert [len(below[j]) for j in range(4)] == [1, 0, 0, 0]
    assert [len(gap1[j]) for j in range(4)] == [1, 0, 1, 1]
    assert len(find_lobes(gap1, np.arange(4.0), "below_gap1")) == 1
    assert len(find_lobes(gap1, np.arange(4.0), "below")) == 2


def test_find_lobes_groups_overlapping_rungs():
    T = np.array([200.0, 225.0, 250.0])
    rungs = {0: [Gap(0.0, 0.25, 1, 0.1, 0), Gap(0.5, 0.75, 1, 0.1, 0)],
             1: [Gap(0.05, 0.25, 1, 0.1, 0), Gap(0.55, 0.7, 1, 0.1, 0)],
             2: [Gap(0.1, 0.6, 1, 0.1, 0)]}
    (lobe,) = find_lobes(rungs, T)
    assert (lobe["n_rungs"], lobe["T_bot"], lobe["T_top"]) == (5, 200.0, 250.0)
    assert (lobe["x_lo"], lobe["x_hi"]) == (0.0, 0.75)

    rungs[2] = [Gap(0.1, 0.2, 1, 0.1, 0)]
    lobes = find_lobes(rungs, T)
    assert [(L["x_lo"], L["x_hi"], L["T_top"]) for L in lobes] == [(0.0, 0.25, 250.0),
                                                                   (0.5, 0.75, 225.0)]


def test_ordered_compounds_chain_from_the_lowest_row():
    g = lambda a, b: Gap(a / N_SITES, b / N_SITES, 1, 0.1, 0)  # noqa: E731
    rungs = {j: [g(0, 4), g(4, 8), g(8, 16)] for j in range(3)}
    rungs[3] = [g(0, 5), g(5, 9), g(9, 16)]       # termini drift by one step
    rungs[4] = [g(0, 5), g(5, 16)]                # the terminus near x = 1/2 ends
    rungs[5] = [g(0, 8), g(8, 16)]                # a jump of three steps ends the chain
    assert ordered_compounds(rungs, N_SITES) == [(np.mean([4, 4, 4, 5, 5]) / N_SITES, 5),
                                                 (np.mean([8, 8, 8, 9]) / N_SITES, 4)]
    assert ordered_compounds(rungs, N_SITES, drift=0) == [(0.25, 3), (0.5, 3)]
    assert ordered_compounds(rungs, N_SITES, j_anchor=5) == [(0.5, 1)]
    rungs[0] = [g(0, 2), g(2, 16)]                # termini within two steps of an end do not count
    assert ordered_compounds(rungs, N_SITES) == []


def test_free_energy_is_exact_on_a_synthetic_sweep():
    x = np.arange(N_SITES + 1) / N_SITES
    T, dmu, k = 300.0, [-0.02, 0.0, 0.03], 20
    s = synthetic_sweep([double_well(x)], [T], dmu, k=k)
    f, f_se, inflation = fixed_composition_free_energy(
        s["A_samples"][:, 0], s["N_samples"][:, 0], s["dmu"], T, N_SITES)
    assert f == pytest.approx(double_well(x), abs=1e-12)
    assert f_se == pytest.approx(K_B * T / (N_SITES * np.sqrt(len(dmu) * k)))
    assert inflation == 1.0


def test_inflation_is_the_birge_ratio():
    x = np.arange(N_SITES + 1) / N_SITES
    T, k, delta = 300.0, 10, 0.5
    s = synthetic_sweep([double_well(x)], [T], [0.0, 0.0], k=k)
    A = s["A_samples"][:, 0].copy()
    A[0] += delta
    A[1] -= delta
    _, f_se, inflation = fixed_composition_free_energy(A, s["N_samples"][:, 0], s["dmu"], T,
                                                       N_SITES)
    assert inflation == pytest.approx(np.sqrt(2 * k * delta ** 2))
    assert f_se == pytest.approx(inflation * K_B * T / (N_SITES * np.sqrt(2 * k)))


def test_construct_phase_diagram_end_to_end(tmp_path):
    x = np.arange(N_SITES + 1) / N_SITES
    T = [300.0, 325.0, 350.0, 375.0]
    f_rows = [double_well(x)] * 3 + [0.01 * (x - 0.5) ** 2]
    sweep = synthetic_sweep(f_rows, T, [-0.01, 0.0, 0.01])
    pd = construct_phase_diagram(sweep)
    assert pd.n_sites == N_SITES and not pd.partial
    assert [(g.x_lo, g.x_hi) for j in range(4) for g in pd.rungs[j]] == [(0.25, 0.75)] * 3
    assert [(L["T_bot"], L["T_top"]) for L in pd.lobes] == [(300.0, 350.0)]
    assert pd.compounds() == []

    path = tmp_path / "rungs.npz"
    pd.save(path)
    back = PhaseDiagram.load(path)
    assert np.array_equal(back.rung_table(), pd.rung_table())
    assert back.lobes == pd.lobes and back.settings == pd.settings

    sweep["N_samples"][0, 1] = np.nan
    assert construct_phase_diagram(sweep).partial


def test_site_count_comes_from_meta():
    sweep = dict(N_samples=np.array([[[3.0, 5.0]]]), meta=dict(N=8))
    assert site_count(sweep) == 8
    sweep["meta"] = dict(N=4)
    with pytest.raises(ValueError):
        site_count(sweep)
