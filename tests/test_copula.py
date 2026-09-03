"""Unit tests for the pure pair-copula functions and vine construction.

These exercise ``gendantic.copula`` in isolation (no sampler, no RNG state):
the inverse h-functions must round-trip against the h-functions, the closed
forms must agree with a numerical inverse, and the forest builder must reject
structurally invalid specifications.
"""

import numpy as np
import pytest
from scipy import optimize, stats

from gendantic import copula as cp
from gendantic.distributions import CopulaType as C

FAMILIES_CORR = [
    (C.GAUSSIAN, 0.6),
    (C.GAUSSIAN, -0.5),
    (C.STUDENT_T, 0.6),
    (C.STUDENT_T, -0.4),
    (C.CLAYTON, 0.5),
    (C.GUMBEL, 0.5),
    (C.FRANK, 0.5),
    (C.FRANK, -0.4),
]


@pytest.mark.parametrize("family,corr", FAMILIES_CORR)
def test_hinv_round_trips_against_h(family, corr) -> None:
    rng = np.random.default_rng(0)
    u = rng.uniform(0.01, 0.99, 3000)
    w = rng.uniform(0.01, 0.99, 3000)
    param = cp.edge_param(family, corr)
    u2 = cp.hinv(family, w, u, param)
    recovered = cp.h(family, u2, u, param)
    assert np.max(np.abs(recovered - w)) < 1e-6


def test_student_t_uses_df_plus_one_inner_quantile() -> None:
    # Regression guard: using df (not df+1) for the inner quantile breaks the
    # round-trip, so the correct implementation must beat that broken variant.
    rng = np.random.default_rng(1)
    u = rng.uniform(0.02, 0.98, 2000)
    w = rng.uniform(0.02, 0.98, 2000)
    df = cp.STUDENT_T_DF

    def wrong(w, u, rho):
        x = stats.t.ppf(u, df)
        y = stats.t.ppf(w, df)  # wrong: should be df + 1
        scale = np.sqrt((df + x**2) * (1 - rho**2) / (df + 1))
        return np.clip(stats.t.cdf(rho * x + y * scale, df), 1e-10, 1 - 1e-10)

    correct = cp.hinv(C.STUDENT_T, w, u, 0.6)
    wrong_err = np.max(np.abs(cp.h(C.STUDENT_T, wrong(w, u, 0.6), u, 0.6) - w))
    correct_err = np.max(np.abs(cp.h(C.STUDENT_T, correct, u, 0.6) - w))
    assert correct_err < 1e-6 < wrong_err


@pytest.mark.parametrize("family,corr", FAMILIES_CORR)
def test_closed_form_hinv_matches_numeric_inverse(family, corr) -> None:
    param = cp.edge_param(family, corr)
    u = np.array([0.2, 0.5, 0.8])
    w = np.array([0.3, 0.5, 0.7])
    closed = cp.hinv(family, w, u, param)
    for i in range(len(u)):
        numeric = optimize.brentq(
            lambda v, i=i: float(cp.h(family, np.array([v]), u[i : i + 1], param)[0])
            - w[i],
            1e-9,
            1 - 1e-9,
        )
        assert closed[i] == pytest.approx(numeric, abs=1e-5)


@pytest.mark.parametrize("family", [C.GAUSSIAN, C.CLAYTON, C.GUMBEL, C.FRANK])
def test_degenerate_param_is_independence(family) -> None:
    # Zero correlation -> independence: hinv returns w unchanged. (Student-t is
    # excluded: a bivariate t with zero correlation is uncorrelated but not
    # independent -- it keeps tail dependence -- so param=0 is not independence.)
    w = np.array([0.1, 0.4, 0.9])
    u = np.array([0.5, 0.5, 0.5])
    param = cp.edge_param(family, 0.0)
    assert np.allclose(cp.hinv(family, w, u, param), w)


@pytest.mark.parametrize("family", [C.CLAYTON, C.GUMBEL])
def test_archimedean_edge_param_at_perfect_correlation_is_finite(family) -> None:
    # corr == 1.0 sends theta -> inf for Clayton/Gumbel (1 - corr == 0); the
    # param must be clamped to a large-but-finite value instead of dividing by
    # zero (which used to raise ZeroDivisionError / produce inf).
    param = cp.edge_param(family, 1.0)
    assert np.isfinite(param)
    assert param > 0.0


@pytest.mark.parametrize("tau", [0.6, 0.3, -0.3, -0.6])
def test_frank_theta_from_tau_round_trips(tau) -> None:
    theta = cp.edge_param(C.FRANK, tau)
    # Recover Kendall's tau from theta via the Debye-1 relation.
    u = np.random.default_rng(2).uniform(size=20000)
    w = np.random.default_rng(3).uniform(size=20000)
    v = cp.hinv(C.FRANK, w, u, theta)
    assert stats.kendalltau(u, v).statistic == pytest.approx(tau, abs=0.03)


def test_build_vine_rejects_duplicate_pair() -> None:
    with pytest.raises(ValueError, match="more than once"):
        cp.build_vine([("a", "b", 0.5, "gaussian"), ("b", "a", 0.6, "gumbel")])


def test_build_vine_rejects_cycle() -> None:
    with pytest.raises(ValueError, match="cycle"):
        cp.build_vine(
            [
                ("a", "b", 0.5, "gaussian"),
                ("b", "c", 0.5, "gaussian"),
                ("a", "c", 0.5, "gaussian"),
            ]
        )


@pytest.mark.parametrize("family", ["clayton", "gumbel"])
def test_build_vine_rejects_negative_archimedean(family) -> None:
    with pytest.raises(ValueError, match="positive dependence only"):
        cp.build_vine([("a", "b", -0.4, family)])


def test_build_vine_produces_component_trees_parent_before_child() -> None:
    trees = cp.build_vine(
        [
            ("b", "a", 0.5, "gaussian"),
            ("a", "c", 0.6, "gumbel"),
            ("x", "y", 0.4, "frank"),
        ]
    )
    assert len(trees) == 2  # {a, b, c} and {x, y}
    for tree in trees:
        drawn = {tree.root}
        for edge in tree.edges:
            assert edge.parent in drawn  # parent already available when child drawn
            drawn.add(edge.child)
