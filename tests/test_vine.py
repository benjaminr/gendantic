"""End-to-end vine sampling through ``DistributionSampler``.

These drive the full ``_sample_vine`` path: per-pair dependence must be
recovered on the sampled data, non-adjacent fields must be only conditionally
dependent, marginals must stay uniform-on-the-copula-scale (i.e. the requested
distribution is preserved), sampling must be deterministic under a fixed seed,
and structurally invalid specs must raise.
"""

import numpy as np
import pytest
from scipy import stats

from gendantic.distributions import Correlations, Normal
from gendantic.sampler import DistributionSampler

UNBOUNDED: dict[str, float | None] = {"ge": None, "le": None, "gt": None, "lt": None}


def _field():
    return (Normal(mean=0.0, std=1.0), float, UNBOUNDED)


def _specs(*names: str) -> dict:
    return {n: _field() for n in names}


def _col(records: list[dict], name: str) -> np.ndarray:
    return np.array([r[name] for r in records])


def _kendall(records: list[dict], a: str, b: str) -> float:
    return float(stats.kendalltau(_col(records, a), _col(records, b)).statistic)


@pytest.mark.parametrize("family", ["gaussian", "clayton", "gumbel", "frank"])
def test_per_pair_kendall_tau_recovered(family) -> None:
    # Archimedean families take corr as target tau directly; Gaussian's rho=0.6
    # corresponds to tau = 2/pi * asin(0.6) ~= 0.41.
    tau_target = 2.0 / np.pi * np.arcsin(0.6) if family == "gaussian" else 0.6
    sampler = DistributionSampler(seed=7)
    records = sampler.sample_fields(
        _specs("a", "b"),
        8000,
        correlations=Correlations(("a", "b", 0.6, family)),
    )
    assert _kendall(records, "a", "b") == pytest.approx(tau_target, abs=0.04)


def test_chain_conditional_independence() -> None:
    # a - b - c chain: a and c share no edge, so their dependence must be weaker
    # than the direct a-b and b-c edges (conditional independence given b).
    sampler = DistributionSampler(seed=8)
    records = sampler.sample_fields(
        _specs("a", "b", "c"),
        8000,
        correlations=Correlations(
            ("a", "b", 0.7, "gaussian"), ("b", "c", 0.7, "gaussian")
        ),
    )
    tau_ab = _kendall(records, "a", "b")
    tau_bc = _kendall(records, "b", "c")
    tau_ac = _kendall(records, "a", "c")
    assert tau_ab > 0.3 and tau_bc > 0.3
    assert tau_ac < tau_ab and tau_ac < tau_bc  # indirect link is weaker


def test_marginals_preserved_uniform_on_copula_scale() -> None:
    # After the probability-integral transform, each column should be standard
    # normal (the requested marginal): a KS test must not reject.
    sampler = DistributionSampler(seed=9)
    records = sampler.sample_fields(
        _specs("a", "b"),
        5000,
        correlations=Correlations(("a", "b", 0.5, "clayton")),
    )
    for name in ("a", "b"):
        ks = stats.kstest(_col(records, name), "norm")
        assert ks.pvalue > 0.01


def test_sampling_is_deterministic_under_seed() -> None:
    corr = Correlations(("a", "b", 0.6, "gumbel"), ("b", "c", 0.5, "frank"))
    r1 = DistributionSampler(seed=11).sample_fields(_specs("a", "b", "c"), 200, correlations=corr)
    r2 = DistributionSampler(seed=11).sample_fields(_specs("a", "b", "c"), 200, correlations=corr)
    assert r1 == r2


def test_unpaired_capable_field_is_still_sampled() -> None:
    # 'c' has no correlation edge but is a correlatable field; it must appear.
    sampler = DistributionSampler(seed=12)
    records = sampler.sample_fields(
        _specs("a", "b", "c"),
        50,
        correlations=Correlations(("a", "b", 0.5, "gaussian")),
    )
    assert all(set(r) == {"a", "b", "c"} for r in records)


def test_cycle_specification_raises() -> None:
    sampler = DistributionSampler(seed=13)
    with pytest.raises(ValueError, match="cycle"):
        sampler.sample_fields(
            _specs("a", "b", "c"),
            100,
            correlations=Correlations(
                ("a", "b", 0.5, "clayton"),
                ("b", "c", 0.5, "clayton"),
                ("a", "c", 0.5, "clayton"),
            ),
        )


def test_duplicate_pair_specification_raises() -> None:
    sampler = DistributionSampler(seed=14)
    with pytest.raises(ValueError, match="more than once"):
        sampler.sample_fields(
            _specs("a", "b"),
            100,
            correlations=Correlations(
                ("a", "b", 0.5, "gaussian"), ("b", "a", 0.6, "frank")
            ),
        )


def test_tail_asymmetry_direction_clayton_vs_gumbel() -> None:
    # Clayton clusters in the lower tail, Gumbel in the upper tail. Compare the
    # two on the same seed and field layout.
    sampler = DistributionSampler(seed=15)
    records = sampler.sample_fields(
        _specs("a", "b", "c"),
        8000,
        correlations=Correlations(
            ("a", "b", 0.6, "clayton"), ("a", "c", 0.6, "gumbel")
        ),
    )
    a, b, c = _col(records, "a"), _col(records, "b"), _col(records, "c")
    lo_a, hi_a = np.quantile(a, [0.05, 0.95])
    lo_b, hi_b = np.quantile(b, [0.05, 0.95])
    lo_c, hi_c = np.quantile(c, [0.05, 0.95])
    assert np.mean((a < lo_a) & (b < lo_b)) > np.mean((a > hi_a) & (b > hi_b))
    assert np.mean((a > hi_a) & (c > hi_c)) > np.mean((a < lo_a) & (c < lo_c))
