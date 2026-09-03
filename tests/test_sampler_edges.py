"""Edge-case routing in correlated copula sampling.

Covers the branches in ``DistributionSampler._columns_correlated`` that decide
*how* fields get sampled: excluding distributions that cannot be correlated,
honouring mixed copula families per-pair through the vine, treating zero
correlation as independence, and rejecting negative dependence for
positive-only Archimedean copulas.
"""

import logging

import numpy as np
import pytest

from gendantic.distributions import Categorical, Correlations, Normal
from gendantic.sampler import DistributionSampler

UNBOUNDED: dict[str, float | None] = {"ge": None, "le": None, "gt": None, "lt": None}


def _normal_field(mean: float = 0.0, std: float = 1.0):
    return (Normal(mean=mean, std=std), float, UNBOUNDED)


def _corr(records: list[dict], a: str, b: str) -> float:
    return float(np.corrcoef([r[a] for r in records], [r[b] for r in records])[0, 1])


def test_categorical_field_is_excluded_from_correlation(caplog) -> None:
    specs = {
        "salary": _normal_field(50000, 10000),
        "dept": (Categorical(weights={"Eng": 0.5, "Sales": 0.5}), str, UNBOUNDED),
    }
    sampler = DistributionSampler(seed=1)
    with caplog.at_level(logging.WARNING, logger="gendantic"):
        records = sampler.sample_fields(
            specs, 30, correlations=Correlations(("salary", "dept", 0.5))
        )

    # The categorical is still sampled (independently), and a warning explains why.
    assert all("dept" in r and "salary" in r for r in records)
    assert {r["dept"] for r in records} <= {"Eng", "Sales"}
    assert "does not support correlation" in caplog.text


def test_mixed_copula_types_are_honoured_per_pair() -> None:
    # Mixed families no longer collapse to a Gaussian base: each pair keeps its
    # own copula via the vine. a-b is Clayton (lower tail), a-c is Gumbel (upper
    # tail); both correlations are induced and the tail asymmetries are opposite.
    specs = {"a": _normal_field(), "b": _normal_field(), "c": _normal_field()}
    sampler = DistributionSampler(seed=2)
    records = sampler.sample_fields(
        specs,
        6000,
        correlations=Correlations(
            ("a", "b", 0.6, "clayton"), ("a", "c", 0.6, "gumbel")
        ),
    )
    assert _corr(records, "a", "b") > 0.4
    assert _corr(records, "a", "c") > 0.4

    a = np.array([r["a"] for r in records])
    lo_a, hi_a = np.quantile(a, [0.05, 0.95])
    # Clayton pair (a, b): more joint-low co-movement than joint-high.
    b = np.array([r["b"] for r in records])
    lo_b, hi_b = np.quantile(b, [0.05, 0.95])
    clayton_low = np.mean((a < lo_a) & (b < lo_b))
    clayton_high = np.mean((a > hi_a) & (b > hi_b))
    assert clayton_low > clayton_high
    # Gumbel pair (a, c): more joint-high co-movement than joint-low.
    c = np.array([r["c"] for r in records])
    lo_c, hi_c = np.quantile(c, [0.05, 0.95])
    gumbel_low = np.mean((a < lo_a) & (c < lo_c))
    gumbel_high = np.mean((a > hi_a) & (c > hi_c))
    assert gumbel_high > gumbel_low


def test_zero_correlation_samples_independently() -> None:
    specs = {"x": _normal_field(), "y": _normal_field()}
    sampler = DistributionSampler(seed=3)
    records = sampler.sample_fields(
        specs, 10, correlations=Correlations(("x", "y", 0.0))
    )
    assert len(records) == 10
    assert sorted(records[0]) == ["x", "y"]


@pytest.mark.parametrize("copula", ["clayton", "gumbel"])
def test_positive_only_copula_rejects_negative_corr(copula) -> None:
    # Clayton/Gumbel model positive dependence only. Rather than silently
    # substituting a Gaussian, a negative correlation now raises.
    specs = {"x": _normal_field(), "y": _normal_field()}
    sampler = DistributionSampler(seed=4)
    with pytest.raises(ValueError, match="positive dependence only"):
        sampler.sample_fields(
            specs, 300, correlations=Correlations(("x", "y", -0.6, copula))
        )
