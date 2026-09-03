"""Edge-case routing in correlated copula sampling.

Covers the branches in ``DistributionSampler._columns_correlated`` /
``_sample_archimedean_copula`` that decide *how* fields get sampled: excluding
distributions that cannot be correlated, falling back to a Gaussian base for
mixed copula specs, treating zero correlation as independence, and rejecting
negative dependence for positive-only Archimedean copulas.
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


def test_mixed_copula_types_fall_back_to_gaussian_base() -> None:
    # Two different copula types among capable fields -> Gaussian base is used.
    specs = {"a": _normal_field(), "b": _normal_field(), "c": _normal_field()}
    sampler = DistributionSampler(seed=2)
    records = sampler.sample_fields(
        specs,
        500,
        correlations=Correlations(("a", "b", 0.7, "clayton"), ("a", "c", 0.6, "gumbel")),
    )
    # Correlation is still induced (Gaussian base honours the requested value).
    assert _corr(records, "a", "b") > 0.5


def test_zero_correlation_samples_independently() -> None:
    specs = {"x": _normal_field(), "y": _normal_field()}
    sampler = DistributionSampler(seed=3)
    records = sampler.sample_fields(
        specs, 10, correlations=Correlations(("x", "y", 0.0))
    )
    assert len(records) == 10
    assert sorted(records[0]) == ["x", "y"]


@pytest.mark.parametrize("copula", ["clayton", "gumbel"])
def test_positive_only_copula_falls_back_to_gaussian_for_negative_corr(
    copula, caplog
) -> None:
    specs = {"x": _normal_field(), "y": _normal_field()}
    sampler = DistributionSampler(seed=4)
    with caplog.at_level(logging.WARNING, logger="gendantic"):
        records = sampler.sample_fields(
            specs, 300, correlations=Correlations(("x", "y", -0.6, copula))
        )

    # Achieving a negative correlation proves the Gaussian fallback ran:
    # Clayton/Gumbel are positive-only and could never produce it.
    assert _corr(records, "x", "y") < -0.3
    assert "non-positive" in caplog.text
