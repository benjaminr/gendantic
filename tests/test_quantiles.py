"""Quantile (inverse-CDF) correctness for every distribution.

``DistributionSpec.quantile`` is the path used for copula-based correlated
sampling: uniform [0, 1] draws are mapped through it to produce correlated
marginals. The ``.sample`` methods are exercised elsewhere, but the quantile
functions were previously only hit indirectly (and only for Normal/Uniform),
so these tests pin the inverse-CDF of each distribution directly.
"""

import numpy as np
import pytest

from gendantic.distributions import (
    Beta,
    Binomial,
    Categorical,
    Exponential,
    LogNormal,
    Normal,
    Poisson,
    Uniform,
)

MEDIAN = np.array([0.5])


def test_normal_quantile_median_is_mean() -> None:
    assert Normal(mean=10.0, std=2.0).quantile(MEDIAN)[0] == pytest.approx(10.0)


def test_uniform_quantile_maps_linearly() -> None:
    dist = Uniform(min=0.0, max=10.0)
    u = np.array([0.0, 0.25, 0.5, 1.0])
    assert dist.quantile(u).tolist() == pytest.approx([0.0, 2.5, 5.0, 10.0])


def test_lognormal_quantile_median_is_exp_mean() -> None:
    # Median of a log-normal is exp(mean).
    assert LogNormal(mean=2.0, sigma=0.5).quantile(MEDIAN)[0] == pytest.approx(
        np.exp(2.0)
    )


def test_exponential_quantile_median() -> None:
    # Median of Exponential(scale) is scale * ln(2).
    scale = 4.0
    assert Exponential(scale=scale).quantile(MEDIAN)[0] == pytest.approx(
        scale * np.log(2)
    )


def test_poisson_quantile_median() -> None:
    assert Poisson(lam=10.0).quantile(MEDIAN)[0] == pytest.approx(10.0)


def test_beta_symmetric_quantile_median() -> None:
    assert Beta(alpha=2.0, beta=2.0).quantile(MEDIAN)[0] == pytest.approx(0.5)


def test_binomial_quantile_median() -> None:
    assert Binomial(n=10, p=0.5).quantile(MEDIAN)[0] == pytest.approx(5.0)


def test_categorical_quantile_uses_inverse_cdf() -> None:
    cat = Categorical(weights={"a": 0.5, "b": 0.5})
    # u below the first cumulative mass -> "a"; above it -> "b".
    result = cat.quantile(np.array([0.1, 0.25, 0.75, 0.99]))
    assert result.tolist() == ["a", "a", "b", "b"]


def test_categorical_quantile_clips_at_upper_edge() -> None:
    # u == 1.0 must not index past the last category.
    cat = Categorical(weights={"x": 0.3, "y": 0.7})
    assert cat.quantile(np.array([1.0]))[0] == "y"


@pytest.mark.parametrize(
    "dist",
    [
        Normal(mean=0.0, std=1.0),
        LogNormal(mean=0.0, sigma=1.0),
        Exponential(scale=2.0),
        Beta(alpha=2.0, beta=5.0),
    ],
)
def test_continuous_quantiles_are_monotonic(dist) -> None:
    u = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    values = dist.quantile(u)
    assert np.all(np.diff(values) > 0)
