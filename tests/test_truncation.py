"""Bounded fields are truncated at the source, not clamped afterwards.

Clamping out-of-range draws to the boundary piles a spike of mass on the bound
and makes the marginal disagree with its declared distribution. These tests
assert the sampler instead draws from the conditional distribution on the
allowed interval (proper inverse-CDF truncation), that bounds are respected,
that the copula/conditional paths truncate too, and that ``fidelity_report``
compares against the truncated distribution so bounded fields still pass.
"""

from typing import Annotated

import numpy as np
import pytest
from pydantic import BaseModel, Field
from scipy import stats

from gendantic import (
    Correlations,
    DistributionSampler,
    LLMDrivenModelAnalyser,
    Normal,
    Poisson,
    fidelity_report,
)
from gendantic.distributions import TruncatedSpec, truncate


def _bounds(ge=None, le=None, gt=None, lt=None) -> dict[str, float | None]:
    return {"ge": ge, "le": le, "gt": gt, "lt": lt}


def _col(records: list[dict], name: str) -> np.ndarray:
    return np.array([r[name] for r in records])


# --- TruncatedSpec / truncate unit behaviour ------------------------------


def test_truncate_is_identity_without_bounds() -> None:
    spec = Normal(mean=0.0, std=1.0)
    assert truncate(spec, float, _bounds()) is spec


def test_truncate_is_identity_for_non_numeric_target() -> None:
    spec = Normal(mean=0.0, std=1.0)
    assert truncate(spec, str, _bounds(ge=0.0)) is spec


def test_truncate_wraps_when_bounded() -> None:
    wrapped = truncate(Normal(mean=0.0, std=1.0), float, _bounds(ge=0.0))
    assert isinstance(wrapped, TruncatedSpec)


def test_truncated_quantile_stays_within_window() -> None:
    wrapped = truncate(Normal(mean=0.0, std=1.0), float, _bounds(ge=-1.0, le=2.0))
    u = np.linspace(1e-6, 1 - 1e-6, 500)
    x = wrapped.quantile(u)
    assert x.min() >= -1.0 - 1e-9
    assert x.max() <= 2.0 + 1e-9


def test_truncated_cdf_is_zero_below_and_one_above() -> None:
    wrapped = truncate(Normal(mean=0.0, std=1.0), float, _bounds(ge=-1.0, le=2.0))
    assert float(wrapped.cdf(np.array([-5.0]))[0]) == pytest.approx(0.0)
    assert float(wrapped.cdf(np.array([5.0]))[0]) == pytest.approx(1.0)
    # Monotone in between.
    grid = np.linspace(-1.0, 2.0, 50)
    c = wrapped.cdf(grid)
    assert np.all(np.diff(c) >= -1e-12)


# --- Sampler: no boundary spike, bounds respected -------------------------


def test_no_mass_piled_on_the_boundary() -> None:
    # Normal(0,1) truncated at ge=0 is a half-normal: about 2*phi(0)*0.01 ~= 0.8%
    # of mass falls in [0, 0.01). Clamping would instead pile ~50% exactly on 0.
    specs = {"x": (Normal(mean=0.0, std=1.0), float, _bounds(ge=0.0))}
    records = DistributionSampler(seed=1).sample_fields(specs, 20000)
    x = _col(records, "x")
    assert x.min() >= 0.0
    assert np.mean(x == 0.0) < 0.001  # no spike at the bound
    assert np.mean(x < 0.01) < 0.03  # matches truncated density, not a clamp


def test_truncated_sample_matches_truncated_distribution() -> None:
    spec = Normal(mean=0.0, std=1.0)
    specs = {"x": (spec, float, _bounds(ge=0.0, le=2.0))}
    records = DistributionSampler(seed=2).sample_fields(specs, 5000)
    x = _col(records, "x")
    wrapped = truncate(spec, float, _bounds(ge=0.0, le=2.0))
    ks = stats.kstest(x, wrapped.cdf)
    assert ks.pvalue > 0.01
    # And it does NOT look like the full (untruncated) normal.
    assert stats.kstest(x, spec.cdf).pvalue < 1e-6


@pytest.mark.parametrize(
    "bounds",
    [_bounds(ge=1.0), _bounds(le=3.0), _bounds(gt=0.0), _bounds(lt=5.0),
     _bounds(ge=-2.0, le=2.0)],
)
def test_bounds_are_respected(bounds) -> None:
    specs = {"x": (Normal(mean=0.0, std=2.0), float, bounds)}
    records = DistributionSampler(seed=3).sample_fields(specs, 4000)
    x = _col(records, "x")
    if bounds["ge"] is not None:
        assert x.min() >= bounds["ge"]
    if bounds["gt"] is not None:
        assert x.min() > bounds["gt"]
    if bounds["le"] is not None:
        assert x.max() <= bounds["le"]
    if bounds["lt"] is not None:
        assert x.max() < bounds["lt"]


def test_integer_bounds_snap_to_support() -> None:
    # int target with ge=0, le=10: values land in {0..10}, and 0 is not a spike.
    specs = {"n": (Normal(mean=5.0, std=4.0), int, _bounds(ge=0.0, le=10.0))}
    records = DistributionSampler(seed=4).sample_fields(specs, 5000)
    n = _col(records, "n")
    assert set(np.unique(n)).issubset(set(range(0, 11)))
    assert n.min() == 0 and n.max() == 10


def test_poisson_lower_bound_excludes_below() -> None:
    # Poisson(lam=2) truncated at ge=2: no counts below 2.
    specs = {"k": (Poisson(lam=2.0), int, _bounds(ge=2.0))}
    records = DistributionSampler(seed=5).sample_fields(specs, 4000)
    assert _col(records, "k").min() >= 2


# --- Copula path truncates while preserving dependence --------------------


def test_correlated_bounded_fields_stay_bounded_and_correlated() -> None:
    specs = {
        "a": (Normal(mean=0.0, std=1.0), float, _bounds(ge=0.0)),
        "b": (Normal(mean=0.0, std=1.0), float, _bounds(ge=0.0)),
    }
    records = DistributionSampler(seed=6).sample_fields(
        specs, 6000, correlations=Correlations(("a", "b", 0.6))
    )
    a, b = _col(records, "a"), _col(records, "b")
    assert a.min() >= 0.0 and b.min() >= 0.0
    assert float(np.corrcoef(a, b)[0, 1]) > 0.3  # dependence survives truncation


# --- End-to-end fidelity is truncation-aware ------------------------------


class BoundedModel(BaseModel):
    """A model with a hard lower bound on an otherwise-unbounded Normal."""

    income: Annotated[float, Normal(mean=0.0, std=1.0)] = Field(ge=0.0)


def _generate_with_types(model: type[BaseModel], count: int, seed: int) -> list[dict]:
    specs = LLMDrivenModelAnalyser.extract_distribution_specs_with_types(model)
    correlations = getattr(model, "__correlations__", None)
    return DistributionSampler(seed=seed).sample_fields(specs, count, correlations)


def test_fidelity_passes_for_bounded_field() -> None:
    records = _generate_with_types(BoundedModel, 3000, seed=7)
    report = fidelity_report(records, BoundedModel)
    assert report.passed, report.summary()
    assert min(r["income"] for r in records) >= 0.0
