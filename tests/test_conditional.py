"""Conditional distributions and cross-field ordering constraints.

All tests are LLM-free: ``DistributionSampler`` (and ``generate_synthetic_data``
on all-distribution models) produce spec-compliant data deterministically from a
seed, so per-group marginals, threshold binning, and ordering are reproducible.
"""

import statistics
from typing import Annotated

import pytest
from pydantic import BaseModel

from gendantic import (
    Conditional,
    Constraints,
    Correlations,
    LLMDrivenModelAnalyser,
    Normal,
    Ordering,
    Range,
    Uniform,
    generate_synthetic_data_sync,
)
from gendantic.distributions import Categorical
from gendantic.sampler import DistributionSampler


def _specs(model: type[BaseModel]) -> dict:
    return LLMDrivenModelAnalyser.extract_distribution_specs_with_types(model)


# --------------------------------------------------------------------------
# Conditional distributions - categorical discriminator
# --------------------------------------------------------------------------


class Employee(BaseModel):
    department: Annotated[str, Categorical({"Eng": 0.5, "Sales": 0.3, "HR": 0.2})]
    salary: Annotated[
        float,
        Conditional(
            on="department",
            cases={"Eng": Normal(90000, 3000), "Sales": Normal(70000, 3000)},
            default=Normal(50000, 3000),
        ),
    ]


def test_conditional_field_is_extracted() -> None:
    specs = LLMDrivenModelAnalyser.extract_distribution_specs(Employee)
    assert isinstance(specs["salary"], Conditional)
    assert specs["salary"].on == "department"


def test_categorical_conditional_samples_per_group() -> None:
    records = DistributionSampler(seed=1).sample_fields(_specs(Employee), 4000)
    groups: dict[str, list[float]] = {}
    for r in records:
        groups.setdefault(r["department"], []).append(r["salary"])

    assert statistics.mean(groups["Eng"]) == pytest.approx(90000, abs=800)
    assert statistics.mean(groups["Sales"]) == pytest.approx(70000, abs=800)
    # HR has no case and falls through to the default.
    assert statistics.mean(groups["HR"]) == pytest.approx(50000, abs=800)


# --------------------------------------------------------------------------
# Conditional distributions - numeric thresholds
# --------------------------------------------------------------------------


class Banded(BaseModel):
    age: Annotated[int, Uniform(20, 60)]
    bonus: Annotated[
        float,
        Conditional(
            on="age",
            cases={
                Range(max=30): Normal(3000, 100),
                Range(30, 50): Normal(6000, 100),
                Range(min=50): Normal(9000, 100),
            },
            default=Normal(0, 1),
        ),
    ]


def test_numeric_thresholds_bin_on_the_converted_value() -> None:
    records = DistributionSampler(seed=2).sample_fields(_specs(Banded), 5000)
    buckets: dict[str, list[float]] = {"low": [], "mid": [], "high": []}
    for r in records:
        key = "low" if r["age"] < 30 else "mid" if r["age"] < 50 else "high"
        buckets[key].append(r["bonus"])

    # Bins line up exactly with the int age the record exposes (no default leak).
    assert statistics.mean(buckets["low"]) == pytest.approx(3000, abs=40)
    assert statistics.mean(buckets["mid"]) == pytest.approx(6000, abs=40)
    assert statistics.mean(buckets["high"]) == pytest.approx(9000, abs=40)


# --------------------------------------------------------------------------
# Dependency resolution
# --------------------------------------------------------------------------


def test_conditional_can_depend_on_another_conditional() -> None:
    class Chain(BaseModel):
        tier: Annotated[str, Categorical({"gold": 0.5, "silver": 0.5})]
        rate: Annotated[
            float,
            Conditional(
                on="tier",
                cases={"gold": Normal(10, 0.5), "silver": Normal(20, 0.5)},
                default=Normal(0, 1),
            ),
        ]
        # depends on `rate`, which is itself conditional
        fee: Annotated[
            float,
            Conditional(
                on="rate",
                cases={Range(max=15): Normal(100, 1), Range(min=15): Normal(200, 1)},
                default=Normal(0, 1),
            ),
        ]

    records = DistributionSampler(seed=5).sample_fields(_specs(Chain), 2000)
    for r in records:
        # gold -> rate~10 -> fee~100; silver -> rate~20 -> fee~200
        expected_fee = 100 if r["tier"] == "gold" else 200
        assert abs(r["fee"] - expected_fee) < 10


def test_self_dependency_raises() -> None:
    sampler = DistributionSampler(seed=0)
    specs = {
        "x": (
            Conditional(on="x", cases={"a": Normal(1, 1)}, default=Normal(0, 1)),
            float,
            {"ge": None, "le": None, "gt": None, "lt": None},
        )
    }
    with pytest.raises(ValueError, match="cannot depend on itself"):
        sampler.sample_fields(specs, 10)


def test_unknown_discriminator_raises() -> None:
    class Bad(BaseModel):
        salary: Annotated[
            float,
            Conditional(on="ghost", cases={"a": Normal(1, 1)}, default=Normal(0, 1)),
        ]

    with pytest.raises(ValueError, match="not a distribution-sampled field"):
        DistributionSampler(seed=0).sample_fields(_specs(Bad), 10)


def test_circular_dependency_raises() -> None:
    no_c = {"ge": None, "le": None, "gt": None, "lt": None}
    specs = {
        "a": (
            Conditional(on="b", cases={Range(min=0): Normal(1, 1)}, default=Normal(0, 1)),
            float,
            no_c,
        ),
        "b": (
            Conditional(on="a", cases={Range(min=0): Normal(1, 1)}, default=Normal(0, 1)),
            float,
            no_c,
        ),
    }
    with pytest.raises(ValueError, match="Circular dependency"):
        DistributionSampler(seed=0).sample_fields(specs, 10)


def test_correlation_referencing_conditional_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class Mixed(BaseModel):
        age: Annotated[float, Uniform(20, 60)]
        salary: Annotated[
            float,
            Conditional(
                on="age",
                cases={Range(min=40): Normal(90000, 5000)},
                default=Normal(50000, 5000),
            ),
        ]
        __correlations__ = Correlations(("age", "salary", 0.6))

    with caplog.at_level("WARNING", logger="gendantic"):
        DistributionSampler(seed=0).sample_fields(
            _specs(Mixed), 100, correlations=Mixed.__correlations__
        )
    assert any("conditional field" in r.message for r in caplog.records)


# --------------------------------------------------------------------------
# Ordering constraints
# --------------------------------------------------------------------------


class Booking(BaseModel):
    start: Annotated[float, Uniform(0, 100)]
    end: Annotated[float, Uniform(0, 100)]
    __constraints__ = Constraints(Ordering("start", "end"))


def test_ordering_holds_for_every_record() -> None:
    records = DistributionSampler(seed=3).sample_fields(
        _specs(Booking), 5000, constraints=Booking.__constraints__
    )
    assert all(r["start"] <= r["end"] for r in records)


def test_ordering_three_fields() -> None:
    class Milestones(BaseModel):
        a: Annotated[float, Uniform(0, 100)]
        b: Annotated[float, Uniform(0, 100)]
        c: Annotated[float, Uniform(0, 100)]
        __constraints__ = Constraints(Ordering("a", "b", "c"))

    records = DistributionSampler(seed=4).sample_fields(
        _specs(Milestones), 3000, constraints=Milestones.__constraints__
    )
    assert all(r["a"] <= r["b"] <= r["c"] for r in records)


def test_ordering_missing_field_raises() -> None:
    class OnlyOne(BaseModel):
        start: Annotated[float, Uniform(0, 100)]

    with pytest.raises(ValueError, match="not sampled"):
        DistributionSampler(seed=0).sample_fields(
            _specs(OnlyOne), 10, constraints=Constraints(Ordering("start", "end"))
        )


# --------------------------------------------------------------------------
# Determinism, code-gen round-trip, and end-to-end integration
# --------------------------------------------------------------------------


def test_conditional_sampling_is_deterministic() -> None:
    a = DistributionSampler(seed=99).sample_fields(_specs(Employee), 500)
    b = DistributionSampler(seed=99).sample_fields(_specs(Employee), 500)
    assert a == b


def test_reprs_round_trip() -> None:
    namespace = {
        "Conditional": Conditional,
        "Range": Range,
        "Normal": Normal,
        "Ordering": Ordering,
        "Constraints": Constraints,
    }
    cond = Conditional(
        on="age",
        cases={Range(30, 50): Normal(mean=6000, std=1000)},
        default=Normal(mean=5000, std=1000),
    )
    assert eval(repr(cond), namespace) == cond  # noqa: S307 - round-trips our own code-gen

    cons = Constraints(Ordering("start", "end"))
    rebuilt = eval(cons.to_code(), namespace)  # noqa: S307 - round-trips our own code-gen
    assert list(rebuilt) == list(cons)


def test_end_to_end_generation_with_conditional_and_ordering() -> None:
    # All fields are distribution-driven, so no LLM call is made.
    class Contract(BaseModel):
        tier: Annotated[str, Categorical({"pro": 0.5, "basic": 0.5})]
        price: Annotated[
            float,
            Conditional(
                on="tier",
                cases={"pro": Normal(1000, 10), "basic": Normal(100, 5)},
                default=Normal(0, 1),
            ),
        ]
        start: Annotated[float, Uniform(0, 50)]
        end: Annotated[float, Uniform(50, 100)]
        __constraints__ = Constraints(Ordering("start", "end"))

    records = generate_synthetic_data_sync(Contract, count=300, seed=7)
    assert len(records) == 300
    for r in records:
        assert r.start <= r.end
        expected = 1000 if r.tier == "pro" else 100
        assert abs(r.price - expected) < 60
