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
    fidelity_report,
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
            Conditional(
                on="b", cases={Range(min=0): Normal(1, 1)}, default=Normal(0, 1)
            ),
            float,
            no_c,
        ),
        "b": (
            Conditional(
                on="a", cases={Range(min=0): Normal(1, 1)}, default=Normal(0, 1)
            ),
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
# Ordering constraints - resample method (marginal-preserving)
# --------------------------------------------------------------------------


class Career(BaseModel):
    # Disjoint ranges, so the order always holds and no rejection is needed;
    # each field keeps its own marginal exactly.
    birth: Annotated[float, Uniform(0, 30)]
    hire: Annotated[float, Uniform(30, 60)]
    termination: Annotated[float, Uniform(60, 100)]
    __constraints__ = Constraints(
        Ordering("birth", "hire", "termination", method="resample")
    )


def test_resample_ordering_holds_and_preserves_marginals() -> None:
    records = DistributionSampler(seed=1).sample_fields(
        _specs(Career), 5000, constraints=Career.__constraints__
    )
    assert all(r["birth"] <= r["hire"] <= r["termination"] for r in records)

    # Each field's mean stays at its own distribution's midpoint - the sort
    # method would instead pull them to order statistics.
    assert statistics.mean(r["birth"] for r in records) == pytest.approx(15, abs=1)
    assert statistics.mean(r["hire"] for r in records) == pytest.approx(45, abs=1)
    assert statistics.mean(r["termination"] for r in records) == pytest.approx(
        80, abs=1
    )


def test_resample_ordering_passes_fidelity_when_sort_would_not() -> None:
    # With disjoint marginals, resample preserves them, so each field still
    # matches its declared Uniform (fidelity passes on the marginals).
    records = DistributionSampler(seed=2).sample_fields(
        _specs(Career), 4000, constraints=Career.__constraints__
    )
    report = fidelity_report(records, Career, alpha=0.01)
    assert report.passed, report.summary()


def test_resample_ordering_preserves_overlapping_marginals() -> None:
    # Overlapping but well-separated Normals: a few draws violate and get
    # redrawn, but the marginals stay put (unlike sort).
    class Overlap(BaseModel):
        low: Annotated[float, Normal(10, 1)]
        high: Annotated[float, Normal(20, 1)]
        __constraints__ = Constraints(Ordering("low", "high", method="resample"))

    records = DistributionSampler(seed=3).sample_fields(
        _specs(Overlap), 5000, constraints=Overlap.__constraints__
    )
    assert all(r["low"] <= r["high"] for r in records)
    assert statistics.mean(r["low"] for r in records) == pytest.approx(10, abs=0.2)
    assert statistics.mean(r["high"] for r in records) == pytest.approx(20, abs=0.2)


def test_resample_ordering_raises_when_budget_exhausted() -> None:
    # b is always ~5 and a always ~0, so requiring b <= a is essentially
    # impossible; rejection can never satisfy it.
    class Impossible(BaseModel):
        a: Annotated[float, Normal(0, 0.001)]
        b: Annotated[float, Normal(5, 0.001)]
        __constraints__ = Constraints(Ordering("b", "a", method="resample"))

    with pytest.raises(ValueError, match="could not be satisfied"):
        DistributionSampler(seed=0).sample_fields(
            _specs(Impossible), 200, constraints=Impossible.__constraints__
        )


def test_resample_ordering_rejects_conditional_field() -> None:
    class WithConditional(BaseModel):
        tier: Annotated[str, Categorical({"a": 0.5, "b": 0.5})]
        x: Annotated[
            float,
            Conditional(on="tier", cases={"a": Normal(1, 1)}, default=Normal(2, 1)),
        ]
        y: Annotated[float, Uniform(0, 10)]
        __constraints__ = Constraints(Ordering("x", "y", method="resample"))

    with pytest.raises(ValueError, match="conditional field"):
        DistributionSampler(seed=0).sample_fields(
            _specs(WithConditional), 50, constraints=WithConditional.__constraints__
        )


def test_resample_ordering_rejects_correlated_field() -> None:
    class WithCorrelation(BaseModel):
        a: Annotated[float, Uniform(0, 10)]
        b: Annotated[float, Uniform(0, 10)]
        __correlations__ = Correlations(("a", "b", 0.5))
        __constraints__ = Constraints(Ordering("a", "b", method="resample"))

    with pytest.raises(ValueError, match="correlated field"):
        DistributionSampler(seed=0).sample_fields(
            _specs(WithCorrelation),
            50,
            correlations=WithCorrelation.__correlations__,
            constraints=WithCorrelation.__constraints__,
        )


def test_resample_ordering_holds_on_converted_int_values() -> None:
    # One field rounds to int; the guarantee must hold on the values the record
    # exposes, not just the raw floats.
    class Mixed(BaseModel):
        start: Annotated[int, Uniform(0, 40)]
        end: Annotated[float, Uniform(40, 80)]
        __constraints__ = Constraints(Ordering("start", "end", method="resample"))

    records = DistributionSampler(seed=5).sample_fields(
        _specs(Mixed), 3000, constraints=Mixed.__constraints__
    )
    assert all(r["start"] <= r["end"] for r in records)


def test_resample_ordering_is_deterministic() -> None:
    a = DistributionSampler(seed=9).sample_fields(
        _specs(Career), 500, constraints=Career.__constraints__
    )
    b = DistributionSampler(seed=9).sample_fields(
        _specs(Career), 500, constraints=Career.__constraints__
    )
    assert a == b


def test_ordering_method_validation_and_repr() -> None:
    with pytest.raises(ValueError, match="method must be one of"):
        Ordering("a", "b", method="bogus")

    resample = Ordering("a", "b", method="resample")
    assert resample != Ordering("a", "b")  # method participates in equality
    assert repr(resample) == "Ordering('a', 'b', method='resample')"
    # to_code round-trips the method
    namespace = {"Constraints": Constraints, "Ordering": Ordering}
    cons = Constraints(resample)
    assert list(eval(cons.to_code(), namespace)) == [resample]  # noqa: S307


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
