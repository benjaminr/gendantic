"""Fidelity checks must pass on spec-compliant data and fail on wrong data.

These tests never touch an LLM: ``DistributionSampler`` produces spec-compliant
records deterministically from a seed, so the "good data passes" direction is
reproducible. The "bad data fails" direction feeds deliberately wrong values and
asserts the report flags the specific field or correlation.
"""

from typing import Annotated

from pydantic import BaseModel

from gendantic import (
    Beta,
    Binomial,
    Categorical,
    Correlations,
    DistributionSampler,
    LLMDrivenModelAnalyser,
    Normal,
    Poisson,
    Uniform,
    fidelity_report,
)


class Spec(BaseModel):
    """A model exercising every fidelity code path."""

    salary: Annotated[float, Normal(mean=50000, std=15000)]
    age: Annotated[float, Uniform(min=18, max=65)]
    dept: Annotated[str, Categorical(weights={"Eng": 0.4, "Sales": 0.3, "HR": 0.3})]
    errors: Annotated[int, Poisson(lam=3.0)]
    rate: Annotated[float, Beta(alpha=2, beta=5)]
    heads: Annotated[int, Binomial(n=10, p=0.5)]

    __correlations__ = Correlations(("age", "salary", 0.6))


def _generate(model: type[BaseModel], count: int, seed: int) -> list[dict]:
    specs = LLMDrivenModelAnalyser.extract_distribution_specs(model)
    correlations = getattr(model, "__correlations__", None)
    return DistributionSampler(seed=seed).sample_fields(specs, count, correlations)


def test_spec_compliant_data_passes() -> None:
    records = _generate(Spec, 2000, seed=42)
    report = fidelity_report(records, Spec)

    assert report.passed, report.summary()
    assert report.sample_size == 2000
    # Every distribution field is checked, none of the free-text kind exist here.
    assert {f.field for f in report.fields} == {
        "salary",
        "age",
        "dept",
        "errors",
        "rate",
        "heads",
    }
    assert {c.field1 for c in report.correlations} == {"age"}


def test_continuous_uses_ks_and_discrete_categorical_use_chi2() -> None:
    records = _generate(Spec, 1000, seed=7)
    report = fidelity_report(records, Spec)
    by_field = {f.field: f for f in report.fields}

    assert by_field["salary"].test == "ks"
    assert by_field["age"].test == "ks"
    assert by_field["rate"].test == "ks"
    assert by_field["errors"].test == "chi2"
    assert by_field["heads"].test == "chi2"
    assert by_field["dept"].test == "chi2"


def test_wrong_continuous_distribution_fails() -> None:
    # Salaries an order of magnitude off the declared Normal(50000, 15000).
    records = [
        {
            "salary": 5000.0,
            "age": 40.0,
            "dept": "Eng",
            "errors": 3,
            "rate": 0.3,
            "heads": 5,
        }
        for _ in range(500)
    ]
    report = fidelity_report(records, Spec)
    salary = next(f for f in report.fields if f.field == "salary")

    assert not salary.passed
    assert not report.passed


def test_wrong_categorical_frequencies_fail() -> None:
    records = _generate(Spec, 1000, seed=1)
    # Override department to be all "Eng" - far from the 40/30/30 split.
    for record in records:
        record["dept"] = "Eng"

    report = fidelity_report(records, Spec)
    dept = next(f for f in report.fields if f.field == "dept")

    assert not dept.passed


def test_broken_correlation_fails() -> None:
    class Uncorrelated(BaseModel):
        age: Annotated[float, Uniform(min=18, max=65)]
        salary: Annotated[float, Normal(mean=50000, std=15000)]

        __correlations__ = Correlations(("age", "salary", 0.8))

    # Sample the two fields independently (no correlation structure), so the
    # observed correlation is ~0, nowhere near the declared 0.8.
    specs = LLMDrivenModelAnalyser.extract_distribution_specs(Uncorrelated)
    records = DistributionSampler(seed=3).sample_fields(specs, 1000)

    report = fidelity_report(records, Uncorrelated)

    assert len(report.correlations) == 1
    corr = report.correlations[0]
    assert not corr.passed
    assert corr.error > 0.15
    assert not report.passed


def test_correlation_on_categorical_is_skipped() -> None:
    class WithCategorical(BaseModel):
        salary: Annotated[float, Normal(mean=50000, std=15000)]
        dept: Annotated[str, Categorical(weights={"A": 0.5, "B": 0.5})]

        # Categorical fields cannot be correlated - this pair must be skipped.
        __correlations__ = Correlations(("salary", "dept", 0.5))

    records = _generate(WithCategorical, 500, seed=9)
    report = fidelity_report(records, WithCategorical)

    assert report.correlations == []


def test_non_distribution_fields_are_ignored() -> None:
    class Mixed(BaseModel):
        salary: Annotated[float, Normal(mean=50000, std=15000)]
        note: str  # no distribution spec

    records = [
        {"salary": r["salary"], "note": "hello"}
        for r in DistributionSampler(seed=5).sample_fields(
            {"salary": Normal(mean=50000, std=15000)}, 500
        )
    ]
    report = fidelity_report(records, Mixed)

    assert {f.field for f in report.fields} == {"salary"}


def test_empty_records_returns_empty_report() -> None:
    report = fidelity_report([], Spec)
    assert report.sample_size == 0
    assert report.fields == []
    assert report.correlations == []
    assert report.passed


def test_accepts_model_instances() -> None:
    records = _generate(Spec, 500, seed=11)
    instances = [Spec(**r) for r in records]

    report = fidelity_report(instances, Spec)
    assert report.passed, report.summary()


def test_summary_is_printable_and_marks_status() -> None:
    records = _generate(Spec, 500, seed=13)
    report = fidelity_report(records, Spec)
    text = str(report)

    assert "Fidelity report" in text
    assert "PASS" in text
    assert "salary" in text
    assert "age~salary" in text
