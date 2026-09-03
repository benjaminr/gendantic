"""Tests for distribution specifications and numpy sampling."""

from typing import Annotated

import numpy as np
import pytest
from pydantic import BaseModel, Field

from gendantic import (
    Beta,
    Binomial,
    Categorical,
    DistributionSampler,
    Exponential,
    LogNormal,
    Normal,
    Poisson,
    Uniform,
)
from gendantic.llm_driven_analyser import LLMDrivenModelAnalyser


class TestDistributionSpecs:
    """Test distribution specification classes."""

    def test_normal_distribution(self):
        """Test Normal distribution spec."""
        dist = Normal(mean=50000, std=15000)

        assert dist.distribution_type == "normal"
        assert dist.mean == 50000
        assert dist.std == 15000

        # Test sampling
        rng = np.random.default_rng(42)
        samples = dist.sample(1000, rng)

        assert len(samples) == 1000
        # Check that samples are roughly normal (within 3 std of mean)
        assert np.abs(np.mean(samples) - 50000) < 3000  # Within ~0.2 std
        assert np.abs(np.std(samples) - 15000) < 3000

    def test_uniform_distribution(self):
        """Test Uniform distribution spec."""
        dist = Uniform(min=18, max=65)

        assert dist.distribution_type == "uniform"
        assert dist.min == 18
        assert dist.max == 65

        # Test sampling
        rng = np.random.default_rng(42)
        samples = dist.sample(1000, rng)

        assert len(samples) == 1000
        assert np.all(samples >= 18)
        assert np.all(samples < 65)
        # Uniform mean should be (18+65)/2 = 41.5
        assert np.abs(np.mean(samples) - 41.5) < 2

    def test_categorical_distribution(self):
        """Test Categorical distribution spec."""
        dist = Categorical(weights={"Eng": 0.4, "Sales": 0.3, "HR": 0.3})

        assert dist.distribution_type == "categorical"
        assert dist.weights == {"Eng": 0.4, "Sales": 0.3, "HR": 0.3}

        # Test sampling
        rng = np.random.default_rng(42)
        samples = dist.sample(1000, rng)

        assert len(samples) == 1000
        # Check approximate proportions
        eng_count = np.sum(samples == "Eng")
        sales_count = np.sum(samples == "Sales")
        hr_count = np.sum(samples == "HR")

        # Allow 5% tolerance
        assert abs(eng_count / 1000 - 0.4) < 0.05
        assert abs(sales_count / 1000 - 0.3) < 0.05
        assert abs(hr_count / 1000 - 0.3) < 0.05

    def test_categorical_weights_must_sum_to_one(self):
        """Test that Categorical raises error for invalid weights."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            Categorical(weights={"A": 0.3, "B": 0.3})  # Sums to 0.6

    def test_categorical_weights_within_tolerance_are_renormalised(self):
        """Weights that only sum to ~1.0 (allowed) must still sample/quantile.

        numpy's sampler requires probabilities that sum to exactly 1.0, so a
        spec accepted within the 0.01 tolerance would otherwise crash at sample
        time. The spec renormalises internally.
        """
        # Sums to 0.995 -- inside the 0.01 tolerance, but not exactly 1.0.
        dist = Categorical(weights={"A": 0.4, "B": 0.3, "C": 0.295})
        rng = np.random.default_rng(0)

        samples = dist.sample(500, rng)
        assert len(samples) == 500
        assert set(np.unique(samples)) <= {"A", "B", "C"}

        # quantile round-trip over the full unit interval hits every category.
        q = dist.quantile(np.array([0.0, 0.5, 0.999]))
        assert set(q) <= {"A", "B", "C"}

    def test_lognormal_distribution(self):
        """Test LogNormal distribution spec."""
        dist = LogNormal(mean=10.8, sigma=0.5)  # median ~50000

        assert dist.distribution_type == "lognormal"

        rng = np.random.default_rng(42)
        samples = dist.sample(1000, rng)

        assert len(samples) == 1000
        assert np.all(samples > 0)  # Log-normal is always positive

    def test_exponential_distribution(self):
        """Test Exponential distribution spec."""
        dist = Exponential(scale=5.0)

        assert dist.distribution_type == "exponential"
        assert dist.scale == 5.0

        rng = np.random.default_rng(42)
        samples = dist.sample(1000, rng)

        assert len(samples) == 1000
        assert np.all(samples >= 0)
        # Mean of exponential should be scale
        assert np.abs(np.mean(samples) - 5.0) < 0.5

    def test_poisson_distribution(self):
        """Test Poisson distribution spec."""
        dist = Poisson(lam=3.5)

        assert dist.distribution_type == "poisson"
        assert dist.lam == 3.5

        rng = np.random.default_rng(42)
        samples = dist.sample(1000, rng)

        assert len(samples) == 1000
        assert np.all(samples >= 0)
        assert np.all(samples == samples.astype(int))  # All integers
        # Mean should be lambda
        assert np.abs(np.mean(samples) - 3.5) < 0.3

    def test_beta_distribution(self):
        """Test Beta distribution spec."""
        dist = Beta(alpha=2, beta=5)

        assert dist.distribution_type == "beta"

        rng = np.random.default_rng(42)
        samples = dist.sample(1000, rng)

        assert len(samples) == 1000
        assert np.all(samples >= 0)
        assert np.all(samples <= 1)

    def test_binomial_distribution(self):
        """Test Binomial distribution spec."""
        dist = Binomial(n=10, p=0.5)

        assert dist.distribution_type == "binomial"

        rng = np.random.default_rng(42)
        samples = dist.sample(1000, rng)

        assert len(samples) == 1000
        assert np.all(samples >= 0)
        assert np.all(samples <= 10)
        # Mean should be n*p = 5
        assert np.abs(np.mean(samples) - 5) < 0.3


class TestDistributionSampler:
    """Test the DistributionSampler class."""

    def test_sample_multiple_fields(self):
        """Test sampling multiple fields at once."""
        sampler = DistributionSampler(seed=42)

        specs = {
            "salary": Normal(mean=50000, std=15000),
            "age": Uniform(min=18, max=65),
            "department": Categorical(weights={"Eng": 0.5, "Sales": 0.5}),
        }

        records = sampler.sample_fields(specs, count=100)

        assert len(records) == 100
        assert all("salary" in r for r in records)
        assert all("age" in r for r in records)
        assert all("department" in r for r in records)

        # Check types are Python native (not numpy)
        assert all(isinstance(r["salary"], float) for r in records)
        assert all(isinstance(r["age"], float) for r in records)
        assert all(isinstance(r["department"], str) for r in records)

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same results."""
        specs = {"value": Normal(mean=100, std=10)}

        sampler1 = DistributionSampler(seed=42)
        records1 = sampler1.sample_fields(specs, count=10)

        sampler2 = DistributionSampler(seed=42)
        records2 = sampler2.sample_fields(specs, count=10)

        assert records1 == records2

    def test_different_seeds_different_results(self):
        """Test that different seeds produce different results."""
        specs = {"value": Normal(mean=100, std=10)}

        sampler1 = DistributionSampler(seed=42)
        records1 = sampler1.sample_fields(specs, count=10)

        sampler2 = DistributionSampler(seed=123)
        records2 = sampler2.sample_fields(specs, count=10)

        assert records1 != records2

    def test_empty_specs(self):
        """Test sampling with no distribution specs."""
        sampler = DistributionSampler(seed=42)
        records = sampler.sample_fields({}, count=5)

        assert len(records) == 5
        assert all(r == {} for r in records)

    def test_unexpected_tuple_length_rejected(self):
        """A fully-specified spec must be a (spec, type, constraints) 3-tuple."""
        sampler = DistributionSampler(seed=42)
        bad_specs = {"value": (Normal(mean=0, std=1), float)}  # 2-tuple

        with pytest.raises(ValueError, match="Unexpected tuple length"):
            sampler.sample_fields(bad_specs, count=5)  # type: ignore[arg-type]


class TestDistributionExtraction:
    """Test extraction of distribution specs from Pydantic models."""

    def test_extract_from_annotated_model(self):
        """Test extracting distribution specs from a model with Annotated types."""

        class TestModel(BaseModel):
            salary: Annotated[int, Normal(mean=50000, std=15000)]
            age: Annotated[int, Uniform(min=18, max=65)]
            department: Annotated[str, Categorical(weights={"Eng": 0.5, "Sales": 0.5})]
            name: str  # No distribution

        specs = LLMDrivenModelAnalyser.extract_distribution_specs(TestModel)

        assert "salary" in specs
        assert "age" in specs
        assert "department" in specs
        assert "name" not in specs

        assert isinstance(specs["salary"], Normal)
        assert isinstance(specs["age"], Uniform)
        assert isinstance(specs["department"], Categorical)

    def test_extract_with_field_constraints(self):
        """Test that Field constraints don't interfere with distribution extraction."""

        class TestModel(BaseModel):
            value: Annotated[int, Normal(mean=100, std=20), Field(ge=0, le=200)]

        specs = LLMDrivenModelAnalyser.extract_distribution_specs(TestModel)

        assert "value" in specs
        assert isinstance(specs["value"], Normal)
        assert specs["value"].mean == 100
        assert specs["value"].std == 20

    def test_extract_no_distributions(self):
        """Test extracting from a model with no distribution specs."""

        class PlainModel(BaseModel):
            name: str
            age: int = Field(ge=0)

        specs = LLMDrivenModelAnalyser.extract_distribution_specs(PlainModel)

        assert specs == {}

    def test_only_first_distribution_per_field(self):
        """Test that only the first distribution spec is used per field."""

        class TestModel(BaseModel):
            # This is unusual but test that we handle it
            value: Annotated[int, Normal(mean=50, std=10), Uniform(min=0, max=100)]

        specs = LLMDrivenModelAnalyser.extract_distribution_specs(TestModel)

        assert "value" in specs
        # Should get the first one (Normal)
        assert isinstance(specs["value"], Normal)


class TestCorrelations:
    """Test the Correlations class and correlated sampling."""

    def test_correlations_creation(self):
        """Test creating a Correlations instance."""
        from gendantic import Correlations

        corr = Correlations(
            ("age", "salary", 0.5),
            ("age", "experience", 0.8),
        )

        assert len(corr) == 2
        # get_correlation returns (correlation, copula_type) tuple
        assert corr.get_correlation("age", "salary") == (0.5, "gaussian")
        assert corr.get_correlation("salary", "age") == (0.5, "gaussian")  # Symmetric
        assert corr.get_correlation("age", "experience") == (0.8, "gaussian")
        assert corr.get_correlation("salary", "experience") is None

    def test_correlations_validation(self):
        """Test that Correlations validates inputs."""
        from gendantic import Correlations

        # Invalid correlation value
        with pytest.raises(ValueError, match="between -1 and 1"):
            Correlations(("a", "b", 1.5))

        # Self-correlation
        with pytest.raises(ValueError, match="with itself"):
            Correlations(("a", "a", 0.5))

    def test_correlations_reject_bad_tuple_length(self):
        """A spec that is not a 3- or 4-tuple is rejected."""
        from gendantic import Correlations

        with pytest.raises(ValueError, match="Each spec must be"):
            Correlations(("a", "b"))  # type: ignore[arg-type]

    def test_correlations_reject_non_string_fields(self):
        """Field names must be strings."""
        from gendantic import Correlations

        with pytest.raises(ValueError, match="must be strings"):
            Correlations((1, "b", 0.5))  # type: ignore[arg-type]

    def test_correlations_repr_roundtrips_specs(self):
        """repr() shows every spec as a 4-tuple including the copula."""
        from gendantic import Correlations

        corr = Correlations(("a", "b", 0.5), ("b", "c", 0.3, "gumbel"))
        text = repr(corr)
        assert text.startswith("Correlations(")
        assert "('a', 'b', 0.5, 'gaussian')" in text
        assert "('b', 'c', 0.3, 'gumbel')" in text

    def test_correlations_get_fields(self):
        """Test getting all fields in correlations."""
        from gendantic import Correlations

        corr = Correlations(
            ("a", "b", 0.5),
            ("b", "c", 0.3),
        )

        fields = corr.get_fields()
        assert fields == {"a", "b", "c"}

    def test_build_correlation_matrix(self):
        """Test building a correlation matrix."""
        from gendantic import Correlations

        corr = Correlations(
            ("age", "salary", 0.5),
            ("age", "experience", 0.8),
        )

        matrix = corr.build_correlation_matrix(["age", "salary", "experience"])

        # Check diagonal is 1
        assert matrix[0, 0] == 1.0
        assert matrix[1, 1] == 1.0
        assert matrix[2, 2] == 1.0

        # Check correlations
        assert matrix[0, 1] == 0.5  # age-salary
        assert matrix[1, 0] == 0.5  # symmetric
        assert matrix[0, 2] == 0.8  # age-experience
        assert matrix[2, 0] == 0.8  # symmetric
        assert matrix[1, 2] == 0.0  # salary-experience (not specified)

    def test_correlated_sampling(self):
        """Test that correlated sampling produces correlated values."""
        from gendantic import Correlations

        sampler = DistributionSampler(seed=42)

        specs = {
            "x": Normal(mean=0, std=1),
            "y": Normal(mean=0, std=1),
        }

        # High positive correlation
        correlations = Correlations(("x", "y", 0.9))

        records = sampler.sample_fields(specs, count=1000, correlations=correlations)

        x_values = np.array([r["x"] for r in records])
        y_values = np.array([r["y"] for r in records])

        # Calculate observed correlation
        observed_corr = np.corrcoef(x_values, y_values)[0, 1]

        # Should be close to specified correlation (allow some sampling variance)
        assert abs(observed_corr - 0.9) < 0.1

    def test_negative_correlation(self):
        """Test negative correlation."""
        from gendantic import Correlations

        sampler = DistributionSampler(seed=42)

        specs = {
            "x": Normal(mean=0, std=1),
            "y": Normal(mean=0, std=1),
        }

        correlations = Correlations(("x", "y", -0.8))

        records = sampler.sample_fields(specs, count=1000, correlations=correlations)

        x_values = np.array([r["x"] for r in records])
        y_values = np.array([r["y"] for r in records])

        observed_corr = np.corrcoef(x_values, y_values)[0, 1]

        # Should be negative and close to -0.8
        assert observed_corr < 0
        assert abs(observed_corr - (-0.8)) < 0.1

    def test_correlated_uniform_distributions(self):
        """Test correlation with uniform distributions."""
        from gendantic import Correlations

        sampler = DistributionSampler(seed=42)

        specs = {
            "age": Uniform(min=20, max=60),
            "salary": Uniform(min=30000, max=100000),
        }

        correlations = Correlations(("age", "salary", 0.7))

        records = sampler.sample_fields(specs, count=1000, correlations=correlations)

        ages = np.array([r["age"] for r in records])
        salaries = np.array([r["salary"] for r in records])

        # Check marginals are still within bounds
        assert np.all(ages >= 20) and np.all(ages <= 60)
        assert np.all(salaries >= 30000) and np.all(salaries <= 100000)

        # Check correlation
        observed_corr = np.corrcoef(ages, salaries)[0, 1]
        assert abs(observed_corr - 0.7) < 0.15

    def test_three_field_correlations(self):
        """Test correlations with three fields."""
        from gendantic import Correlations

        sampler = DistributionSampler(seed=42)

        specs = {
            "age": Uniform(min=22, max=65),
            "experience": Uniform(min=0, max=40),
            "salary": Normal(mean=75000, std=20000),
        }

        correlations = Correlations(
            ("age", "experience", 0.85),  # Strong: older = more experience
            ("experience", "salary", 0.6),  # Moderate: more experience = higher salary
            ("age", "salary", 0.4),  # Weaker direct age-salary correlation
        )

        records = sampler.sample_fields(specs, count=1000, correlations=correlations)

        ages = np.array([r["age"] for r in records])
        experience = np.array([r["experience"] for r in records])
        salaries = np.array([r["salary"] for r in records])

        # Check all correlations are approximately correct
        corr_matrix = np.corrcoef([ages, experience, salaries])

        assert abs(corr_matrix[0, 1] - 0.85) < 0.15  # age-experience
        assert abs(corr_matrix[1, 2] - 0.6) < 0.15  # experience-salary
        assert abs(corr_matrix[0, 2] - 0.4) < 0.15  # age-salary

    def test_model_with_correlations_attribute(self):
        """Test that models can define correlations via __correlations__."""
        from gendantic import Correlations

        class Employee(BaseModel):
            age: Annotated[int, Uniform(min=22, max=65)]
            salary: Annotated[float, Normal(mean=75000, std=20000)]

            __correlations__ = Correlations(("age", "salary", 0.5))

        # Check that __correlations__ is accessible
        assert hasattr(Employee, "__correlations__")
        assert isinstance(Employee.__correlations__, Correlations)
        # get_correlation returns (correlation, copula_type) tuple
        assert Employee.__correlations__.get_correlation("age", "salary") == (
            0.5,
            "gaussian",
        )

    def test_reproducibility_with_correlations(self):
        """Test that correlated sampling is reproducible with seeds."""
        from gendantic import Correlations

        specs = {
            "x": Normal(mean=0, std=1),
            "y": Normal(mean=0, std=1),
        }
        correlations = Correlations(("x", "y", 0.7))

        sampler1 = DistributionSampler(seed=42)
        records1 = sampler1.sample_fields(specs, count=10, correlations=correlations)

        sampler2 = DistributionSampler(seed=42)
        records2 = sampler2.sample_fields(specs, count=10, correlations=correlations)

        assert records1 == records2


class TestCopulaTypes:
    """Test different copula types for modelling dependencies."""

    def test_copula_type_constants(self):
        """Test CopulaType constants are correct."""
        from gendantic import CopulaType

        assert CopulaType.GAUSSIAN == "gaussian"
        assert CopulaType.STUDENT_T == "student_t"
        assert CopulaType.CLAYTON == "clayton"
        assert CopulaType.GUMBEL == "gumbel"
        assert CopulaType.FRANK == "frank"
        assert len(CopulaType.all_types()) == 5

    def test_correlations_with_copula_type(self):
        """Test creating correlations with explicit copula types."""
        from gendantic import Correlations

        corr = Correlations(
            ("age", "salary", 0.5),  # Default gaussian
            ("performance", "bonus", 0.7, "gumbel"),  # Upper tail
            ("risk", "loss", 0.6, "clayton"),  # Lower tail
        )

        assert corr.get_correlation("age", "salary") == (0.5, "gaussian")
        assert corr.get_correlation("performance", "bonus") == (0.7, "gumbel")
        assert corr.get_correlation("risk", "loss") == (0.6, "clayton")

    def test_invalid_copula_type_rejected(self):
        """Test that invalid copula types are rejected."""
        from gendantic import Correlations

        with pytest.raises(ValueError, match="Unknown copula type"):
            Correlations(("a", "b", 0.5, "invalid_copula"))

    def test_copula_groups(self):
        """Test grouping correlations by copula type."""
        from gendantic import Correlations

        corr = Correlations(
            ("a", "b", 0.5),  # gaussian (default)
            ("c", "d", 0.6, "gaussian"),
            ("e", "f", 0.7, "gumbel"),
            ("g", "h", 0.8, "clayton"),
        )

        groups = corr.get_copula_groups()
        assert "gaussian" in groups
        assert "gumbel" in groups
        assert "clayton" in groups
        assert len(groups["gaussian"]) == 2
        assert len(groups["gumbel"]) == 1
        assert len(groups["clayton"]) == 1

    def test_student_t_copula_sampling(self):
        """Test Student's t copula produces correlated samples with heavier tails."""
        from gendantic import Correlations

        sampler = DistributionSampler(seed=42)

        specs = {
            "x": Normal(mean=0, std=1),
            "y": Normal(mean=0, std=1),
        }

        correlations = Correlations(("x", "y", 0.7, "student_t"))
        records = sampler.sample_fields(specs, count=1000, correlations=correlations)

        x_values = np.array([r["x"] for r in records])
        y_values = np.array([r["y"] for r in records])

        # Should still produce correlated values
        observed_corr = np.corrcoef(x_values, y_values)[0, 1]
        assert abs(observed_corr - 0.7) < 0.2  # Allow more variance for t-copula

    def test_clayton_copula_sampling(self):
        """Test Clayton copula (lower tail dependence) sampling."""
        from gendantic import Correlations

        sampler = DistributionSampler(seed=42)

        specs = {
            "x": Normal(mean=0, std=1),
            "y": Normal(mean=0, std=1),
        }

        correlations = Correlations(("x", "y", 0.6, "clayton"))
        records = sampler.sample_fields(specs, count=1000, correlations=correlations)

        x_values = np.array([r["x"] for r in records])
        y_values = np.array([r["y"] for r in records])

        # Should produce positively correlated values
        observed_corr = np.corrcoef(x_values, y_values)[0, 1]
        assert observed_corr > 0.2  # Clayton should produce positive correlation

    def test_gumbel_copula_sampling(self):
        """Test Gumbel copula (upper tail dependence) sampling."""
        from gendantic import Correlations

        sampler = DistributionSampler(seed=42)

        specs = {
            "x": Normal(mean=0, std=1),
            "y": Normal(mean=0, std=1),
        }

        correlations = Correlations(("x", "y", 0.6, "gumbel"))
        records = sampler.sample_fields(specs, count=1000, correlations=correlations)

        x_values = np.array([r["x"] for r in records])
        y_values = np.array([r["y"] for r in records])

        # Should produce positively correlated values
        observed_corr = np.corrcoef(x_values, y_values)[0, 1]
        assert observed_corr > 0.2  # Gumbel should produce positive correlation

    def test_frank_copula_sampling(self):
        """Test Frank copula (symmetric, no tail dependence) sampling."""
        from gendantic import Correlations

        sampler = DistributionSampler(seed=42)

        specs = {
            "x": Normal(mean=0, std=1),
            "y": Normal(mean=0, std=1),
        }

        correlations = Correlations(("x", "y", 0.5, "frank"))
        records = sampler.sample_fields(specs, count=1000, correlations=correlations)

        x_values = np.array([r["x"] for r in records])
        y_values = np.array([r["y"] for r in records])

        # Should produce correlated values
        observed_corr = np.corrcoef(x_values, y_values)[0, 1]
        assert observed_corr > 0  # Should be positive for positive input correlation

    def test_default_copula_parameter(self):
        """Test the default_copula parameter for Correlations."""
        from gendantic import Correlations

        # Default all to gumbel
        corr = Correlations(
            ("a", "b", 0.5),
            ("c", "d", 0.6),
            default_copula="gumbel",
        )

        assert corr.get_correlation("a", "b") == (0.5, "gumbel")
        assert corr.get_correlation("c", "d") == (0.6, "gumbel")
        assert corr.default_copula == "gumbel"

    def test_correlations_to_code(self):
        """Test generating Python code from Correlations."""
        from gendantic import Correlations

        corr = Correlations(
            ("age", "salary", 0.5),
            ("performance", "bonus", 0.7, "gumbel"),
        )

        code = corr.to_code()
        assert "Correlations(" in code
        assert '("age", "salary", 0.5)' in code
        assert '("performance", "bonus", 0.7, "gumbel")' in code
