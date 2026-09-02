"""
Statistical distribution specifications for gendantic.

These classes define statistical distributions that can be used with Python's
Annotated type hints to specify how field values should be sampled.

Usage:
    from typing import Annotated
    from pydantic import BaseModel, Field
    from gendantic import Normal, Uniform, Categorical, Correlations

    class Employee(BaseModel):
        salary: Annotated[int, Normal(mean=50000, std=15000)]
        age: Annotated[int, Uniform(min=18, max=65)]
        department: Annotated[str, Categorical(weights={"Eng": 0.4, "Sales": 0.6})]

        # Specify correlations between numeric fields
        __correlations__ = Correlations(
            ("age", "salary", 0.5),  # Moderate positive correlation
        )
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats


class DistributionSpec(ABC):
    """Base class for statistical distribution specifications."""

    @abstractmethod
    def sample(self, count: int, rng: np.random.Generator) -> np.ndarray:
        """
        Sample values from this distribution.

        Args:
            count: Number of values to sample
            rng: NumPy random generator for reproducibility

        Returns:
            Array of sampled values
        """
        pass

    @abstractmethod
    def quantile(self, u: np.ndarray) -> np.ndarray:
        """
        Compute the quantile function (inverse CDF).

        Used for copula-based correlated sampling.

        Args:
            u: Array of uniform [0, 1] values

        Returns:
            Array of values from this distribution
        """
        pass

    @property
    @abstractmethod
    def distribution_type(self) -> str:
        """Return the distribution type name."""
        pass

    @property
    def supports_correlation(self) -> bool:
        """Whether this distribution can participate in correlation structures."""
        return True


@dataclass(frozen=True)
class Normal(DistributionSpec):
    """
    Normal (Gaussian) distribution specification.

    Args:
        mean: Center of the distribution
        std: Standard deviation (spread)

    Example:
        salary: Annotated[int, Normal(mean=50000, std=15000)]
    """

    mean: float
    std: float

    def sample(self, count: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(self.mean, self.std, count)

    def quantile(self, u: np.ndarray) -> np.ndarray:
        return np.asarray(stats.norm.ppf(u, loc=self.mean, scale=self.std))

    @property
    def distribution_type(self) -> str:
        return "normal"


@dataclass(frozen=True)
class Uniform(DistributionSpec):
    """
    Uniform distribution specification.

    Values are spread evenly between min and max.

    Args:
        min: Lower bound (inclusive)
        max: Upper bound (exclusive)

    Example:
        age: Annotated[int, Uniform(min=18, max=65)]
    """

    min: float
    max: float

    def sample(self, count: int, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.min, self.max, count)

    def quantile(self, u: np.ndarray) -> np.ndarray:
        return self.min + (self.max - self.min) * u

    @property
    def distribution_type(self) -> str:
        return "uniform"


@dataclass(frozen=True)
class Categorical(DistributionSpec):
    """
    Categorical distribution with specified proportions.

    Note: Categorical fields do not participate in correlation structures;
    they are always sampled independently (see ``supports_correlation``).
    A correlation spec that references a categorical field is ignored with a
    warning.

    Args:
        weights: Dict mapping categories to their probabilities.
                 Probabilities should sum to 1.0.

    Example:
        department: Annotated[str, Categorical(weights={"Eng": 0.4, "Sales": 0.3, "HR": 0.3})]
    """

    weights: dict[str, float]

    def __post_init__(self) -> None:
        # Validate weights sum to approximately 1.0
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Categorical weights must sum to 1.0, got {total}. "
                f"Weights: {self.weights}"
            )

    def sample(self, count: int, rng: np.random.Generator) -> np.ndarray:
        categories = list(self.weights.keys())
        probabilities = list(self.weights.values())
        return rng.choice(categories, size=count, p=probabilities)

    def quantile(self, u: np.ndarray) -> np.ndarray:
        # For categorical, we use the inverse CDF approach
        categories = list(self.weights.keys())
        probabilities = list(self.weights.values())
        cumsum = np.cumsum(probabilities)
        indices = np.searchsorted(cumsum, u)
        indices = np.clip(indices, 0, len(categories) - 1)
        return np.array([categories[i] for i in indices])

    @property
    def distribution_type(self) -> str:
        return "categorical"

    @property
    def supports_correlation(self) -> bool:
        # Unordered categories have no well-defined correlation (any induced
        # dependency would hinge on arbitrary key order), so categorical fields
        # are always sampled independently, even inside a correlation structure.
        return False


@dataclass(frozen=True)
class LogNormal(DistributionSpec):
    """
    Log-normal distribution for right-skewed data.

    Useful for values that are always positive and have a long tail
    (e.g., incomes, file sizes, response times).

    Args:
        mean: Mean of the underlying normal distribution (mu)
        sigma: Standard deviation of the underlying normal distribution

    Note:
        The median of the log-normal is exp(mean).
        Most values cluster near exp(mean) with occasional large values.

    Example:
        income: Annotated[float, LogNormal(mean=10.8, sigma=0.5)]  # median ~50000
    """

    mean: float  # mu parameter (mean of log)
    sigma: float  # sigma parameter (std of log)

    def sample(self, count: int, rng: np.random.Generator) -> np.ndarray:
        return rng.lognormal(self.mean, self.sigma, count)

    def quantile(self, u: np.ndarray) -> np.ndarray:
        return np.asarray(stats.lognorm.ppf(u, s=self.sigma, scale=np.exp(self.mean)))

    @property
    def distribution_type(self) -> str:
        return "lognormal"


@dataclass(frozen=True)
class Exponential(DistributionSpec):
    """
    Exponential distribution for waiting times and decay processes.

    Args:
        scale: Scale parameter (1/lambda). This is the mean of the distribution.

    Example:
        wait_time: Annotated[float, Exponential(scale=5.0)]  # mean wait of 5 units
    """

    scale: float  # 1/lambda, equals the mean

    def sample(self, count: int, rng: np.random.Generator) -> np.ndarray:
        return rng.exponential(self.scale, count)

    def quantile(self, u: np.ndarray) -> np.ndarray:
        return np.asarray(stats.expon.ppf(u, scale=self.scale))

    @property
    def distribution_type(self) -> str:
        return "exponential"


@dataclass(frozen=True)
class Poisson(DistributionSpec):
    """
    Poisson distribution for count data.

    Useful for modelling number of events in a fixed interval
    (e.g., number of errors, number of purchases).

    Args:
        lam: Expected number of events (lambda parameter)

    Example:
        error_count: Annotated[int, Poisson(lam=2.5)]
    """

    lam: float  # lambda parameter (expected count)

    def sample(self, count: int, rng: np.random.Generator) -> np.ndarray:
        return rng.poisson(self.lam, count)

    def quantile(self, u: np.ndarray) -> np.ndarray:
        return np.asarray(stats.poisson.ppf(u, mu=self.lam))

    @property
    def distribution_type(self) -> str:
        return "poisson"


@dataclass(frozen=True)
class Beta(DistributionSpec):
    """
    Beta distribution for values bounded between 0 and 1.

    Useful for proportions, probabilities, and percentages.

    Args:
        alpha: Shape parameter alpha (>0)
        beta: Shape parameter beta (>0)

    Common shapes:
        - alpha=1, beta=1: Uniform on [0,1]
        - alpha=2, beta=2: Symmetric bell curve centred at 0.5
        - alpha=2, beta=5: Skewed toward 0
        - alpha=5, beta=2: Skewed toward 1

    Example:
        completion_rate: Annotated[float, Beta(alpha=2, beta=5)]
    """

    alpha: float
    beta: float

    def sample(self, count: int, rng: np.random.Generator) -> np.ndarray:
        return rng.beta(self.alpha, self.beta, count)

    def quantile(self, u: np.ndarray) -> np.ndarray:
        return np.asarray(stats.beta.ppf(u, self.alpha, self.beta))

    @property
    def distribution_type(self) -> str:
        return "beta"


@dataclass(frozen=True)
class Binomial(DistributionSpec):
    """
    Binomial distribution for number of successes in n trials.

    Args:
        n: Number of trials
        p: Probability of success in each trial

    Example:
        heads_count: Annotated[int, Binomial(n=10, p=0.5)]
    """

    n: int
    p: float

    def sample(self, count: int, rng: np.random.Generator) -> np.ndarray:
        return rng.binomial(self.n, self.p, count)

    def quantile(self, u: np.ndarray) -> np.ndarray:
        return np.asarray(stats.binom.ppf(u, self.n, self.p))

    @property
    def distribution_type(self) -> str:
        return "binomial"


class CopulaType:
    """
    Copula types for modelling different dependency structures.

    Each copula captures different aspects of how variables move together:

    - GAUSSIAN: Standard correlation, no tail dependence. Good for most business data.
    - STUDENT_T: Like Gaussian but with heavier tails. Variables are more likely to
      have extreme values together. Good for financial data.
    - CLAYTON: Lower tail dependence - variables crash together but don't boom together.
      Good for risk modelling, market downturns.
    - GUMBEL: Upper tail dependence - variables boom together but don't crash together.
      Good for success metrics, performance bonuses.
    - FRANK: Symmetric, no tail dependence. Similar to Gaussian but works well for
      weak to moderate correlations.
    """

    GAUSSIAN = "gaussian"
    STUDENT_T = "student_t"
    CLAYTON = "clayton"
    GUMBEL = "gumbel"
    FRANK = "frank"

    @classmethod
    def all_types(cls) -> list[str]:
        return [cls.GAUSSIAN, cls.STUDENT_T, cls.CLAYTON, cls.GUMBEL, cls.FRANK]


# Type alias for correlation spec: (field1, field2, correlation, copula_type)
CorrelationSpec = tuple[str, str, float] | tuple[str, str, float, str]


@dataclass
class Correlations:
    """
    Specify correlations between distribution-sampled fields with optional copula types.

    Uses copulas to preserve marginal distributions while enforcing correlation structure.
    Different copula types model different dependency patterns.

    Args:
        *specs: Tuples of (field1, field2, correlation) or
                (field1, field2, correlation, copula_type)
        default_copula: Default copula type when not specified per-pair

    Copula Types:
        - "gaussian": Standard correlation, no tail dependence (default)
        - "student_t": Heavy tails, extreme values occur together
        - "clayton": Lower tail dependence (things crash together)
        - "gumbel": Upper tail dependence (things boom together)
        - "frank": Symmetric, no tail dependence

    Example:
        class Employee(BaseModel):
            age: Annotated[int, Uniform(min=22, max=65)]
            salary: Annotated[float, Normal(mean=75000, std=20000)]
            performance: Annotated[float, Beta(alpha=5, beta=2)]
            risk_score: Annotated[float, Beta(alpha=2, beta=5)]

            __correlations__ = Correlations(
                ("age", "salary", 0.5),                        # Gaussian (default)
                ("performance", "salary", 0.6, "gumbel"),      # High performers get big bonuses
                ("risk_score", "salary", -0.3, "clayton"),     # High risk = lower salary (crashes together)
            )
    """

    _specs: list[tuple[str, str, float, str]] = field(default_factory=list)
    _default_copula: str = field(default=CopulaType.GAUSSIAN)

    def __init__(
        self,
        *specs: CorrelationSpec,
        default_copula: str = CopulaType.GAUSSIAN,
    ) -> None:
        object.__setattr__(self, "_specs", [])
        object.__setattr__(self, "_default_copula", default_copula)

        for spec in specs:
            if len(spec) == 3:
                field1, field2, corr = spec
                copula = default_copula
            elif len(spec) == 4:
                field1, field2, corr, copula = spec
            else:
                raise ValueError(
                    f"Each spec must be (field1, field2, corr) or (field1, field2, corr, copula), got {spec}"
                )

            if not isinstance(field1, str) or not isinstance(field2, str):
                raise ValueError(f"Field names must be strings, got {field1}, {field2}")
            if not -1 <= corr <= 1:
                raise ValueError(
                    f"Correlation must be between -1 and 1, got {corr} for ({field1}, {field2})"
                )
            if field1 == field2:
                raise ValueError(f"Cannot correlate a field with itself: {field1}")
            if copula not in CopulaType.all_types():
                raise ValueError(
                    f"Unknown copula type '{copula}'. Must be one of: {CopulaType.all_types()}"
                )

            self._specs.append((field1, field2, corr, copula))

    def __iter__(self) -> Iterator[tuple[str, str, float, str]]:
        return iter(self._specs)

    def __len__(self) -> int:
        return len(self._specs)

    @property
    def default_copula(self) -> str:
        return self._default_copula

    def get_correlation(self, field1: str, field2: str) -> tuple[float, str] | None:
        """Get the correlation and copula type between two fields."""
        for f1, f2, corr, copula in self._specs:
            if (f1 == field1 and f2 == field2) or (f1 == field2 and f2 == field1):
                return (corr, copula)
        return None

    def get_fields(self) -> set[str]:
        """Get all fields involved in correlations."""
        fields = set()
        for f1, f2, _, _ in self._specs:
            fields.add(f1)
            fields.add(f2)
        return fields

    def get_copula_groups(self) -> dict[str, list[tuple[str, str, float]]]:
        """
        Group correlations by copula type.

        Returns dict mapping copula type to list of (field1, field2, corr) tuples.
        """
        groups: dict[str, list[tuple[str, str, float]]] = {}
        for f1, f2, corr, copula in self._specs:
            if copula not in groups:
                groups[copula] = []
            groups[copula].append((f1, f2, corr))
        return groups

    def build_correlation_matrix(self, field_order: list[str]) -> np.ndarray:
        """
        Build a correlation matrix for the given field order.

        Note: This returns correlations only, ignoring copula types.
        Use get_copula_groups() for copula-aware sampling.
        """
        n = len(field_order)
        matrix = np.eye(n)

        field_to_idx = {f: i for i, f in enumerate(field_order)}

        for f1, f2, corr, _ in self._specs:
            if f1 in field_to_idx and f2 in field_to_idx:
                i, j = field_to_idx[f1], field_to_idx[f2]
                matrix[i, j] = corr
                matrix[j, i] = corr

        return matrix

    def __repr__(self) -> str:
        specs_str = ", ".join(
            f"({f1!r}, {f2!r}, {c}, {cop!r})" for f1, f2, c, cop in self._specs
        )
        return f"Correlations({specs_str})"

    def to_code(self) -> str:
        """Generate Python code representation for this Correlations instance."""
        lines = ["Correlations("]
        for f1, f2, corr, copula in self._specs:
            if copula == CopulaType.GAUSSIAN:
                lines.append(f'    ("{f1}", "{f2}", {corr}),')
            else:
                lines.append(f'    ("{f1}", "{f2}", {corr}, "{copula}"),')
        lines.append(")")
        return "\n".join(lines)


def is_distribution_spec(obj: Any) -> bool:
    """Check if an object is a distribution specification."""
    return isinstance(obj, DistributionSpec)
