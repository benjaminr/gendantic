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
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats


class DistributionSpec(ABC):
    """Base class for statistical distribution specifications."""

    @abstractmethod
    def sample(self, count: int, rng: np.random.Generator) -> NDArray[Any]:
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
    def quantile(self, u: NDArray[Any]) -> NDArray[Any]:
        """
        Compute the quantile function (inverse CDF).

        Used for copula-based correlated sampling.

        Args:
            u: Array of uniform [0, 1] values

        Returns:
            Array of values from this distribution
        """
        pass

    @abstractmethod
    def cdf(self, x: NDArray[Any]) -> NDArray[Any]:
        """
        Compute the cumulative distribution function.

        The inverse of :meth:`quantile`. Used for goodness-of-fit testing
        (e.g. Kolmogorov-Smirnov) when validating that generated samples match
        this distribution.

        Args:
            x: Array of values from this distribution's support

        Returns:
            Array of cumulative probabilities in [0, 1]
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

    def sample(self, count: int, rng: np.random.Generator) -> NDArray[Any]:
        return rng.normal(self.mean, self.std, count)

    def quantile(self, u: NDArray[Any]) -> NDArray[Any]:
        return np.asarray(stats.norm.ppf(u, loc=self.mean, scale=self.std))

    def cdf(self, x: NDArray[Any]) -> NDArray[Any]:
        return np.asarray(stats.norm.cdf(x, loc=self.mean, scale=self.std))

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

    def sample(self, count: int, rng: np.random.Generator) -> NDArray[Any]:
        return rng.uniform(self.min, self.max, count)

    def quantile(self, u: NDArray[Any]) -> NDArray[Any]:
        return self.min + (self.max - self.min) * u

    def cdf(self, x: NDArray[Any]) -> NDArray[Any]:
        return np.clip((np.asarray(x) - self.min) / (self.max - self.min), 0.0, 1.0)

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

    def sample(self, count: int, rng: np.random.Generator) -> NDArray[Any]:
        categories = list(self.weights.keys())
        probabilities = list(self.weights.values())
        return rng.choice(categories, size=count, p=probabilities)

    def quantile(self, u: NDArray[Any]) -> NDArray[Any]:
        # For categorical, we use the inverse CDF approach
        categories = list(self.weights.keys())
        probabilities = list(self.weights.values())
        cumsum = np.cumsum(probabilities)
        indices = np.searchsorted(cumsum, u)
        indices = np.clip(indices, 0, len(categories) - 1)
        return np.array([categories[i] for i in indices])

    def cdf(self, x: NDArray[Any]) -> NDArray[Any]:
        # Cumulative probability up to and including each category, using the
        # same weight order as ``quantile``. Categories carry no natural order,
        # so this exists for interface symmetry; goodness-of-fit uses observed
        # category frequencies directly rather than this CDF.
        categories = list(self.weights.keys())
        cumsum = np.cumsum(list(self.weights.values()))
        index_of = {category: i for i, category in enumerate(categories)}
        return np.array([cumsum[index_of[value]] for value in x])

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

    def sample(self, count: int, rng: np.random.Generator) -> NDArray[Any]:
        return rng.lognormal(self.mean, self.sigma, count)

    def quantile(self, u: NDArray[Any]) -> NDArray[Any]:
        return np.asarray(stats.lognorm.ppf(u, s=self.sigma, scale=np.exp(self.mean)))

    def cdf(self, x: NDArray[Any]) -> NDArray[Any]:
        return np.asarray(stats.lognorm.cdf(x, s=self.sigma, scale=np.exp(self.mean)))

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

    def sample(self, count: int, rng: np.random.Generator) -> NDArray[Any]:
        return rng.exponential(self.scale, count)

    def quantile(self, u: NDArray[Any]) -> NDArray[Any]:
        return np.asarray(stats.expon.ppf(u, scale=self.scale))

    def cdf(self, x: NDArray[Any]) -> NDArray[Any]:
        return np.asarray(stats.expon.cdf(x, scale=self.scale))

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

    def sample(self, count: int, rng: np.random.Generator) -> NDArray[Any]:
        return rng.poisson(self.lam, count)

    def quantile(self, u: NDArray[Any]) -> NDArray[Any]:
        return np.asarray(stats.poisson.ppf(u, mu=self.lam))

    def cdf(self, x: NDArray[Any]) -> NDArray[Any]:
        return np.asarray(stats.poisson.cdf(x, mu=self.lam))

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

    def sample(self, count: int, rng: np.random.Generator) -> NDArray[Any]:
        return rng.beta(self.alpha, self.beta, count)

    def quantile(self, u: NDArray[Any]) -> NDArray[Any]:
        return np.asarray(stats.beta.ppf(u, self.alpha, self.beta))

    def cdf(self, x: NDArray[Any]) -> NDArray[Any]:
        return np.asarray(stats.beta.cdf(x, self.alpha, self.beta))

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

    def sample(self, count: int, rng: np.random.Generator) -> NDArray[Any]:
        return rng.binomial(self.n, self.p, count)

    def quantile(self, u: NDArray[Any]) -> NDArray[Any]:
        return np.asarray(stats.binom.ppf(u, self.n, self.p))

    def cdf(self, x: NDArray[Any]) -> NDArray[Any]:
        return np.asarray(stats.binom.cdf(x, self.n, self.p))

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

    def __init__(
        self,
        *specs: CorrelationSpec,
        default_copula: str = CopulaType.GAUSSIAN,
    ) -> None:
        self._specs: list[tuple[str, str, float, str]] = []
        self._default_copula = default_copula

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

    def build_correlation_matrix(self, field_order: list[str]) -> NDArray[Any]:
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


@dataclass(frozen=True)
class Range:
    """
    A half-open numeric interval ``[min, max)`` used as a ``Conditional`` case key.

    Either bound may be ``None`` to leave that side unbounded. A value ``v``
    matches when ``(min is None or v >= min) and (max is None or v < max)`` -
    so adjacent ranges like ``Range(max=30)``, ``Range(30, 50)``,
    ``Range(min=50)`` tile the number line without overlap or gaps.

    Args:
        min: Inclusive lower bound, or ``None`` for unbounded below.
        max: Exclusive upper bound, or ``None`` for unbounded above.

    Example:
        Range(30, 50)     # 30 <= v < 50
        Range(max=30)     # v < 30
        Range(min=50)     # v >= 50
    """

    min: float | None = None
    max: float | None = None

    def __post_init__(self) -> None:
        if self.min is None and self.max is None:
            raise ValueError("Range must set at least one of min or max")
        if self.min is not None and self.max is not None and self.min >= self.max:
            raise ValueError(
                f"Range min ({self.min}) must be strictly less than max ({self.max})"
            )

    def contains(self, value: float) -> bool:
        """Whether ``value`` falls in this half-open interval."""
        if self.min is not None and value < self.min:
            return False
        if self.max is not None and value >= self.max:
            return False
        return True


# Keys of a Conditional's ``cases`` mapping: an exact discriminator value
# (str/int/bool/...) for equality matching, or a Range for numeric binning.
ConditionalKey = Any


@dataclass(frozen=True)
class Conditional:
    """
    A distribution whose parameters depend on another field's value.

    Placed in an ``Annotated`` type like any distribution spec, but instead of
    sampling from a single marginal it selects a distribution per record based
    on the already-sampled value of the ``on`` field.

    The ``cases`` mapping matches either exact discriminator values (for a
    categorical ``on`` field) or :class:`Range` intervals (for a numeric ``on``
    field); the two key styles cannot be mixed in one ``Conditional``. Any value
    matching no case falls back to ``default``.

    The ``on`` field must itself be a distribution-sampled field (a plain spec
    or another ``Conditional``); it cannot depend on an LLM-generated field,
    which is not known at sampling time. A ``Conditional`` field is sampled
    per-group and does not participate in copula correlations.

    Args:
        on: Name of the discriminator field this distribution depends on.
        cases: Mapping of discriminator value (or ``Range``) to distribution.
        default: Distribution used when no case matches. Required.

    Example:
        salary: Annotated[float, Conditional(
            on="department",
            cases={
                "Engineering": Normal(90000, 15000),
                "Sales": Normal(70000, 20000),
            },
            default=Normal(50000, 10000),
        )]
    """

    on: str
    cases: dict[ConditionalKey, DistributionSpec]
    default: DistributionSpec

    def __post_init__(self) -> None:
        if not self.on or not isinstance(self.on, str):
            raise ValueError("Conditional 'on' must be a non-empty field name")
        if not self.cases:
            raise ValueError("Conditional requires at least one case")
        if not isinstance(self.default, DistributionSpec):
            raise ValueError("Conditional 'default' must be a DistributionSpec")
        for key, spec in self.cases.items():
            if not isinstance(spec, DistributionSpec):
                raise ValueError(
                    f"Conditional case for {key!r} must map to a DistributionSpec, "
                    f"got {type(spec).__name__}"
                )
        range_keys = [isinstance(k, Range) for k in self.cases]
        if any(range_keys) and not all(range_keys):
            raise ValueError(
                "Conditional cannot mix exact-value keys and Range keys; "
                "use one style consistently"
            )

    def spec_for(self, value: Any) -> DistributionSpec:
        """Return the distribution to sample for a given discriminator value."""
        for key, spec in self.cases.items():
            if isinstance(key, Range):
                if key.contains(value):
                    return spec
            elif key == value:
                return spec
        return self.default


class Ordering:
    """
    An ascending ordering constraint across two or more numeric fields.

    Declares that, within every generated record, the listed fields are in
    non-decreasing order: ``fields[0] <= fields[1] <= ...``. Enforced after
    sampling by sorting each record's values across these fields.

    Note:
        Because enforcement sorts the sampled values, the constrained fields end
        up sharing the pooled distribution's order statistics - their individual
        marginals shift. This is the intended behaviour when the fields share a
        domain (e.g. two timestamps) and is not appropriate for fields with very
        different marginals.

    Args:
        *fields: Two or more field names, in the required ascending order.

    Example:
        __constraints__ = Constraints(Ordering("start", "end"))
    """

    def __init__(self, *fields: str) -> None:
        if len(fields) < 2:
            raise ValueError("Ordering requires at least two field names")
        for f in fields:
            if not isinstance(f, str):
                raise ValueError(f"Ordering field names must be strings, got {f!r}")
        if len(set(fields)) != len(fields):
            raise ValueError(f"Ordering field names must be unique, got {fields}")
        self.fields: tuple[str, ...] = fields

    def __repr__(self) -> str:
        return f"Ordering({', '.join(repr(f) for f in self.fields)})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Ordering) and other.fields == self.fields

    def __hash__(self) -> int:
        return hash(("Ordering", self.fields))


class Constraints:
    """
    A collection of cross-field constraints declared on a model.

    Attach to a model as ``__constraints__``, parallel to ``__correlations__``.
    Currently supports :class:`Ordering` constraints.

    Args:
        *constraints: One or more constraint objects (e.g. ``Ordering(...)``).

    Example:
        class Booking(BaseModel):
            start: Annotated[float, Uniform(0, 100)]
            end: Annotated[float, Uniform(0, 100)]
            __constraints__ = Constraints(Ordering("start", "end"))
    """

    def __init__(self, *constraints: Ordering) -> None:
        for c in constraints:
            if not isinstance(c, Ordering):
                raise ValueError(
                    f"Unsupported constraint {c!r}; expected an Ordering instance"
                )
        self._constraints: list[Ordering] = list(constraints)

    def __iter__(self) -> Iterator[Ordering]:
        return iter(self._constraints)

    def __len__(self) -> int:
        return len(self._constraints)

    def orderings(self) -> list[Ordering]:
        """Return the declared ordering constraints."""
        return [c for c in self._constraints if isinstance(c, Ordering)]

    def fields(self) -> set[str]:
        """All field names referenced by any constraint."""
        names: set[str] = set()
        for c in self._constraints:
            names.update(c.fields)
        return names

    def __repr__(self) -> str:
        return f"Constraints({', '.join(repr(c) for c in self._constraints)})"

    def to_code(self) -> str:
        """Generate a Python code representation for this Constraints instance."""
        if not self._constraints:
            return "Constraints()"
        lines = ["Constraints("]
        for c in self._constraints:
            lines.append(f"    {c!r},")
        lines.append(")")
        return "\n".join(lines)
