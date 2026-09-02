"""
Distribution sampler for generating field values using numpy.

This module provides the DistributionSampler class which orchestrates
numpy-based sampling for fields that have distribution specifications.
Supports correlated sampling via multiple copula types.
"""

import logging
from typing import Any

import numpy as np
from scipy import integrate, optimize, stats

from .distributions import CopulaType, Correlations, DistributionSpec

logger = logging.getLogger("gendantic")


class DistributionSampler:
    """
    Samples values for fields with distribution specifications.

    Uses numpy's random generator for reproducible, statistically correct sampling.
    Supports correlated sampling using various copula types.

    Args:
        seed: Optional seed for reproducibility. Same seed produces same samples.

    Copula Types:
        - gaussian: Standard correlation via multivariate normal
        - student_t: Heavy tails, extreme values together
        - clayton: Lower tail dependence (things crash together)
        - gumbel: Upper tail dependence (things boom together)
        - frank: Symmetric, no tail dependence
    """

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.seed = seed

    def sample_fields(
        self,
        distribution_specs: (
            dict[str, DistributionSpec]
            | dict[str, tuple[DistributionSpec, type]]
            | dict[str, tuple[DistributionSpec, type, dict[str, float | None]]]
        ),
        count: int,
        correlations: Correlations | None = None,
    ) -> list[dict[str, Any]]:
        """
        Sample all distribution fields and return as list of partial records.

        If correlations are specified, uses appropriate copulas to preserve
        marginal distributions while enforcing the correlation/dependency structure.

        Args:
            distribution_specs: Dict of {field: DistributionSpec} or
                               {field: (DistributionSpec, target_type)} or
                               {field: (DistributionSpec, target_type, constraints)}
            count: Number of records to sample
            correlations: Optional correlation structure
        """
        if not distribution_specs:
            return [{} for _ in range(count)]

        # Normalise to (spec, type, constraints) format
        normalised_specs: dict[
            str, tuple[DistributionSpec, type, dict[str, float | None]]
        ] = {}
        empty_constraints: dict[str, float | None] = {
            "ge": None,
            "le": None,
            "gt": None,
            "lt": None,
        }

        for field_name, spec_or_tuple in distribution_specs.items():
            if isinstance(spec_or_tuple, tuple):
                if len(spec_or_tuple) == 3:
                    normalised_specs[field_name] = spec_or_tuple
                elif len(spec_or_tuple) == 2:
                    spec, target_type = spec_or_tuple
                    normalised_specs[field_name] = (
                        spec,
                        target_type,
                        empty_constraints.copy(),
                    )
                else:
                    raise ValueError(
                        f"Unexpected tuple length for {field_name}: {len(spec_or_tuple)}"
                    )
            else:
                # Legacy format without type - default to float
                normalised_specs[field_name] = (
                    spec_or_tuple,
                    float,
                    empty_constraints.copy(),
                )

        if correlations is None or len(correlations) == 0:
            return self._sample_independent(normalised_specs, count)

        return self._sample_correlated(normalised_specs, count, correlations)

    def _sample_independent(
        self,
        distribution_specs: dict[
            str, tuple[DistributionSpec, type, dict[str, float | None]]
        ],
        count: int,
    ) -> list[dict[str, Any]]:
        """Sample fields independently (no correlation structure)."""
        samples: dict[str, np.ndarray] = {}
        target_types: dict[str, type] = {}
        constraints_map: dict[str, dict[str, float | None]] = {}

        for field_name, (spec, target_type, constraints) in distribution_specs.items():
            samples[field_name] = spec.sample(count, self.rng)
            target_types[field_name] = target_type
            constraints_map[field_name] = constraints

        return [
            {
                field_name: self._convert_numpy_value(
                    samples[field_name][i],
                    target_types[field_name],
                    constraints_map[field_name],
                )
                for field_name in samples
            }
            for i in range(count)
        ]

    def _sample_correlated(
        self,
        distribution_specs: dict[
            str, tuple[DistributionSpec, type, dict[str, float | None]]
        ],
        count: int,
        correlations: Correlations,
    ) -> list[dict[str, Any]]:
        """
        Sample fields with correlation structure using copulas.

        Fields whose distribution does not support correlation (e.g.
        Categorical) are sampled independently; only correlation-capable
        fields are drawn through the copula. Correlation specs that reference
        an unsupported field are ignored with a warning.

        For simplicity and consistency, we use the dominant copula type
        for the entire correlation structure. If mixed copulas are specified,
        we use Gaussian for the overall structure but apply tail adjustments.
        """
        target_types: dict[str, type] = {
            name: t for name, (_, t, _) in distribution_specs.items()
        }
        constraints_map: dict[str, dict[str, float | None]] = {
            name: c for name, (_, _, c) in distribution_specs.items()
        }
        samples: dict[str, np.ndarray] = {}

        # Partition fields into correlation-capable and independent
        corr_fields = [
            name
            for name, (spec, _, _) in distribution_specs.items()
            if spec.supports_correlation
        ]
        indep_fields = [name for name in distribution_specs if name not in corr_fields]

        # Warn if correlations reference a field that cannot be correlated
        excluded_referenced = correlations.get_fields() & set(indep_fields)
        if excluded_referenced:
            logger.warning(
                "Correlation spec references field(s) %s whose distribution does "
                "not support correlation (e.g. Categorical); these are sampled "
                "independently.",
                ", ".join(sorted(excluded_referenced)),
            )

        # Independent fields: sample directly from their marginal
        for name in indep_fields:
            samples[name] = distribution_specs[name][0].sample(count, self.rng)

        # Determine whether any real correlations remain among capable fields
        corr_matrix = correlations.build_correlation_matrix(corr_fields)
        has_correlation = len(corr_fields) >= 2 and not np.allclose(
            corr_matrix, np.eye(len(corr_fields))
        )

        if not has_correlation:
            # Nothing left to correlate - sample capable fields independently too
            for name in corr_fields:
                samples[name] = distribution_specs[name][0].sample(count, self.rng)
        else:
            # Determine dominant copula from specs among capable fields
            copula_groups = correlations.get_copula_groups()
            if len(copula_groups) == 1:
                dominant_copula = next(iter(copula_groups))
            else:
                # Mixed copulas - use Gaussian as base
                dominant_copula = CopulaType.GAUSSIAN

            corr_matrix = self._ensure_positive_semidefinite(corr_matrix)

            try:
                u_samples = self._sample_copula(
                    dominant_copula, corr_matrix, count, len(corr_fields)
                )
            except Exception:
                # Fall back to independent sampling for capable fields
                for name in corr_fields:
                    samples[name] = distribution_specs[name][0].sample(count, self.rng)
            else:
                for i, name in enumerate(corr_fields):
                    u_col = np.clip(u_samples[:, i], 1e-10, 1 - 1e-10)
                    samples[name] = distribution_specs[name][0].quantile(u_col)

        return [
            {
                name: self._convert_numpy_value(
                    samples[name][i],
                    target_types[name],
                    constraints_map[name],
                )
                for name in distribution_specs
            }
            for i in range(count)
        ]

    def _sample_copula(
        self,
        copula_type: str,
        corr_matrix: np.ndarray,
        count: int,
        n_dims: int,
    ) -> np.ndarray:
        """
        Sample from a copula, returning uniform [0,1] values.

        Returns array of shape (count, n_dims) with values in [0, 1].
        """
        if copula_type == CopulaType.GAUSSIAN:
            return self._sample_gaussian_copula(corr_matrix, count, n_dims)
        elif copula_type == CopulaType.STUDENT_T:
            return self._sample_student_t_copula(corr_matrix, count, n_dims, df=4)
        elif copula_type == CopulaType.CLAYTON:
            return self._sample_clayton_copula(corr_matrix, count, n_dims)
        elif copula_type == CopulaType.GUMBEL:
            return self._sample_gumbel_copula(corr_matrix, count, n_dims)
        elif copula_type == CopulaType.FRANK:
            return self._sample_frank_copula(corr_matrix, count, n_dims)
        else:
            # Default to Gaussian
            return self._sample_gaussian_copula(corr_matrix, count, n_dims)

    def _sample_gaussian_copula(
        self, corr_matrix: np.ndarray, count: int, n_dims: int
    ) -> np.ndarray:
        """
        Sample from Gaussian copula.

        Standard approach: multivariate normal -> standard normal CDF.
        """
        mean = np.zeros(n_dims)
        z_samples = self.rng.multivariate_normal(mean, corr_matrix, size=count)
        return np.asarray(stats.norm.cdf(z_samples))

    def _sample_student_t_copula(
        self, corr_matrix: np.ndarray, count: int, n_dims: int, df: int = 4
    ) -> np.ndarray:
        """
        Sample from Student's t copula.

        Like Gaussian but with heavier tails - extreme values occur together.
        Uses multivariate t distribution.
        """
        mean = np.zeros(n_dims)

        # Sample from multivariate normal
        z_samples = self.rng.multivariate_normal(mean, corr_matrix, size=count)

        # Sample chi-squared for t distribution
        chi2_samples = self.rng.chisquare(df, size=count)

        # Scale to get t distribution
        t_samples = z_samples / np.sqrt(chi2_samples[:, np.newaxis] / df)

        # Transform through t CDF
        return np.asarray(stats.t.cdf(t_samples, df=df))

    def _sample_clayton_copula(
        self, corr_matrix: np.ndarray, count: int, n_dims: int
    ) -> np.ndarray:
        """
        Sample from a Clayton copula (lower tail dependence).

        Clayton has lower tail dependence - variables take extreme low values
        together (crash together). Sampled exactly via the Marshall-Olkin
        frailty method with a Gamma mixing variable, giving uniform marginals
        and true lower-tail dependence lambda_L = 2^(-1/theta).

        Single-parameter Archimedean copulas are exchangeable, so the average
        (positive) correlation is used as Kendall's tau to set theta. Negative
        average correlation is not representable and falls back to Gaussian.
        """
        avg_corr = self._average_offdiagonal(corr_matrix, n_dims)
        if avg_corr <= 1e-3:
            logger.warning(
                "Clayton copula models positive dependence only; average "
                "correlation %.3f is non-positive, falling back to Gaussian copula.",
                avg_corr,
            )
            return self._sample_gaussian_copula(corr_matrix, count, n_dims)

        tau = min(avg_corr, 0.99)
        theta = 2 * tau / (1 - tau)  # Kendall's tau -> Clayton theta

        # Marshall-Olkin frailty: V ~ Gamma(1/theta, 1), E_i ~ Exp(1)
        v = self.rng.gamma(1.0 / theta, 1.0, size=count)
        e = self.rng.exponential(1.0, size=(count, n_dims))
        return (1.0 + e / v[:, np.newaxis]) ** (-1.0 / theta)

    def _sample_gumbel_copula(
        self, corr_matrix: np.ndarray, count: int, n_dims: int
    ) -> np.ndarray:
        """
        Sample from a Gumbel copula (upper tail dependence).

        Gumbel has upper tail dependence - variables take extreme high values
        together (boom together). Sampled exactly via the Marshall-Olkin
        frailty method with a positive-stable mixing variable, giving uniform
        marginals and true upper-tail dependence lambda_U = 2 - 2^(1/theta).

        Exchangeable: average (positive) correlation is used as Kendall's tau.
        Negative average correlation is not representable and falls back to
        Gaussian.
        """
        avg_corr = self._average_offdiagonal(corr_matrix, n_dims)
        if avg_corr <= 1e-3:
            logger.warning(
                "Gumbel copula models positive dependence only; average "
                "correlation %.3f is non-positive, falling back to Gaussian copula.",
                avg_corr,
            )
            return self._sample_gaussian_copula(corr_matrix, count, n_dims)

        tau = min(avg_corr, 0.99)
        theta = 1.0 / (1.0 - tau)  # Kendall's tau -> Gumbel theta (>= 1)
        alpha = 1.0 / theta  # stable index in (0, 1)

        # Marshall-Olkin frailty: V ~ positive-stable(alpha), E_i ~ Exp(1)
        v = self._sample_positive_stable(alpha, count)
        e = self.rng.exponential(1.0, size=(count, n_dims))
        return np.asarray(np.exp(-((e / v[:, np.newaxis]) ** alpha)))

    def _sample_frank_copula(
        self, corr_matrix: np.ndarray, count: int, n_dims: int
    ) -> np.ndarray:
        """
        Sample from a Frank copula (symmetric, no tail dependence).

        Sampled exactly via the Marshall-Olkin frailty method with a
        log-series mixing variable, giving uniform marginals. Good for weak to
        moderate association without extreme co-movements.

        The frailty construction requires theta > 0 (positive association).
        Near-zero average correlation yields independence; negative average
        correlation falls back to a Gaussian copula (which, like Frank, has no
        tail dependence and does honour negative correlations).
        """
        avg_corr = self._average_offdiagonal(corr_matrix, n_dims)
        if abs(avg_corr) <= 1e-3:
            return self.rng.uniform(0.0, 1.0, size=(count, n_dims))
        if avg_corr < 0:
            logger.warning(
                "Frank frailty sampling requires positive association; average "
                "correlation %.3f is negative, falling back to Gaussian copula.",
                avg_corr,
            )
            return self._sample_gaussian_copula(corr_matrix, count, n_dims)

        tau = min(avg_corr, 0.99)
        theta = self._frank_theta_from_tau(tau)

        # Marshall-Olkin frailty: V ~ LogSeries(1 - e^-theta), E_i ~ Exp(1)
        p = 1.0 - np.exp(-theta)
        v = self.rng.logseries(p, size=count)
        e = self.rng.exponential(1.0, size=(count, n_dims))
        # U_i = phi(E_i / V) with phi(t) = -1/theta * log(1 - (1 - e^-theta) e^-t)
        return np.asarray(-1.0 / theta * np.log1p(-p * np.exp(-e / v[:, np.newaxis])))

    def _average_offdiagonal(self, corr_matrix: np.ndarray, n_dims: int) -> float:
        """Average of the off-diagonal correlation entries (signed)."""
        if n_dims < 2:
            return 0.0
        off_sum = corr_matrix.sum() - np.trace(corr_matrix)
        return float(off_sum / (n_dims * (n_dims - 1)))

    def _sample_positive_stable(self, alpha: float, count: int) -> np.ndarray:
        """
        Sample a positive stable variable with Laplace transform exp(-t^alpha).

        Uses the standard Chambers-Mallows-Stuck representation. alpha in (0, 1);
        alpha -> 1 degenerates to the constant 1 (Gumbel independence limit).
        """
        if alpha >= 1.0:
            return np.ones(count)
        u = self.rng.uniform(0.0, np.pi, size=count)
        w = self.rng.exponential(1.0, size=count)
        term1 = np.sin(alpha * u) / np.power(np.sin(u), 1.0 / alpha)
        term2 = np.power(np.sin((1.0 - alpha) * u) / w, (1.0 - alpha) / alpha)
        return np.asarray(term1 * term2)

    def _frank_theta_from_tau(self, tau: float) -> float:
        """
        Invert Kendall's tau -> Frank theta numerically (theta > 0).

        tau(theta) = 1 + 4/theta * (D_1(theta) - 1), where D_1 is the first
        Debye function. Solved with Brent's method; falls back to a linear
        approximation if the solve fails.
        """

        def integrand(t: float) -> float:
            # t/(e^t - 1); limit is 1 at t->0 and ~0 for large t (guard overflow)
            if t <= 0.0:
                return 1.0
            if t > 700.0:
                return 0.0
            return float(t / np.expm1(t))

        def debye1(theta: float) -> float:
            value, _ = integrate.quad(integrand, 0.0, theta)
            return float(value / theta)

        def tau_of(theta: float) -> float:
            return 1.0 + 4.0 / theta * (debye1(theta) - 1.0)

        try:
            return float(optimize.brentq(lambda th: tau_of(th) - tau, 1e-6, 745.0))
        except Exception:
            return max(tau * 10.0, 0.1)

    def _ensure_positive_semidefinite(
        self, matrix: np.ndarray, epsilon: float = 1e-6
    ) -> np.ndarray:
        """Ensure a correlation matrix is positive semi-definite."""
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        eigenvalues = np.maximum(eigenvalues, epsilon)
        result = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        d = np.sqrt(np.diag(result))
        result = result / np.outer(d, d)
        return np.asarray(result)

    def _convert_numpy_value(
        self,
        value: Any,
        target_type: type = float,
        constraints: dict[str, float | None] | None = None,
    ) -> Any:
        """Convert numpy types to Python native types, respecting target type and constraints.

        Args:
            value: The numpy value to convert
            target_type: The desired Python type (int, float, str, etc.)
            constraints: Optional dict with 'ge', 'le', 'gt', 'lt' bounds for clipping
        """
        # First convert to Python native type
        result: Any
        if isinstance(value, np.integer):
            result = int(value)
        elif isinstance(value, np.floating):
            result = float(value)
        elif isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, np.str_):
            return str(value)
        else:
            result = value

        # Apply constraints (clipping) for numeric types
        if constraints and isinstance(result, (int, float)):
            # Apply lower bounds
            ge = constraints.get("ge")
            if ge is not None:
                result = max(result, ge)
            gt = constraints.get("gt")
            if gt is not None:
                # For gt, we need strictly greater, so use a tiny epsilon above
                if result <= gt:
                    result = gt + (1 if target_type is int else 1e-9)

            # Apply upper bounds
            le = constraints.get("le")
            if le is not None:
                result = min(result, le)
            lt = constraints.get("lt")
            if lt is not None:
                # For lt, we need strictly less, so use a tiny epsilon below
                if result >= lt:
                    result = lt - (1 if target_type is int else 1e-9)

        # Cast to target type after clipping
        if target_type is int and isinstance(result, float):
            return int(round(result))

        return result

    def sample_single_field(
        self,
        spec: DistributionSpec,
        count: int,
    ) -> list[Any]:
        """Sample a single field's values."""
        samples = spec.sample(count, self.rng)
        return [self._convert_numpy_value(v) for v in samples]
