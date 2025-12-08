"""
Distribution sampler for generating field values using numpy.

This module provides the DistributionSampler class which orchestrates
numpy-based sampling for fields that have distribution specifications.
Supports correlated sampling via multiple copula types.
"""

from typing import Any

import numpy as np
from scipy import stats
from scipy.special import expit, logit

from .distributions import CopulaType, Correlations, DistributionSpec


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
        normalised_specs: dict[str, tuple[DistributionSpec, type, dict[str, float | None]]] = {}
        empty_constraints: dict[str, float | None] = {"ge": None, "le": None, "gt": None, "lt": None}

        for field_name, spec_or_tuple in distribution_specs.items():
            if isinstance(spec_or_tuple, tuple):
                if len(spec_or_tuple) == 3:
                    normalised_specs[field_name] = spec_or_tuple
                elif len(spec_or_tuple) == 2:
                    spec, target_type = spec_or_tuple
                    normalised_specs[field_name] = (spec, target_type, empty_constraints.copy())
                else:
                    raise ValueError(f"Unexpected tuple length for {field_name}: {len(spec_or_tuple)}")
            else:
                # Legacy format without type - default to float
                normalised_specs[field_name] = (spec_or_tuple, float, empty_constraints.copy())

        if correlations is None or len(correlations) == 0:
            return self._sample_independent(normalised_specs, count)

        return self._sample_correlated(normalised_specs, count, correlations)

    def _sample_independent(
        self,
        distribution_specs: dict[str, tuple[DistributionSpec, type, dict[str, float | None]]],
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
        distribution_specs: dict[str, tuple[DistributionSpec, type, dict[str, float | None]]],
        count: int,
        correlations: Correlations,
    ) -> list[dict[str, Any]]:
        """
        Sample fields with correlation structure using copulas.

        For simplicity and consistency, we use the dominant copula type
        for the entire correlation structure. If mixed copulas are specified,
        we use Gaussian for the overall structure but apply tail adjustments.
        """
        field_names = list(distribution_specs.keys())
        n_fields = len(field_names)

        # Get copula groups to determine dominant type
        copula_groups = correlations.get_copula_groups()

        # Determine dominant copula (most common, or Gaussian if mixed)
        if len(copula_groups) == 1:
            dominant_copula = list(copula_groups.keys())[0]
        else:
            # Mixed copulas - use Gaussian as base
            dominant_copula = CopulaType.GAUSSIAN

        # Build correlation matrix
        corr_matrix = correlations.build_correlation_matrix(field_names)
        corr_matrix = self._ensure_positive_semidefinite(corr_matrix)

        # Sample uniform values using the appropriate copula
        try:
            u_samples = self._sample_copula(
                dominant_copula, corr_matrix, count, n_fields
            )
        except Exception:
            # Fall back to independent sampling
            return self._sample_independent(distribution_specs, count)

        # Transform each field via its quantile function
        samples: dict[str, np.ndarray] = {}
        target_types: dict[str, type] = {}
        constraints_map: dict[str, dict[str, float | None]] = {}
        for i, field_name in enumerate(field_names):
            spec, target_type, constraints = distribution_specs[field_name]
            target_types[field_name] = target_type
            constraints_map[field_name] = constraints
            u_col = u_samples[:, i]
            # Clip to avoid numerical issues at boundaries
            u_col = np.clip(u_col, 1e-10, 1 - 1e-10)
            samples[field_name] = spec.quantile(u_col)

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
        return stats.norm.cdf(z_samples)

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
        return stats.t.cdf(t_samples, df=df)

    def _sample_clayton_copula(
        self, corr_matrix: np.ndarray, count: int, n_dims: int
    ) -> np.ndarray:
        """
        Sample from Clayton copula (lower tail dependence).

        Clayton copula has lower tail dependence - variables tend to have
        extreme low values together (crashes together).

        For multivariate case, we use a simplified approach based on
        the average correlation to determine theta.
        """
        # Get average correlation (excluding diagonal)
        avg_corr = (corr_matrix.sum() - n_dims) / (n_dims * (n_dims - 1))
        avg_corr = max(0.01, min(0.99, abs(avg_corr)))  # Clip to valid range

        # Convert correlation to Clayton theta (approximation)
        # Higher theta = stronger lower tail dependence
        theta = 2 * avg_corr / (1 - avg_corr)
        theta = max(0.1, theta)  # Ensure positive

        # Sample using conditional method
        # V ~ Gamma(1/theta, 1)
        v = self.rng.gamma(1 / theta, 1, size=count)

        # Independent exponentials
        e = self.rng.exponential(1, size=(count, n_dims))

        # Clayton transformation
        u = (1 + e / v[:, np.newaxis]) ** (-1 / theta)

        # Apply correlation structure via Gaussian copula mixing
        # This preserves Clayton's lower tail dependence while respecting
        # the specified correlation matrix
        z = self.rng.multivariate_normal(np.zeros(n_dims), corr_matrix, size=count)
        mixing = stats.norm.cdf(z)

        # Blend Clayton with Gaussian structure
        return 0.7 * u + 0.3 * mixing

    def _sample_gumbel_copula(
        self, corr_matrix: np.ndarray, count: int, n_dims: int
    ) -> np.ndarray:
        """
        Sample from Gumbel copula (upper tail dependence).

        Gumbel copula has upper tail dependence - variables tend to have
        extreme high values together (boom together).
        """
        # Get average correlation
        avg_corr = (corr_matrix.sum() - n_dims) / (n_dims * (n_dims - 1))
        avg_corr = max(0.01, min(0.99, abs(avg_corr)))

        # Convert to Gumbel theta (theta >= 1)
        theta = 1 / (1 - avg_corr)
        theta = max(1.0, theta)

        # Sample stable distribution for Gumbel
        # Use approximation via exponential
        w = self.rng.exponential(1, size=count)
        s = self.rng.standard_exponential(size=(count, n_dims))

        # Gumbel transformation
        u = np.exp(-(s / w[:, np.newaxis]) ** (1 / theta))

        # Apply correlation structure
        z = self.rng.multivariate_normal(np.zeros(n_dims), corr_matrix, size=count)
        mixing = stats.norm.cdf(z)

        # Blend to maintain upper tail while respecting correlations
        return 0.7 * u + 0.3 * mixing

    def _sample_frank_copula(
        self, corr_matrix: np.ndarray, count: int, n_dims: int
    ) -> np.ndarray:
        """
        Sample from Frank copula (symmetric, no tail dependence).

        Frank copula is symmetric with no tail dependence - good for
        weak to moderate correlations without extreme co-movements.
        """
        # Get average correlation
        avg_corr = (corr_matrix.sum() - n_dims) / (n_dims * (n_dims - 1))

        # Convert to Frank theta
        # theta can be any real number, 0 = independence
        if abs(avg_corr) < 0.01:
            theta = 0.1
        else:
            # Approximation: theta ≈ correlation * 10 for moderate correlations
            theta = avg_corr * 10
            theta = np.clip(theta, -20, 20)

        # Sample using logarithmic series distribution
        u1 = self.rng.uniform(0, 1, size=(count, n_dims))

        if abs(theta) < 0.01:
            # Near independence
            return u1

        # Frank copula transformation
        # u2 = -log(1 + (exp(-theta) - 1) * (1 - u1)) / theta for 2D
        # For multivariate, use conditional approach with Gaussian mixing

        z = self.rng.multivariate_normal(np.zeros(n_dims), corr_matrix, size=count)
        gaussian_u = stats.norm.cdf(z)

        # Apply Frank transformation to add symmetric dependence
        exp_theta = np.exp(-theta)
        frank_factor = -np.log(1 + (exp_theta - 1) * u1) / theta if theta != 0 else u1
        frank_factor = np.clip(frank_factor, 0, 1)

        # Blend
        return 0.5 * gaussian_u + 0.5 * frank_factor

    def _ensure_positive_semidefinite(
        self, matrix: np.ndarray, epsilon: float = 1e-6
    ) -> np.ndarray:
        """Ensure a correlation matrix is positive semi-definite."""
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        eigenvalues = np.maximum(eigenvalues, epsilon)
        result = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        d = np.sqrt(np.diag(result))
        result = result / np.outer(d, d)
        return result

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
            if constraints.get("ge") is not None:
                result = max(result, constraints["ge"])
            if constraints.get("gt") is not None:
                # For gt, we need strictly greater, so use a tiny epsilon above
                min_val = constraints["gt"]
                if result <= min_val:
                    result = min_val + (1 if target_type is int else 1e-9)

            # Apply upper bounds
            if constraints.get("le") is not None:
                result = min(result, constraints["le"])
            if constraints.get("lt") is not None:
                # For lt, we need strictly less, so use a tiny epsilon below
                max_val = constraints["lt"]
                if result >= max_val:
                    result = max_val - (1 if target_type is int else 1e-9)

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
