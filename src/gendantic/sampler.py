"""
Distribution sampler for generating field values using numpy.

This module provides the DistributionSampler class which orchestrates
numpy-based sampling for fields that have distribution specifications.
Supports correlated sampling via multiple copula types.
"""

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from . import copula
from .distributions import (
    Conditional,
    Constraints,
    CopulaType,
    Correlations,
    DistributionSpec,
    Ordering,
    truncate,
)

logger = logging.getLogger("gendantic")

# Maximum number of resampling rounds for Ordering(method="resample") before
# giving up: each round redraws all still-violating records, so a compatible
# ordering converges geometrically and this bound is only hit when the fields'
# marginals overlap too heavily to order by rejection.
_MAX_RESAMPLE_ROUNDS = 1000


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

    def sample_fields(
        self,
        distribution_specs: (
            dict[str, DistributionSpec | Conditional]
            | dict[str, tuple[DistributionSpec | Conditional, type, dict[str, float | None]]]
        ),
        count: int,
        correlations: Correlations | None = None,
        constraints: Constraints | None = None,
    ) -> list[dict[str, Any]]:
        """
        Sample all distribution fields and return as list of partial records.

        If correlations are specified, uses appropriate copulas to preserve
        marginal distributions while enforcing the correlation/dependency structure.

        ``Conditional`` specs select a distribution per record based on the
        already-sampled value of their ``on`` field; they are resolved after the
        plain (and correlated) fields, in dependency order. ``constraints``
        (e.g. ``Ordering``) are enforced on the assembled columns afterwards.

        Args:
            distribution_specs: Either a plain ``{field: spec}`` map (types
                default to float, no constraints) or the fully-specified
                ``{field: (spec, target_type, constraints)}`` form produced by
                ``extract_distribution_specs_with_types``. A spec may be a
                ``DistributionSpec`` or a ``Conditional``.
            count: Number of records to sample
            correlations: Optional correlation structure
            constraints: Optional cross-field constraints (e.g. ``Ordering``)
        """
        if not distribution_specs:
            return [{} for _ in range(count)]

        # Normalise to (spec, type, constraints) format
        normalised_specs: dict[
            str, tuple[DistributionSpec | Conditional, type, dict[str, float | None]]
        ] = {}

        for field_name, spec_or_tuple in distribution_specs.items():
            if isinstance(spec_or_tuple, tuple):
                if len(spec_or_tuple) != 3:
                    raise ValueError(
                        f"Unexpected tuple length for {field_name}: {len(spec_or_tuple)}"
                    )
                normalised_specs[field_name] = spec_or_tuple
            else:
                # Plain spec without type/constraints - default to float, unclipped
                normalised_specs[field_name] = (
                    spec_or_tuple,
                    float,
                    {"ge": None, "le": None, "gt": None, "lt": None},
                )

        # Split conditional fields out; plain fields are sampled first so their
        # values are available as discriminators.
        plain_specs: dict[
            str, tuple[DistributionSpec, type, dict[str, float | None]]
        ] = {}
        conditional_specs: dict[str, Conditional] = {}
        for name, (spec, target_type, field_constraints) in normalised_specs.items():
            if isinstance(spec, Conditional):
                conditional_specs[name] = spec
            else:
                # Truncate to the field's numeric bounds at the source (proper
                # inverse-CDF truncation) rather than clamping draws afterwards.
                # The per-record clamp in _convert_numpy_value stays as a cheap
                # safety net (e.g. against int-rounding at the boundary).
                effective = truncate(spec, target_type, field_constraints)
                plain_specs[name] = (effective, target_type, field_constraints)

        if correlations is None or len(correlations) == 0:
            samples = self._columns_independent(plain_specs, count)
        else:
            conflict = correlations.get_fields() & set(conditional_specs)
            if conflict:
                logger.warning(
                    "Correlation spec references conditional field(s) %s; conditional "
                    "fields are sampled per-group and are not correlated.",
                    ", ".join(sorted(conflict)),
                )
            samples = self._columns_correlated(plain_specs, count, correlations)

        if conditional_specs:
            self._resolve_conditionals(
                conditional_specs, normalised_specs, samples, count
            )

        if constraints is not None:
            correlated_fields = (
                correlations.get_fields() if correlations is not None else set()
            )
            self._apply_constraints(
                constraints,
                samples,
                normalised_specs,
                set(conditional_specs),
                correlated_fields,
            )

        return [
            {
                name: self._convert_numpy_value(
                    samples[name][i], target_type, field_constraints
                )
                for name, (_, target_type, field_constraints) in normalised_specs.items()
            }
            for i in range(count)
        ]

    def _columns_independent(
        self,
        distribution_specs: dict[
            str, tuple[DistributionSpec, type, dict[str, float | None]]
        ],
        count: int,
    ) -> dict[str, NDArray[Any]]:
        """Sample fields independently (no correlation structure), as columns.

        Iterating in insertion order keeps the RNG draw sequence deterministic.
        """
        return {
            field_name: spec.sample(count, self.rng)
            for field_name, (spec, _, _) in distribution_specs.items()
        }

    def _resolve_conditionals(
        self,
        conditional_specs: dict[str, Conditional],
        normalised_specs: dict[
            str, tuple[DistributionSpec | Conditional, type, dict[str, float | None]]
        ],
        samples: dict[str, NDArray[Any]],
        count: int,
    ) -> None:
        """Sample conditional fields into ``samples``, in dependency order.

        A conditional field's ``on`` discriminator must be another
        distribution-sampled field (a plain field already in ``samples`` or
        another conditional). Conditionals are resolved once their discriminator
        is available; a remaining set that can make no progress is a cycle.

        The discriminator is matched on its *converted* value (the value that
        will appear in the record), so numeric bins line up with what the user
        sees after int rounding / constraint clipping.
        """
        known = set(samples.keys())
        all_fields = known | set(conditional_specs)

        for name, cond in conditional_specs.items():
            if cond.on == name:
                raise ValueError(f"Conditional field {name!r} cannot depend on itself")
            if cond.on not in all_fields:
                raise ValueError(
                    f"Conditional field {name!r} depends on {cond.on!r}, which is not "
                    f"a distribution-sampled field; conditional discriminators must "
                    f"themselves be sampled fields."
                )

        remaining = dict(conditional_specs)
        while remaining:
            ready = [name for name, cond in remaining.items() if cond.on in known]
            if not ready:
                cycle = ", ".join(sorted(remaining))
                raise ValueError(
                    f"Circular dependency among conditional fields: {cycle}"
                )
            for name in ready:
                cond = remaining.pop(name)
                _, disc_type, disc_constraints = normalised_specs[cond.on]
                discriminator = [
                    self._convert_numpy_value(
                        samples[cond.on][i], disc_type, disc_constraints
                    )
                    for i in range(count)
                ]
                _, target_type, field_constraints = normalised_specs[name]
                samples[name] = self._sample_conditional_column(
                    cond, discriminator, count, target_type, field_constraints
                )
                known.add(name)

    def _sample_conditional_column(
        self,
        cond: Conditional,
        discriminator: list[Any],
        count: int,
        target_type: type = float,
        field_constraints: dict[str, float | None] | None = None,
    ) -> NDArray[Any]:
        """Sample a conditional field by grouping records that share a spec.

        Each record picks a distribution via ``cond.spec_for`` based on its
        (converted) discriminator value; records mapping to the same spec are
        sampled together as one vectorized batch, then scattered back to their
        positions. Group order follows first appearance, keeping the RNG
        sequence deterministic. Each group's spec is truncated to the field's
        numeric bounds, matching the plain-field path.
        """
        constraints = field_constraints or {"ge": None, "le": None, "gt": None, "lt": None}
        result: NDArray[Any] = np.empty(count, dtype=object)
        group_indices: dict[int, list[int]] = {}
        group_specs: dict[int, DistributionSpec] = {}
        for i in range(count):
            spec = cond.spec_for(discriminator[i])
            group_indices.setdefault(id(spec), []).append(i)
            group_specs[id(spec)] = spec

        for spec_id, indices in group_indices.items():
            idx = np.asarray(indices)
            effective = truncate(group_specs[spec_id], target_type, constraints)
            result[idx] = effective.sample(len(indices), self.rng)
        return result

    def _apply_constraints(
        self,
        constraints: Constraints,
        samples: dict[str, NDArray[Any]],
        normalised_specs: dict[
            str, tuple[DistributionSpec | Conditional, type, dict[str, float | None]]
        ],
        conditional_fields: set[str],
        correlated_fields: set[str],
    ) -> None:
        """Enforce cross-field constraints on the sampled columns in place.

        Each ``Ordering`` is enforced with its chosen ``method``: ``"sort"``
        sorts each record's values across the constrained fields (guaranteed,
        but reshapes marginals into order statistics); ``"resample"`` redraws
        only the violating records to preserve each field's marginal.
        """
        for ordering in constraints.orderings():
            missing = [f for f in ordering.fields if f not in samples]
            if missing:
                raise ValueError(
                    f"Ordering references field(s) not sampled via distributions: "
                    f"{', '.join(missing)}"
                )
            if ordering.method == "resample":
                self._resample_ordering(
                    ordering,
                    samples,
                    normalised_specs,
                    conditional_fields,
                    correlated_fields,
                )
            else:
                stacked = np.array(
                    [np.asarray(samples[f], dtype=float) for f in ordering.fields]
                )
                ordered = np.sort(stacked, axis=0)
                for position, field_name in enumerate(ordering.fields):
                    samples[field_name] = ordered[position]

    def _resample_ordering(
        self,
        ordering: Ordering,
        samples: dict[str, NDArray[Any]],
        normalised_specs: dict[
            str, tuple[DistributionSpec | Conditional, type, dict[str, float | None]]
        ],
        conditional_fields: set[str],
        correlated_fields: set[str],
    ) -> None:
        """Enforce an ordering by redrawing only the records that violate it.

        Each field keeps its own marginal: violating records are resampled from
        the fields' distributions until the whole batch complies. Ordering is
        checked on the *converted* values (the ones the record exposes), so the
        guarantee survives int rounding and constraint clipping. Raises if a
        constrained field is not an independent plain distribution, or if the
        rejection budget is exhausted because the marginals overlap too heavily.
        """
        fields = ordering.fields
        for field_name in fields:
            if field_name in conditional_fields:
                raise ValueError(
                    f"Ordering(method='resample') cannot include conditional field "
                    f"{field_name!r}; use method='sort' or make the field independent."
                )
            if field_name in correlated_fields:
                raise ValueError(
                    f"Ordering(method='resample') cannot include correlated field "
                    f"{field_name!r}; use method='sort' or drop it from __correlations__."
                )

        field_specs: dict[str, DistributionSpec] = {}
        for field_name in fields:
            spec = normalised_specs[field_name][0]
            if not isinstance(spec, DistributionSpec):  # pragma: no cover - guarded above
                raise ValueError(
                    f"Ordering(method='resample') field {field_name!r} has no "
                    f"independent distribution to resample from."
                )
            field_specs[field_name] = spec

        cols = {f: np.asarray(samples[f], dtype=float).copy() for f in fields}

        def violating_mask() -> NDArray[np.bool_]:
            converted = [
                self._converted_column(cols[f], *normalised_specs[f][1:]) for f in fields
            ]
            mask = np.zeros(len(converted[0]), dtype=bool)
            for lower, upper in zip(converted[:-1], converted[1:], strict=True):
                mask |= lower > upper
            return mask

        mask = violating_mask()
        rounds = 0
        while mask.any() and rounds < _MAX_RESAMPLE_ROUNDS:
            n = int(mask.sum())
            for field_name in fields:
                redrawn = np.asarray(
                    field_specs[field_name].sample(n, self.rng), dtype=float
                )
                cols[field_name][mask] = redrawn
            mask = violating_mask()
            rounds += 1

        if mask.any():
            rate = float(mask.mean())
            raise ValueError(
                f"Ordering(method='resample') on {fields} could not be satisfied: "
                f"{rate:.1%} of records still violate the order after "
                f"{_MAX_RESAMPLE_ROUNDS} resampling rounds. The fields' marginals "
                f"overlap too heavily to order by rejection; separate the marginals "
                f"or use method='sort'."
            )

        for field_name in fields:
            samples[field_name] = cols[field_name]

    def _converted_column(
        self,
        column: NDArray[Any],
        target_type: type,
        constraints: dict[str, float | None],
    ) -> NDArray[Any]:
        """Convert a whole column to the values the record will expose, as floats.

        Mirrors per-record conversion (int rounding + ge/le/gt/lt clipping) so
        the ordering check matches what the caller ultimately sees.
        """
        return np.array(
            [
                float(self._convert_numpy_value(v, target_type, constraints))
                for v in column
            ]
        )

    def _columns_correlated(
        self,
        distribution_specs: dict[
            str, tuple[DistributionSpec, type, dict[str, float | None]]
        ],
        count: int,
        correlations: Correlations,
    ) -> dict[str, NDArray[Any]]:
        """
        Sample fields with correlation structure using copulas, as columns.

        Fields whose distribution does not support correlation (e.g.
        Categorical) are sampled independently; only correlation-capable
        fields are drawn through the copula. Correlation specs that reference
        an unsupported field are ignored with a warning.

        Routing honours each pair's copula: a Gaussian-only (or Student-t-only)
        structure uses the exact full correlation matrix, while any mix of
        families -- or any Archimedean family -- is sampled per-pair through a
        vine (see :mod:`gendantic.copula`), so each pair keeps its own family
        and strength rather than collapsing to one dominant copula.
        """
        samples: dict[str, NDArray[Any]] = {}

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
        samples.update(self._sample_marginals(distribution_specs, indep_fields, count))

        # Determine whether any real correlations remain among capable fields
        corr_matrix = correlations.build_correlation_matrix(corr_fields)
        has_correlation = len(corr_fields) >= 2 and not np.allclose(
            corr_matrix, np.eye(len(corr_fields))
        )

        # Pairs among capable fields (drop specs touching excluded fields and
        # zero-correlation specs, which contribute no dependence).
        capable = set(corr_fields)
        pairs = [
            (f1, f2, corr, cop)
            for f1, f2, corr, cop in correlations
            if f1 in capable and f2 in capable and corr != 0.0
        ]

        if not has_correlation or not pairs:
            # Nothing left to correlate - sample capable fields independently too
            samples.update(
                self._sample_marginals(distribution_specs, corr_fields, count)
            )
            return samples

        families = {cop for _, _, _, cop in pairs}
        if families == {CopulaType.GAUSSIAN}:
            # Exact: the full matrix honours every Gaussian pair jointly.
            samples.update(
                self._sample_full_matrix(
                    CopulaType.GAUSSIAN, distribution_specs, corr_fields, corr_matrix, count
                )
            )
        elif families == {CopulaType.STUDENT_T}:
            samples.update(
                self._sample_full_matrix(
                    CopulaType.STUDENT_T, distribution_specs, corr_fields, corr_matrix, count
                )
            )
        else:
            # Mixed families or any Archimedean: honour each pair via a vine.
            samples.update(
                self._sample_vine(distribution_specs, corr_fields, pairs, count)
            )

        return samples

    def _sample_full_matrix(
        self,
        copula_type: str,
        distribution_specs: dict[
            str, tuple[DistributionSpec, type, dict[str, float | None]]
        ],
        corr_fields: list[str],
        corr_matrix: NDArray[Any],
        count: int,
    ) -> dict[str, NDArray[Any]]:
        """Sample Gaussian/Student-t fields jointly from the full correlation matrix.

        Falls back to independent marginals if the copula draw fails (e.g. a
        matrix that cannot be repaired to positive semidefinite).
        """
        corr_matrix = self._ensure_positive_semidefinite(corr_matrix)
        try:
            u_samples = self._sample_copula(
                copula_type, corr_matrix, count, len(corr_fields)
            )
        except Exception:
            return self._sample_marginals(distribution_specs, corr_fields, count)
        samples: dict[str, NDArray[Any]] = {}
        for i, name in enumerate(corr_fields):
            u_col = np.clip(u_samples[:, i], 1e-10, 1 - 1e-10)
            samples[name] = distribution_specs[name][0].quantile(u_col)
        return samples

    def _sample_vine(
        self,
        distribution_specs: dict[
            str, tuple[DistributionSpec, type, dict[str, float | None]]
        ],
        corr_fields: list[str],
        pairs: list[tuple[str, str, float, str]],
        count: int,
    ) -> dict[str, NDArray[Any]]:
        """Sample per-pair copulas as a 1-truncated vine (Markov-tree) structure.

        Each connected component of the specified pairs is one tree: the root is
        drawn uniformly, then each child is drawn conditionally on its parent via
        the pair's inverse h-function. Capable fields in no pair are sampled from
        their marginal. All RNG draws happen here in a fixed order so the result
        stays reproducible for a given seed.
        """
        trees = copula.build_vine(pairs)
        u_cols: dict[str, NDArray[Any]] = {}
        for tree in trees:
            u_cols[tree.root] = self.rng.uniform(size=count)
            for edge in tree.edges:
                w = self.rng.uniform(size=count)
                u_cols[edge.child] = copula.hinv(
                    edge.family, w, u_cols[edge.parent], edge.param
                )

        samples: dict[str, NDArray[Any]] = {}
        for name in corr_fields:
            if name in u_cols:
                u_col = np.clip(u_cols[name], 1e-10, 1 - 1e-10)
                samples[name] = distribution_specs[name][0].quantile(u_col)
        unpaired = [name for name in corr_fields if name not in u_cols]
        samples.update(self._sample_marginals(distribution_specs, unpaired, count))
        return samples

    def _sample_marginals(
        self,
        distribution_specs: dict[
            str, tuple[DistributionSpec, type, dict[str, float | None]]
        ],
        names: list[str],
        count: int,
    ) -> dict[str, NDArray[Any]]:
        """Sample the named fields directly from their marginals (uncorrelated).

        Iterating ``names`` in order keeps the RNG draw sequence deterministic.
        """
        return {
            name: distribution_specs[name][0].sample(count, self.rng)
            for name in names
        }

    def _sample_copula(
        self,
        copula_type: str,
        corr_matrix: NDArray[Any],
        count: int,
        n_dims: int,
    ) -> NDArray[Any]:
        """
        Sample from a copula, returning uniform [0,1] values.

        Returns array of shape (count, n_dims) with values in [0, 1].
        """
        if copula_type == CopulaType.STUDENT_T:
            return self._sample_student_t_copula(corr_matrix, count, n_dims, df=4)
        # Gaussian is the only other full-matrix copula; Archimedean and mixed
        # families are handled per-pair by the vine (see _sample_vine).
        return self._sample_gaussian_copula(corr_matrix, count, n_dims)

    def _sample_gaussian_copula(
        self, corr_matrix: NDArray[Any], count: int, n_dims: int
    ) -> NDArray[Any]:
        """
        Sample from Gaussian copula.

        Standard approach: multivariate normal -> standard normal CDF.
        """
        mean = np.zeros(n_dims)
        z_samples = self.rng.multivariate_normal(mean, corr_matrix, size=count)
        return np.asarray(stats.norm.cdf(z_samples))

    def _sample_student_t_copula(
        self, corr_matrix: NDArray[Any], count: int, n_dims: int, df: int = 4
    ) -> NDArray[Any]:
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

    def _ensure_positive_semidefinite(
        self, matrix: NDArray[Any], epsilon: float = 1e-6
    ) -> NDArray[Any]:
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
