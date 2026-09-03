"""Statistical fidelity checking for generated data.

gendantic's promise is that distribution-annotated fields are sampled to match
their specification. :func:`fidelity_report` makes that promise *checkable*: it
compares a batch of generated records against the model's declared
distributions and correlations and returns a structured, printable report.

The report is descriptive, not enforcing - it never raises. Callers inspect
:attr:`FidelityReport.passed` (or the per-field / per-correlation results) and
decide what to do.

Tests used per field:
    - continuous (normal, uniform, lognormal, exponential, beta):
      Kolmogorov-Smirnov against the distribution's CDF.
    - discrete counts (poisson, binomial): chi-square goodness-of-fit on the
      count histogram (small-expectation bins are merged).
    - categorical: chi-square goodness-of-fit on category frequencies.

Correlations are checked with Spearman's rank correlation (what copulas
actually control) as the verdict, with Pearson reported alongside.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel
from scipy import stats

from .distributions import Correlations, DistributionSpec
from .llm_driven_analyser import LLMDrivenModelAnalyser

_CONTINUOUS = {"normal", "uniform", "lognormal", "exponential", "beta"}
_DISCRETE = {"poisson", "binomial"}


@dataclass
class FieldFidelity:
    """Goodness-of-fit result for one distribution-annotated field."""

    field: str
    distribution: str
    test: str  # "ks" | "chi2"
    statistic: float
    p_value: float
    passed: bool
    observed_mean: float | None = None
    expected_mean: float | None = None


@dataclass
class CorrelationFidelity:
    """How closely one declared correlation was reproduced."""

    field1: str
    field2: str
    target: float
    observed_spearman: float
    observed_pearson: float
    error: float  # |observed_spearman - target|
    passed: bool


@dataclass
class FidelityReport:
    """Aggregate fidelity of a batch of generated records to its model spec."""

    fields: list[FieldFidelity] = field(default_factory=list)
    correlations: list[CorrelationFidelity] = field(default_factory=list)
    sample_size: int = 0

    @property
    def passed(self) -> bool:
        """True when every checked field and correlation passed."""
        return all(f.passed for f in self.fields) and all(
            c.passed for c in self.correlations
        )

    def summary(self) -> str:
        """Render a human-readable table of the results."""
        lines = [
            f"Fidelity report ({self.sample_size} records) - "
            f"{'PASS' if self.passed else 'FAIL'}",
        ]

        if self.fields:
            lines.append("")
            lines.append("Fields:")
            for f in self.fields:
                mark = "ok " if f.passed else "XX "
                detail = f"{f.test}={f.statistic:.4f} p={f.p_value:.4f}"
                if f.observed_mean is not None and f.expected_mean is not None:
                    detail += (
                        f" mean obs={f.observed_mean:.4g}"
                        f"/exp={f.expected_mean:.4g}"
                    )
                lines.append(f"  [{mark}] {f.field} ({f.distribution}): {detail}")

        if self.correlations:
            lines.append("")
            lines.append("Correlations:")
            for c in self.correlations:
                mark = "ok " if c.passed else "XX "
                lines.append(
                    f"  [{mark}] {c.field1}~{c.field2}: target={c.target:+.2f} "
                    f"spearman={c.observed_spearman:+.2f} "
                    f"pearson={c.observed_pearson:+.2f} err={c.error:.2f}"
                )

        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def fidelity_report(
    records: Sequence[BaseModel] | Sequence[dict[str, Any]],
    model_class: type[BaseModel],
    *,
    alpha: float = 0.05,
    correlation_tolerance: float = 0.15,
) -> FidelityReport:
    """Check how well generated ``records`` match ``model_class``'s spec.

    Only distribution-annotated fields and declared ``__correlations__`` are
    checked; free-text (LLM-generated) fields are ignored, as there is no
    specification to compare them against.

    Args:
        records: Generated model instances or plain dicts.
        model_class: The model the records were generated for.
        alpha: Significance level for goodness-of-fit tests. A field passes
            when its p-value is >= ``alpha`` (i.e. we fail to reject the null
            hypothesis that the samples came from the specified distribution).
        correlation_tolerance: Maximum absolute difference between the observed
            Spearman correlation and the declared target for a pair to pass.

    Returns:
        A :class:`FidelityReport`. Never raises on statistical failure.
    """
    # Conditional fields are sampled per-group; goodness-of-fit against a single
    # marginal does not apply, so they are skipped here (checked in a follow-up).
    specs = {
        name: spec
        for name, spec in LLMDrivenModelAnalyser.extract_distribution_specs(
            model_class
        ).items()
        if isinstance(spec, DistributionSpec)
    }
    rows = [_as_dict(r) for r in records]
    report = FidelityReport(sample_size=len(rows))

    if not rows:
        return report

    for field_name, spec in specs.items():
        if field_name not in rows[0]:
            continue
        column = [row[field_name] for row in rows]
        report.fields.append(_check_field(field_name, spec, column, alpha))

    correlations = getattr(model_class, "__correlations__", None)
    if isinstance(correlations, Correlations):
        for f1, f2, target, _copula in correlations:
            result = _check_correlation(
                f1, f2, target, specs, rows, correlation_tolerance
            )
            if result is not None:
                report.correlations.append(result)

    return report


def _as_dict(record: BaseModel | dict[str, Any]) -> dict[str, Any]:
    if isinstance(record, BaseModel):
        return dict(record)
    return record


def _check_field(
    field_name: str,
    spec: DistributionSpec,
    column: list[Any],
    alpha: float,
) -> FieldFidelity:
    dist_type = spec.distribution_type

    if dist_type in _CONTINUOUS:
        return _ks_field(field_name, spec, column, alpha)
    if dist_type in _DISCRETE:
        return _discrete_chi2_field(field_name, spec, column, alpha)
    # categorical (and any future nominal type)
    return _categorical_chi2_field(field_name, spec, column, alpha)


def _expected_mean(spec: DistributionSpec, n: int = 2000) -> float:
    """Numerical mean of the distribution via its quantile function."""
    grid = (np.arange(1, n + 1) - 0.5) / n
    return float(np.mean(spec.quantile(grid)))


def _ks_field(
    field_name: str,
    spec: DistributionSpec,
    column: list[Any],
    alpha: float,
) -> FieldFidelity:
    observed = np.asarray(column, dtype=float)
    cdf: Callable[[NDArray[Any]], NDArray[Any]] = spec.cdf
    result = stats.kstest(observed, cdf)
    statistic = float(result.statistic)
    p_value = float(result.pvalue)
    return FieldFidelity(
        field=field_name,
        distribution=spec.distribution_type,
        test="ks",
        statistic=statistic,
        p_value=p_value,
        passed=p_value >= alpha,
        observed_mean=float(np.mean(observed)),
        expected_mean=_expected_mean(spec),
    )


def _merge_small_bins(
    observed: NDArray[Any], expected: NDArray[Any], min_expected: float = 5.0
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Left-to-right merge of adjacent bins until each expected count is large.

    Chi-square is unreliable when expected counts are tiny; merging keeps the
    test valid. A trailing small bin is folded back into the previous one.
    """
    merged_obs: list[float] = []
    merged_exp: list[float] = []
    acc_obs = acc_exp = 0.0
    for obs, exp in zip(observed, expected, strict=True):
        acc_obs += obs
        acc_exp += exp
        if acc_exp >= min_expected:
            merged_obs.append(acc_obs)
            merged_exp.append(acc_exp)
            acc_obs = acc_exp = 0.0
    if acc_exp > 0:
        if merged_exp:
            merged_obs[-1] += acc_obs
            merged_exp[-1] += acc_exp
        else:
            merged_obs.append(acc_obs)
            merged_exp.append(acc_exp)
    return np.asarray(merged_obs), np.asarray(merged_exp)


def _discrete_chi2_field(
    field_name: str,
    spec: DistributionSpec,
    column: list[Any],
    alpha: float,
) -> FieldFidelity:
    observed = np.asarray(column, dtype=float).round().astype(int)
    n = len(observed)
    lo, hi = int(observed.min()), int(observed.max())
    support = np.arange(lo, hi + 1)

    # P(X = k) for each k in the observed range, via CDF differences, plus the
    # probability mass in the tails below lo and above hi.
    pmf = spec.cdf(support) - spec.cdf(support - 1)
    left_tail = float(spec.cdf(np.array([lo - 1]))[0])
    right_tail = 1.0 - float(spec.cdf(np.array([hi]))[0])
    probs = np.concatenate([[left_tail], pmf, [right_tail]])

    obs_counts = np.concatenate(
        [[0.0], [float(np.sum(observed == k)) for k in support], [0.0]]
    )
    exp_counts = probs * n

    obs_m, exp_m = _merge_small_bins(obs_counts, exp_counts)
    return _chi2_result(
        field_name, spec, obs_m, exp_m, alpha, observed_mean=float(np.mean(observed))
    )


def _categorical_chi2_field(
    field_name: str,
    spec: DistributionSpec,
    column: list[Any],
    alpha: float,
) -> FieldFidelity:
    weights: dict[str, float] = spec.weights  # type: ignore[attr-defined]
    n = len(column)
    categories = list(weights.keys())
    obs_counts = np.array([column.count(c) for c in categories], dtype=float)
    exp_counts = np.array([weights[c] * n for c in categories], dtype=float)

    obs_m, exp_m = _merge_small_bins(obs_counts, exp_counts)
    return _chi2_result(field_name, spec, obs_m, exp_m, alpha, observed_mean=None)


def _chi2_result(
    field_name: str,
    spec: DistributionSpec,
    obs: NDArray[Any],
    exp: NDArray[Any],
    alpha: float,
    *,
    observed_mean: float | None,
) -> FieldFidelity:
    dist_type = spec.distribution_type
    if len(exp) < 2:
        # Too few bins to run a meaningful test - report as a trivial pass.
        return FieldFidelity(
            field=field_name,
            distribution=dist_type,
            test="chi2",
            statistic=float("nan"),
            p_value=1.0,
            passed=True,
            observed_mean=observed_mean,
            expected_mean=_expected_mean(spec) if dist_type in _DISCRETE else None,
        )

    statistic, p_value = stats.chisquare(obs, exp)
    return FieldFidelity(
        field=field_name,
        distribution=dist_type,
        test="chi2",
        statistic=float(statistic),
        p_value=float(p_value),
        passed=float(p_value) >= alpha,
        observed_mean=observed_mean,
        expected_mean=_expected_mean(spec) if dist_type in _DISCRETE else None,
    )


def _check_correlation(
    field1: str,
    field2: str,
    target: float,
    specs: dict[str, DistributionSpec],
    rows: list[dict[str, Any]],
    tolerance: float,
) -> CorrelationFidelity | None:
    """Compare an observed correlation to its target, or None if uncheckable.

    Pairs that reference a non-distribution field, a field absent from the
    records, or a distribution that cannot be correlated (e.g. categorical)
    are skipped - they are never correlated during generation.
    """
    for name in (field1, field2):
        spec = specs.get(name)
        if spec is None or not spec.supports_correlation:
            return None
        if name not in rows[0]:
            return None

    x = np.asarray([row[field1] for row in rows], dtype=float)
    y = np.asarray([row[field2] for row in rows], dtype=float)

    # Correlation is undefined when either column is constant; report it as a
    # (failing) NaN rather than triggering scipy/numpy divide-by-zero warnings.
    if np.std(x) == 0 or np.std(y) == 0:
        spearman = pearson = float("nan")
    else:
        spearman = float(stats.spearmanr(x, y).statistic)
        pearson = float(np.corrcoef(x, y)[0, 1])
    error = abs(spearman - target)
    return CorrelationFidelity(
        field1=field1,
        field2=field2,
        target=target,
        observed_spearman=spearman,
        observed_pearson=pearson,
        error=error,
        passed=error <= tolerance,
    )
