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

Conditional fields are checked *per group*: records are split by which case
matched (on the discriminator value stored in the record) and each group is run
against its own case spec, so a conditional field yields one result per branch.

Correlations are checked with the rank statistic the copula family actually
targets: Kendall's tau for the Archimedean families (Clayton, Gumbel, Frank),
whose ``corr`` is a target tau, and Spearman's rho for Gaussian and Student-t,
whose ``corr`` is a latent correlation that Spearman tracks closely. Both rank
statistics plus Pearson are reported alongside so the verdict can be inspected.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel
from scipy import stats

from .distributions import (
    Categorical,
    Conditional,
    CopulaType,
    Correlations,
    DistributionSpec,
    Range,
    truncate,
)
from .llm_driven_analyser import LLMDrivenModelAnalyser

_CONTINUOUS = {"normal", "uniform", "lognormal", "exponential", "beta"}
_DISCRETE = {"poisson", "binomial"}

# Archimedean families interpret ``corr`` as a target Kendall's tau; the others
# (Gaussian, Student-t) interpret it as a latent correlation tracked by Spearman.
_KENDALL_COPULAS = {CopulaType.CLAYTON, CopulaType.GUMBEL, CopulaType.FRANK}


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
    # For conditional fields: which case branch this result covers (e.g.
    # ``department='Eng'`` or ``age=[30,50)``). None for plain fields.
    group: str | None = None


@dataclass
class CorrelationFidelity:
    """How closely one declared correlation was reproduced."""

    field1: str
    field2: str
    target: float
    observed_spearman: float
    observed_pearson: float
    error: float  # |observed rank statistic - target|; see ``basis``
    passed: bool
    observed_kendall: float = float("nan")
    # Which rank statistic the verdict uses: "kendall" for Archimedean
    # families (target is a Kendall's tau), "spearman" otherwise.
    basis: str = "spearman"


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
                        f" mean obs={f.observed_mean:.4g}/exp={f.expected_mean:.4g}"
                    )
                label = f.field if f.group is None else f"{f.field} | {f.group}"
                lines.append(f"  [{mark}] {label} ({f.distribution}): {detail}")

        if self.correlations:
            lines.append("")
            lines.append("Correlations:")
            for c in self.correlations:
                mark = "ok " if c.passed else "XX "
                stat = (
                    f"kendall={c.observed_kendall:+.2f}"
                    if c.basis == "kendall"
                    else f"spearman={c.observed_spearman:+.2f}"
                )
                lines.append(
                    f"  [{mark}] {c.field1}~{c.field2}: target={c.target:+.2f} "
                    f"{stat} pearson={c.observed_pearson:+.2f} err={c.error:.2f}"
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

    Conditional fields are checked per case branch: records are grouped by which
    case matched (on the discriminator value stored in the record) and each
    group is tested against its own case spec, producing one result per branch.

    Args:
        records: Generated model instances or plain dicts.
        model_class: The model the records were generated for.
        alpha: Significance level for goodness-of-fit tests. A field passes
            when its p-value is >= ``alpha`` (i.e. we fail to reject the null
            hypothesis that the samples came from the specified distribution).
        correlation_tolerance: Maximum absolute difference between the observed
            rank statistic (Kendall's tau for Archimedean copula families,
            Spearman's rho otherwise) and the declared target for a pair to pass.

    Returns:
        A :class:`FidelityReport`. Never raises on statistical failure.
    """
    all_specs = LLMDrivenModelAnalyser.extract_distribution_specs_with_types(
        model_class
    )
    # Truncate bounded fields to the same window the sampler uses, so a field
    # declared e.g. ``ge=0`` is compared against the truncated distribution it
    # was actually sampled from rather than the full (untruncated) one.
    specs = {
        name: truncate(spec, target_type, constraints)
        for name, (spec, target_type, constraints) in all_specs.items()
        if isinstance(spec, DistributionSpec)
    }
    conditionals = {
        name: (spec, target_type, constraints)
        for name, (spec, target_type, constraints) in all_specs.items()
        if isinstance(spec, Conditional)
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

    for field_name, (cond, target_type, constraints) in conditionals.items():
        if field_name not in rows[0] or cond.on not in rows[0]:
            continue
        report.fields.extend(
            _check_conditional_field(
                field_name, cond, rows, alpha, target_type, constraints
            )
        )

    correlations = getattr(model_class, "__correlations__", None)
    if isinstance(correlations, Correlations):
        for f1, f2, target, copula in correlations:
            result = _check_correlation(
                f1, f2, target, copula, specs, rows, correlation_tolerance
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


def _format_case_label(key: Any) -> str:
    """Human-readable label for a conditional case key."""
    if isinstance(key, Range):
        if key.min is None:
            return f"<{key.max:g}"
        if key.max is None:
            return f">={key.min:g}"
        return f"[{key.min:g},{key.max:g})"
    return repr(key)


def _check_conditional_field(
    field_name: str,
    cond: Conditional,
    rows: list[dict[str, Any]],
    alpha: float,
    target_type: type = float,
    constraints: dict[str, float | None] | None = None,
) -> list[FieldFidelity]:
    """Goodness-of-fit per case branch of a conditional field.

    Records are grouped by the spec their (already-converted) discriminator
    value maps to - mirroring how the sampler assigned them - and each group is
    tested against that spec. Groups are labelled by the case key(s) that route
    to them (or ``default``), and results are returned in a stable label order.
    """
    # Map each spec object to the label(s) of the case(s) routing to it.
    labels: dict[int, list[str]] = {}
    for key, spec in cond.cases.items():
        labels.setdefault(id(spec), []).append(_format_case_label(key))
    labels.setdefault(id(cond.default), []).append("default")

    # Group the field's values by the matched spec.
    grouped: dict[int, list[Any]] = {}
    group_spec: dict[int, DistributionSpec] = {}
    for row in rows:
        spec = cond.spec_for(row[cond.on])
        key = id(spec)
        grouped.setdefault(key, []).append(row[field_name])
        group_spec[key] = spec

    bounds = constraints or {"ge": None, "le": None, "gt": None, "lt": None}
    results: list[FieldFidelity] = []
    for key, values in grouped.items():
        group = f"{cond.on}={'|'.join(sorted(labels.get(key, ['?'])))}"
        effective = truncate(group_spec[key], target_type, bounds)
        result = _check_field(field_name, effective, values, alpha)
        result.group = group
        results.append(result)

    results.sort(key=lambda r: r.group or "")
    return results


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
    assert isinstance(spec, Categorical)  # noqa: S101 - routed here by type
    n = len(column)
    categories = list(spec.weights.keys())
    obs_counts = np.array([column.count(c) for c in categories], dtype=float)

    # A value outside the declared categories has probability zero under the
    # spec, so the fit is rejected outright. (Reporting it this way, rather
    # than passing mismatched totals to scipy, keeps the "never raises"
    # contract of fidelity_report.)
    if obs_counts.sum() < n:
        return FieldFidelity(
            field=field_name,
            distribution=spec.distribution_type,
            test="chi2",
            statistic=float("inf"),
            p_value=0.0,
            passed=False,
            observed_mean=None,
        )

    # Use the same renormalised probabilities the sampler draws from, so that
    # weights summing to only ~1.0 (within Categorical's tolerance) still give
    # expected counts that total exactly n, as scipy's chi-square requires.
    exp_counts = spec._normalised_probabilities() * n

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
    copula: str,
    specs: dict[str, DistributionSpec],
    rows: list[dict[str, Any]],
    tolerance: float,
) -> CorrelationFidelity | None:
    """Compare an observed correlation to its target, or None if uncheckable.

    The verdict uses the rank statistic the copula family targets: Kendall's
    tau for Archimedean families (whose ``target`` is a tau) and Spearman's rho
    otherwise. Pairs that reference a non-distribution field, a field absent
    from the records, or a distribution that cannot be correlated (e.g.
    categorical) are skipped - they are never correlated during generation.
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
        spearman = pearson = kendall = float("nan")
    else:
        spearman = float(stats.spearmanr(x, y).statistic)
        pearson = float(np.corrcoef(x, y)[0, 1])
        kendall = float(stats.kendalltau(x, y).statistic)

    basis = "kendall" if copula in _KENDALL_COPULAS else "spearman"
    observed = kendall if basis == "kendall" else spearman
    error = abs(observed - target)
    return CorrelationFidelity(
        field1=field1,
        field2=field2,
        target=target,
        observed_spearman=spearman,
        observed_pearson=pearson,
        observed_kendall=kendall,
        basis=basis,
        error=error,
        passed=error <= tolerance,
    )
