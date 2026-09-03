"""Conditional distributions and cross-field constraints (runs offline).

Every field here is backed by a distribution, so gendantic samples the whole
batch with numpy and never calls an LLM. It demonstrates the two expressiveness
mechanisms that copulas cannot cover:

  * ``Conditional`` — a field's distribution depends on another field's value,
    keyed either by exact category or by numeric ``Range`` bins.
  * ``Constraints(Ordering(...))`` — two or more fields always come out sorted
    ascending per record.

Run it with:

    uv run python examples/conditional.py
"""

import statistics
from typing import Annotated

from pydantic import BaseModel

from gendantic import (
    Categorical,
    Conditional,
    Constraints,
    Normal,
    Ordering,
    Range,
    Uniform,
    fidelity_report,
    generate_synthetic_data_sync,
)


class Employee(BaseModel):
    # Discriminator: which department the employee is in.
    department: Annotated[
        str, Categorical(weights={"Eng": 0.5, "Sales": 0.3, "HR": 0.2})
    ]
    # Salary distribution switches on the department. HR has no case of its own,
    # so it falls through to the default.
    salary: Annotated[
        float,
        Conditional(
            on="department",
            cases={
                "Eng": Normal(mean=90000, std=15000),
                "Sales": Normal(mean=70000, std=20000),
            },
            default=Normal(mean=50000, std=10000),
        ),
    ]
    # Discriminator: age, used with numeric Range bins below.
    age: Annotated[int, Uniform(min=20, max=65)]
    # Annual bonus grows with age band. Ranges are half-open [min, max) and are
    # matched on the *converted* (int) age the record exposes.
    bonus: Annotated[
        float,
        Conditional(
            on="age",
            cases={
                Range(max=30): Normal(mean=2000, std=300),
                Range(30, 50): Normal(mean=5000, std=400),
                Range(min=50): Normal(mean=9000, std=500),
            },
            default=Normal(mean=0, std=1),
        ),
    ]


class Project(BaseModel):
    # Cross-field ordering: a project's review never precedes its kickoff, and
    # its deadline never precedes the review.
    kickoff_day: Annotated[float, Uniform(min=0, max=365)]
    review_day: Annotated[float, Uniform(min=0, max=365)]
    deadline_day: Annotated[float, Uniform(min=0, max=365)]

    __constraints__ = Constraints(
        Ordering("kickoff_day", "review_day", "deadline_day"),
    )


def demo_conditionals() -> None:
    employees = generate_synthetic_data_sync(Employee, count=2000, seed=42)
    print(f"Generated {len(employees)} employees (no LLM call)\n")

    # 1. Conditional categorical: mean salary per department matches its case.
    print("Mean salary by department (conditional on department):")
    by_dept: dict[str, list[float]] = {}
    for e in employees:
        by_dept.setdefault(e.department, []).append(e.salary)
    for dept in ("Eng", "Sales", "HR"):
        print(f"  {dept:6s} £{statistics.mean(by_dept[dept]):,.0f}")

    # 2. Conditional numeric bins: mean bonus rises with the age band.
    print("\nMean bonus by age band (conditional on numeric Range):")
    bands: dict[str, list[float]] = {"<30": [], "30-49": [], "50+": []}
    for e in employees:
        key = "<30" if e.age < 30 else "30-49" if e.age < 50 else "50+"
        bands[key].append(e.bonus)
    for key in ("<30", "30-49", "50+"):
        print(f"  {key:6s} £{statistics.mean(bands[key]):,.0f}")

    # 3. Prove it statistically: fidelity checks conditional fields per branch.
    #    alpha=0.01 keeps the aggregate verdict robust to the occasional
    #    goodness-of-fit false positive across many checks.
    print("\nFidelity report (conditional fields checked per case branch):")
    report = fidelity_report(employees, Employee, alpha=0.01)
    for f in report.fields:
        label = f.field if f.group is None else f"{f.field} | {f.group}"
        status = "PASS" if f.passed else "FAIL"
        print(f"  [{status}] {label} ({f.distribution}) p={f.p_value:.3f}")
    print(f"  -> overall passed: {report.passed}")


def demo_ordering() -> None:
    projects = generate_synthetic_data_sync(Project, count=2000, seed=7)

    # The invariant holds for every record.
    violations = sum(
        1
        for p in projects
        if not (p.kickoff_day <= p.review_day <= p.deadline_day)
    )
    print(f"\nOrdering kickoff <= review <= deadline: {violations} violations "
          f"/ {len(projects)}")

    # Trade-off: the sort reassigns which field gets which value, so each
    # constrained field's marginal becomes an order statistic. The three fields
    # are sampled from the *same* Uniform(0, 365), yet their means now spread
    # out (min / median / max) rather than all sitting near 182.
    means = {
        "kickoff": statistics.mean(p.kickoff_day for p in projects),
        "review": statistics.mean(p.review_day for p in projects),
        "deadline": statistics.mean(p.deadline_day for p in projects),
    }
    print("Per-field means after sorting (all sampled from Uniform(0, 365)):")
    for name, value in means.items():
        print(f"  {name:9s} {value:6.1f}")
    print("  (min / median / max order statistics — the ordering trade-off)")


class Career(BaseModel):
    # Separated marginals: born, then hired, then left. method="resample"
    # keeps each field's own distribution and only redraws the rare violating
    # rows, so the per-field marginals are preserved (unlike sort).
    birth_year: Annotated[float, Uniform(min=1960, max=1990)]
    hire_year: Annotated[float, Uniform(min=1990, max=2010)]
    exit_year: Annotated[float, Uniform(min=2010, max=2025)]

    __constraints__ = Constraints(
        Ordering("birth_year", "hire_year", "exit_year", method="resample"),
    )


def demo_resample_ordering() -> None:
    careers = generate_synthetic_data_sync(Career, count=2000, seed=11)

    violations = sum(
        1
        for c in careers
        if not (c.birth_year <= c.hire_year <= c.exit_year)
    )
    print(f"\nResample ordering birth <= hire <= exit: {violations} violations "
          f"/ {len(careers)}")

    # Each field's mean stays at its own distribution's midpoint (1975 / 2000 /
    # 2017), because resample preserves marginals rather than reshaping them.
    means = {
        "birth_year": statistics.mean(c.birth_year for c in careers),
        "hire_year": statistics.mean(c.hire_year for c in careers),
        "exit_year": statistics.mean(c.exit_year for c in careers),
    }
    print("Per-field means (each preserves its own Uniform midpoint):")
    for name, value in means.items():
        print(f"  {name:10s} {value:7.1f}")


def main() -> None:
    demo_conditionals()
    demo_ordering()
    demo_resample_ordering()


if __name__ == "__main__":
    main()
