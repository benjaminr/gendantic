"""Self-referential + multi-table relational generation (runs offline).

Demonstrates two relational features beyond the basic quickstart, with no LLM
call (every non-key field is a distribution, so numpy samples everything):

- **Self-references** via a string forward reference: an ``Employee`` has a
  ``manager_id`` that points at another ``Employee``. You can't name the class
  inside its own body, so pass its name as a string: ``ForeignKey("Employee")``.
- **Nullable foreign keys**: top-level employees (and a few unassigned ones)
  have ``manager_id = None``.

Shape:

    Department  1--*  Employee  *--1  Employee (manager)

Run it with:

    uv run python examples/relational_hierarchy.py
"""

from typing import Annotated, Optional

from pydantic import BaseModel

from gendantic import (
    Categorical,
    ForeignKey,
    LogNormal,
    Normal,
    PrimaryKey,
    generate_dataset_sync,
)


class Department(BaseModel):
    id: Annotated[int, PrimaryKey()]
    budget: Annotated[float, LogNormal(mean=13.0, sigma=0.5)]


class Employee(BaseModel):
    id: Annotated[int, PrimaryKey()]
    department_id: Annotated[int, ForeignKey(Department)]
    # Self-reference: a manager is another Employee. Use a string forward
    # reference because the class isn't defined yet inside its own body.
    manager_id: Annotated[
        Optional[int],
        ForeignKey("Employee", nullable=True, null_probability=0.25),
    ] = None
    level: Annotated[
        str,
        Categorical(weights={"junior": 0.5, "mid": 0.3, "senior": 0.2}),
    ]
    salary: Annotated[float, Normal(mean=55000, std=15000)]


def main() -> None:
    dataset = generate_dataset_sync(
        {Department: 4, Employee: 30},
        seed=7,
    )

    departments = dataset[Department]
    employees = dataset[Employee]
    print(f"Generated {len(departments)} departments, {len(employees)} employees\n")

    # Referential integrity, including the self-reference.
    department_ids = {d.id for d in departments}
    employee_ids = {e.id for e in employees}
    assert all(e.department_id in department_ids for e in employees)
    assert all(
        e.manager_id in employee_ids for e in employees if e.manager_id is not None
    )
    top_level = [e for e in employees if e.manager_id is None]
    assert top_level  # some employees have no manager
    print(
        f"Referential integrity verified. "
        f"{len(top_level)} top-level employees (no manager).\n"
    )

    # to_dataframes() gives a {model_name: DataFrame} mapping (needs pandas).
    try:
        frames = dataset.to_dataframes()
        print("DataFrames:", {name: df.shape for name, df in frames.items()})
    except ImportError:
        print("(install the 'pandas' extra to get DataFrame output)")


if __name__ == "__main__":
    main()
