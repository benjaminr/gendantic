"""Markers must be found wherever Pydantic finds the field.

gendantic used to read ``Annotated`` markers straight off
``model_class.__annotations__``, which only holds the annotations written on
that exact class and, under ``from __future__ import annotations``, holds raw
strings. These tests pin the ``model_fields``-based introspection so that
inherited fields, postponed annotations and ``Optional`` wrappers all work.
"""

from typing import Annotated, Optional

from pydantic import BaseModel, Field

from gendantic import (
    ForeignKey,
    Normal,
    PrimaryKey,
    Uniform,
    generate_synthetic_data,
)
from gendantic._fields import iter_fields, unwrap_optional
from gendantic.llm_driven_analyser import LLMDrivenModelAnalyser
from gendantic.relational import _primary_key_columns, extract_foreign_keys
from gendantic.sampler import DistributionSampler

from . import _pep563_models as pep563


class BaseRecord(BaseModel):
    id: Annotated[int, PrimaryKey()]
    age: Annotated[int, Uniform(min=18, max=65)]


class Employee(BaseRecord):
    salary: Annotated[float, Normal(mean=1000, std=100)] = Field(ge=0)
    name: str


class Assignment(BaseModel):
    employee_id: Annotated[int, ForeignKey(Employee)]


class SubAssignment(Assignment):
    note: str


def test_inherited_distribution_specs_are_found() -> None:
    specs = LLMDrivenModelAnalyser.extract_distribution_specs(Employee)
    assert set(specs) == {"age", "salary"}
    assert specs["age"] == Uniform(min=18, max=65)


def test_inherited_constraints_and_types_are_found() -> None:
    typed = LLMDrivenModelAnalyser.extract_distribution_specs_with_types(Employee)
    assert typed["age"][1] is int
    assert typed["salary"][1] is float
    assert typed["salary"][2]["ge"] == 0.0


def test_inherited_primary_key_is_found() -> None:
    assert [c.field for c in _primary_key_columns(Employee)] == ["id"]


def test_inherited_foreign_key_is_found() -> None:
    assert set(extract_foreign_keys(SubAssignment)) == {"employee_id"}


def test_postponed_annotations_are_resolved() -> None:
    # Sanity: under PEP 563 the raw annotations really are strings.
    assert isinstance(pep563.Member.__annotations__["score"], str)

    specs = LLMDrivenModelAnalyser.extract_distribution_specs(pep563.Member)
    assert set(specs) == {"score"}
    assert [c.field for c in _primary_key_columns(pep563.Member)] == ["id"]
    assert set(extract_foreign_keys(pep563.Member)) == {"team_id"}
    assert _primary_key_columns(pep563.Team)[0].base_type is int


def test_unwrap_optional() -> None:
    assert unwrap_optional(Optional[int]) is int
    assert unwrap_optional(int | None) is int
    assert unwrap_optional(int) is int
    # A union with two real members is not an Optional and is left alone.
    assert unwrap_optional(int | str | None) == (int | str | None)


def test_optional_outer_and_inner_forms_yield_same_markers() -> None:
    class M(BaseModel):
        outer: Annotated[Optional[int], Uniform(min=1, max=5)] = None
        inner: Optional[Annotated[int, Uniform(min=1, max=5)]] = None

    fields = {name: (base, markers) for name, base, markers in iter_fields(M)}
    assert fields["outer"][0] is int
    assert fields["inner"][0] is int
    assert Uniform(min=1, max=5) in fields["outer"][1]
    assert Uniform(min=1, max=5) in fields["inner"][1]


def test_optional_int_distribution_field_is_sampled_as_int() -> None:
    class M(BaseModel):
        n: Annotated[Optional[int], Uniform(min=1, max=5)] = None

    specs = LLMDrivenModelAnalyser.extract_distribution_specs_with_types(M)
    rows = DistributionSampler(seed=0).sample_fields(specs, 50)
    assert all(isinstance(row["n"], int) for row in rows)


async def test_optional_int_distribution_field_validates_end_to_end() -> None:
    class M(BaseModel):
        n: Annotated[Optional[int], Uniform(min=1, max=5)] = None

    # Every field is distribution-sampled, so no LLM call is made; before the
    # fix these came out as floats and every record failed validation.
    records = await generate_synthetic_data(M, count=20, seed=3)
    assert len(records) == 20
    assert all(isinstance(r.n, int) for r in records)
