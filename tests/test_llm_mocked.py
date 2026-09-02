"""Offline tests for the LLM-dependent paths using a mocked LiteLLM client.

These exercise the generation pipeline (numpy sampling + LLM field merge +
validation), the model-analysis wiring, and dynamic model generation without
requiring a live LiteLLM proxy. The single seam we mock is
``get_client().generate_structured(schema, prompt, count)``.
"""

import json
from contextlib import ExitStack, contextmanager
from typing import Annotated, Any, Callable, Iterator
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel, Field

from gendantic import (
    Normal,
    Uniform,
    generate_model_from_description,
    generate_synthetic_data,
    generate_synthetic_data_batch,
)

# A canned analysis payload matching LLMDrivenModelAnalyser._ANALYSIS_SCHEMA.
_ANALYSIS = {
    "model_analysis": {
        "purpose": "test",
        "domain": "testing",
        "use_case": "unit-test",
        "data_patterns": "synthetic",
    },
    "generation_guidance": {
        "overall_strategy": "s",
        "field_relationships": "r",
        "data_quality_approach": "q",
        "cultural_considerations": "c",
    },
}


def _is_analysis_call(schema: dict[str, Any]) -> bool:
    return "model_analysis" in json.dumps(schema)


def make_fake_client(field_values_fn: Callable[[int], dict[str, Any]]) -> Any:
    """Build a fake client whose generate_structured returns analysis payloads
    for analysis calls and ``field_values_fn(i)`` rows for field-generation calls.
    """

    def gen(
        schema: dict[str, Any], prompt: str, count: int = 1
    ) -> list[dict[str, Any]]:
        if _is_analysis_call(schema):
            return [_ANALYSIS]
        return [field_values_fn(i) for i in range(count)]

    client = AsyncMock()
    client.generate_structured = AsyncMock(side_effect=gen)
    return client


@contextmanager
def patch_clients(client: Any) -> Iterator[None]:
    """Patch get_client in every module that imported it."""
    with ExitStack() as stack:
        for mod in ("generator", "llm_driven_analyser", "model_generator"):
            stack.enter_context(
                patch(f"gendantic.{mod}.get_client", return_value=client)
            )
        yield


class Employee(BaseModel):
    """Employee with mixed distribution + LLM fields."""

    name: str
    email: str
    age: Annotated[int, Uniform(min=22, max=65)]
    salary: Annotated[float, Normal(mean=75000, std=20000)]


@pytest.mark.asyncio
async def test_pipeline_merges_sampled_and_generated() -> None:
    """Distribution fields come from numpy; other fields from the LLM; merged."""
    client = make_fake_client(
        lambda i: {"name": f"Person {i}", "email": f"person{i}@example.com"}
    )
    with patch_clients(client):
        rows = await generate_synthetic_data(Employee, count=5, seed=42)

    assert len(rows) == 5
    for r in rows:
        assert isinstance(r, Employee)
        assert r.name  # from LLM
        assert "@" in r.email  # from LLM
        assert 22 <= r.age <= 65  # from numpy sampling
        assert r.salary > 0


@pytest.mark.asyncio
async def test_distribution_fields_are_not_asked_of_llm() -> None:
    """The LLM is only asked for non-distribution fields."""
    seen_field_schemas: list[list[str]] = []

    def gen(schema: dict[str, Any], prompt: str, count: int = 1):
        if _is_analysis_call(schema):
            return [_ANALYSIS]
        props = schema["items"]["properties"].keys()
        seen_field_schemas.append(sorted(props))
        return [{"name": f"n{i}", "email": f"e{i}@x.com"} for i in range(count)]

    client = AsyncMock()
    client.generate_structured = AsyncMock(side_effect=gen)
    with patch_clients(client):
        await generate_synthetic_data(Employee, count=3, seed=1)

    # age/salary are sampled by numpy and must not appear in the LLM schema
    assert seen_field_schemas == [["email", "name"]]


@pytest.mark.asyncio
async def test_seed_is_reproducible() -> None:
    client = make_fake_client(lambda i: {"name": "x", "email": "x@x.com"})
    with patch_clients(client):
        a = await generate_synthetic_data(Employee, count=4, seed=7)
    client2 = make_fake_client(lambda i: {"name": "x", "email": "x@x.com"})
    with patch_clients(client2):
        b = await generate_synthetic_data(Employee, count=4, seed=7)
    assert [r.age for r in a] == [r.age for r in b]
    assert [r.salary for r in a] == [r.salary for r in b]


@pytest.mark.asyncio
async def test_invalid_records_are_dropped_not_raised(caplog: Any) -> None:
    """Records failing validation are skipped with a warning, not fatal."""

    class Bounded(BaseModel):
        label: str = Field(min_length=3)
        age: Annotated[int, Uniform(min=22, max=65)]

    # Return one too-short label (invalid) among valid ones.
    def values(i: int) -> dict[str, Any]:
        return {"label": "x" if i == 0 else f"label{i}"}

    client = make_fake_client(values)
    with patch_clients(client):
        rows = await generate_synthetic_data(Bounded, count=4, seed=3)

    assert len(rows) == 3  # one dropped
    assert all(len(r.label) >= 3 for r in rows)


@pytest.mark.asyncio
async def test_batch_generation_over_contexts() -> None:
    client = make_fake_client(lambda i: {"name": "n", "email": "n@x.com"})
    with patch_clients(client):
        batches = await generate_synthetic_data_batch(
            Employee, contexts=["UK bank", "US startup"], count=3, seed=9
        )
    assert len(batches) == 2
    assert all(len(b) == 3 for b in batches)


@pytest.mark.asyncio
async def test_all_distribution_fields_skips_llm() -> None:
    """A model fully covered by distributions needs no LLM field call."""

    class AllDist(BaseModel):
        age: Annotated[int, Uniform(min=22, max=65)]
        salary: Annotated[float, Normal(mean=50000, std=10000)]

    calls = {"field": 0}

    def gen(schema: dict[str, Any], prompt: str, count: int = 1):
        if _is_analysis_call(schema):
            return [_ANALYSIS]
        calls["field"] += 1
        return [{} for _ in range(count)]

    client = AsyncMock()
    client.generate_structured = AsyncMock(side_effect=gen)
    with patch_clients(client):
        rows = await generate_synthetic_data(AllDist, count=5, seed=2)

    assert len(rows) == 5
    assert calls["field"] == 0  # no field-generation LLM call


@pytest.mark.asyncio
async def test_generate_model_from_description_mocked() -> None:
    """The dynamic model generator builds a working model from LLM code."""
    code = (
        "class Ticket(BaseModel):\n"
        '    """A support ticket."""\n'
        "    title: str\n"
        "    priority: Annotated[int, Uniform(min=1, max=5)]\n"
    )
    client = AsyncMock()
    client.generate_structured = AsyncMock(return_value=[{"code": code}])
    with patch_clients(client):
        model_class, src = await generate_model_from_description(
            "a support ticket", model_name="Ticket"
        )

    assert model_class.__name__ == "Ticket"
    assert "priority" in model_class.model_fields
    assert "title" in model_class.model_fields
    assert src == code
