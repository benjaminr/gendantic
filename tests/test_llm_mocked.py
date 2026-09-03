"""Offline tests for the LLM-dependent paths using a mocked LiteLLM client.

These exercise the generation pipeline (numpy sampling + LLM field merge +
validation), the model-analysis wiring, and dynamic model generation without
requiring a live LiteLLM proxy. The single seam we mock is
``get_client().generate_structured(schema, prompt, count)`` via the
``make_client`` / ``patch_clients`` fixtures in conftest.py.
"""

from typing import Annotated, Any

import pytest
from pydantic import BaseModel, Field

from gendantic import (
    Normal,
    Uniform,
    generate_model_from_description,
    generate_synthetic_data,
    generate_synthetic_data_batch,
    generate_synthetic_data_sync,
)


class Employee(BaseModel):
    """Employee with mixed distribution + LLM fields."""

    name: str
    email: str
    age: Annotated[int, Uniform(min=22, max=65)]
    salary: Annotated[float, Normal(mean=75000, std=20000)]


@pytest.mark.asyncio
async def test_pipeline_merges_sampled_and_generated(make_client, patch_clients) -> None:
    """Distribution fields come from numpy; other fields from the LLM; merged."""
    client = make_client(
        lambda schema, prompt, count: [
            {"name": f"Person {i}", "email": f"person{i}@example.com"}
            for i in range(count)
        ]
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
async def test_distribution_fields_are_not_asked_of_llm(make_client, patch_clients) -> None:
    """The LLM is only asked for non-distribution fields."""
    seen_field_schemas: list[list[str]] = []

    def fields(schema: dict[str, Any], prompt: str, count: int) -> list[dict]:
        seen_field_schemas.append(sorted(schema["items"]["properties"].keys()))
        return [{"name": f"n{i}", "email": f"e{i}@x.com"} for i in range(count)]

    with patch_clients(make_client(fields)):
        await generate_synthetic_data(Employee, count=3, seed=1)

    # age/salary are sampled by numpy and must not appear in the LLM schema
    assert seen_field_schemas == [["email", "name"]]


@pytest.mark.asyncio
async def test_seed_is_reproducible(make_client, patch_clients) -> None:
    def values(schema: dict[str, Any], prompt: str, count: int) -> list[dict]:
        return [{"name": "x", "email": "x@x.com"} for _ in range(count)]

    with patch_clients(make_client(values)):
        a = await generate_synthetic_data(Employee, count=4, seed=7)
    with patch_clients(make_client(values)):
        b = await generate_synthetic_data(Employee, count=4, seed=7)
    assert [r.age for r in a] == [r.age for r in b]
    assert [r.salary for r in a] == [r.salary for r in b]


@pytest.mark.asyncio
async def test_invalid_records_are_topped_up_to_reach_count(
    make_client, patch_clients
) -> None:
    """A transient invalid record is regenerated so the full count is returned."""

    class Bounded(BaseModel):
        label: str = Field(min_length=3)
        age: Annotated[int, Uniform(min=22, max=65)]

    # First-ever record is too short (invalid); every record after that is
    # valid. A single top-up round should recover the shortfall.
    calls = {"n": 0}

    def values(schema: dict[str, Any], prompt: str, count: int) -> list[dict]:
        out = []
        for i in range(count):
            first_ever = calls["n"] == 0 and i == 0
            out.append({"label": "x" if first_ever else f"label{calls['n']}_{i}"})
        calls["n"] += 1
        return out

    with patch_clients(make_client(values)):
        rows = await generate_synthetic_data(Bounded, count=4, seed=3)

    assert len(rows) == 4  # shortfall regenerated, not dropped
    assert all(len(r.label) >= 3 for r in rows)
    assert calls["n"] == 2  # one initial round + one top-up round


@pytest.mark.asyncio
async def test_persistent_validation_failure_raises(make_client, patch_clients) -> None:
    """When every record fails validation, generation raises rather than lying."""

    class Bounded(BaseModel):
        label: str = Field(min_length=3)
        age: Annotated[int, Uniform(min=22, max=65)]

    # Always return too-short labels: no top-up round can succeed.
    def values(schema: dict[str, Any], prompt: str, count: int) -> list[dict]:
        return [{"label": "x"} for _ in range(count)]

    with patch_clients(make_client(values)):
        with pytest.raises(ValueError, match="No valid .* records could be generated"):
            await generate_synthetic_data(Bounded, count=4, seed=3)


@pytest.mark.asyncio
async def test_partial_progress_then_exhaustion_raises(
    make_client, patch_clients
) -> None:
    """If retries make partial progress but can't reach count, it raises."""

    class Bounded(BaseModel):
        label: str = Field(min_length=3)
        age: Annotated[int, Uniform(min=22, max=65)]

    # The first record of every batch is invalid: round 1 (count=4) yields 3
    # valid, the size-1 top-up round then yields 0 and stops.
    def values(schema: dict[str, Any], prompt: str, count: int) -> list[dict]:
        return [{"label": "x" if i == 0 else f"label{i}"} for i in range(count)]

    with patch_clients(make_client(values)):
        with pytest.raises(ValueError, match="Could only generate 3 of 4"):
            await generate_synthetic_data(Bounded, count=4, seed=3)


@pytest.mark.asyncio
async def test_batch_generation_over_contexts(make_client, patch_clients) -> None:
    client = make_client(
        lambda schema, prompt, count: [
            {"name": "n", "email": "n@x.com"} for _ in range(count)
        ]
    )
    with patch_clients(client):
        batches = await generate_synthetic_data_batch(
            Employee, contexts=["UK bank", "US startup"], count=3, seed=9
        )
    assert len(batches) == 2
    assert all(len(b) == 3 for b in batches)


@pytest.mark.asyncio
async def test_all_distribution_fields_skips_llm(make_client, patch_clients) -> None:
    """A model fully covered by distributions needs no LLM field call."""

    class AllDist(BaseModel):
        age: Annotated[int, Uniform(min=22, max=65)]
        salary: Annotated[float, Normal(mean=50000, std=10000)]

    calls = {"field": 0}

    def fields(schema: dict[str, Any], prompt: str, count: int) -> list[dict]:
        calls["field"] += 1
        return [{} for _ in range(count)]

    with patch_clients(make_client(fields)):
        rows = await generate_synthetic_data(AllDist, count=5, seed=2)

    assert len(rows) == 5
    assert calls["field"] == 0  # no field-generation LLM call


def test_sync_wrapper_runs_without_event_loop(make_client, patch_clients) -> None:
    """generate_synthetic_data_sync runs the async pipeline via asyncio.run."""
    client = make_client(
        lambda schema, prompt, count: [
            {"name": f"Person {i}", "email": f"person{i}@example.com"}
            for i in range(count)
        ]
    )
    with patch_clients(client):
        rows = generate_synthetic_data_sync(Employee, count=3, seed=42)

    assert len(rows) == 3
    assert all(isinstance(r, Employee) for r in rows)
    assert all(22 <= r.age <= 65 for r in rows)


@pytest.mark.asyncio
async def test_sync_wrapper_rejects_running_loop() -> None:
    """Calling the sync wrapper inside a running loop raises a clear error."""
    with pytest.raises(RuntimeError, match="running.*event loop"):
        generate_synthetic_data_sync(Employee, count=1)


def test_make_schema_strict_enforces_strict_rules() -> None:
    """Every object level gets additionalProperties:false and full required."""
    from gendantic.llm import _make_schema_strict

    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "address": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
            "tags": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"label": {"type": "string"}},
                },
            },
        },
    }
    strict = _make_schema_strict(schema)

    assert strict["additionalProperties"] is False
    assert sorted(strict["required"]) == ["address", "name", "tags"]
    # Nested object inside a property.
    assert strict["properties"]["address"]["additionalProperties"] is False
    assert strict["properties"]["address"]["required"] == ["city"]
    # Nested object inside an array's items.
    item = strict["properties"]["tags"]["items"]
    assert item["additionalProperties"] is False
    assert item["required"] == ["label"]
    # Original schema is not mutated.
    assert "additionalProperties" not in schema


@pytest.mark.asyncio
async def test_generate_model_from_description_mocked(make_client, patch_clients) -> None:
    """The dynamic model generator builds a working model from LLM code."""
    code = (
        "class Ticket(BaseModel):\n"
        '    """A support ticket."""\n'
        "    title: str\n"
        "    priority: Annotated[int, Uniform(min=1, max=5)]\n"
    )
    client = make_client(lambda schema, prompt, count: [{"code": code}])
    with patch_clients(client):
        model_class, src = await generate_model_from_description(
            "a support ticket", model_name="Ticket"
        )

    assert model_class.__name__ == "Ticket"
    assert "priority" in model_class.model_fields
    assert "title" in model_class.model_fields
    assert src == code
