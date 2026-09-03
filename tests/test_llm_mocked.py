"""Offline tests for the LLM-dependent paths using a mocked LiteLLM client.

These exercise the generation pipeline (numpy sampling + LLM field merge +
validation), the model-analysis wiring, and dynamic model generation without
requiring a live LiteLLM proxy. The single seam we mock is
``get_client().generate_structured(schema, prompt, count)`` via the
``make_client`` / ``patch_clients`` fixtures in conftest.py.
"""

import asyncio
from typing import Annotated, Any
from unittest.mock import AsyncMock

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

from .conftest import ANALYSIS, is_analysis_call


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
async def test_llm_field_calls_are_concurrency_bounded(patch_clients) -> None:
    """Many record batches don't all hit the LLM at once.

    With a large ``count`` the pipeline produces more field-generation batches
    than :data:`_MAX_CONCURRENT_LLM_CALLS`; the semaphore must keep the number
    of simultaneously in-flight calls at or below that cap.
    """
    from gendantic.generator import _MAX_CONCURRENT_LLM_CALLS

    class OnlyLLM(BaseModel):
        name: str  # no distribution fields -> every batch is an LLM call

    inflight = 0
    peak = 0

    async def gen(
        schema: dict[str, Any], prompt: str, count: int = 1
    ) -> list[dict[str, Any]]:
        nonlocal inflight, peak
        if is_analysis_call(schema):
            return [ANALYSIS]
        inflight += 1
        peak = max(peak, inflight)
        await asyncio.sleep(0.01)  # hold the slot so overlap is observable
        inflight -= 1
        return [{"name": f"n{i}"} for i in range(count)]

    client = AsyncMock()
    client.generate_structured = AsyncMock(side_effect=gen)

    # count=200 with the default batch size of 15 yields ~14 batches, well above
    # the concurrency cap of 8.
    with patch_clients(client):
        rows = await generate_synthetic_data(OnlyLLM, count=200, seed=1)

    assert len(rows) == 200
    assert peak > 1  # sanity: batches really do overlap
    assert peak <= _MAX_CONCURRENT_LLM_CALLS  # but never beyond the cap


@pytest.mark.asyncio
async def test_invalid_records_are_dropped_not_raised(make_client, patch_clients) -> None:
    """Records failing validation are skipped with a warning, not fatal."""

    class Bounded(BaseModel):
        label: str = Field(min_length=3)
        age: Annotated[int, Uniform(min=22, max=65)]

    # Return one too-short label (invalid) among valid ones.
    def values(schema: dict[str, Any], prompt: str, count: int) -> list[dict]:
        return [{"label": "x" if i == 0 else f"label{i}"} for i in range(count)]

    with patch_clients(make_client(values)):
        rows = await generate_synthetic_data(Bounded, count=4, seed=3)

    assert len(rows) == 3  # one dropped
    assert all(len(r.label) >= 3 for r in rows)


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
