"""Offline tests for relational (multi-model) generation with a mocked client."""

import json
from contextlib import ExitStack, contextmanager
from typing import Annotated, Any, Iterator
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from gendantic import (
    ForeignKey,
    Normal,
    PrimaryKey,
    generate_dataset,
    generate_dataset_sync,
)
from gendantic.relational import _resolve_generation_order

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


def make_fake_client() -> Any:
    """Return LLM-field values for whatever properties the schema requests."""

    def gen(
        schema: dict[str, Any], prompt: str, count: int = 1
    ) -> list[dict[str, Any]]:
        if _is_analysis_call(schema):
            return [_ANALYSIS]
        props = list(schema["items"]["properties"].keys())
        return [{p: f"{p}-{i}" for p in props} for i in range(count)]

    client = AsyncMock()
    client.generate_structured = AsyncMock(side_effect=gen)
    return client


@contextmanager
def patch_clients(client: Any) -> Iterator[None]:
    with ExitStack() as stack:
        for mod in ("generator", "llm_driven_analyser", "model_generator"):
            stack.enter_context(
                patch(f"gendantic.{mod}.get_client", return_value=client)
            )
        yield


class Customer(BaseModel):
    id: Annotated[int, PrimaryKey()]
    name: str


class Order(BaseModel):
    id: Annotated[int, PrimaryKey()]
    customer_id: Annotated[int, ForeignKey(Customer)]
    amount: Annotated[float, Normal(mean=200, std=50)]


@pytest.mark.asyncio
async def test_referential_integrity_and_counts() -> None:
    with patch_clients(make_fake_client()):
        dataset = await generate_dataset({Customer: 10, Order: 40}, seed=42)

    customers = dataset[Customer]
    orders = dataset[Order]
    assert len(customers) == 10
    assert len(orders) == 40

    customer_ids = {c.id for c in customers}
    assert len(customer_ids) == 10  # primary keys are unique
    # every foreign key points at a real customer
    assert all(o.customer_id in customer_ids for o in orders)


@pytest.mark.asyncio
async def test_generation_order_is_topological() -> None:
    """Child declared first still generates after its parent."""
    with patch_clients(make_fake_client()):
        dataset = await generate_dataset({Order: 20, Customer: 5}, seed=1)

    customer_ids = {c.id for c in dataset[Customer]}
    assert all(o.customer_id in customer_ids for o in dataset[Order])


@pytest.mark.asyncio
async def test_primary_keys_not_asked_of_llm() -> None:
    seen: list[list[str]] = []

    def gen(
        schema: dict[str, Any], prompt: str, count: int = 1
    ) -> list[dict[str, Any]]:
        if _is_analysis_call(schema):
            return [_ANALYSIS]
        props = sorted(schema["items"]["properties"].keys())
        seen.append(props)
        return [{p: f"{p}{i}" for p in props} for i in range(count)]

    client = AsyncMock()
    client.generate_structured = AsyncMock(side_effect=gen)
    with patch_clients(client):
        await generate_dataset({Customer: 3, Order: 3}, seed=7)

    # Customer only needs 'name' from the LLM; Order has no LLM fields at all
    # (id is PK, customer_id is FK, amount is a distribution).
    assert ["name"] in seen
    assert all("id" not in props and "customer_id" not in props for props in seen)


@pytest.mark.asyncio
async def test_foreign_key_context_passed_to_llm() -> None:
    """A child's LLM prompt includes the attributes of its referenced parent."""

    class Review(BaseModel):
        id: Annotated[int, PrimaryKey()]
        customer_id: Annotated[int, ForeignKey(Customer)]
        comment: str  # LLM-generated -> triggers a prompt carrying FK context

    prompts: list[str] = []

    def gen(
        schema: dict[str, Any], prompt: str, count: int = 1
    ) -> list[dict[str, Any]]:
        if _is_analysis_call(schema):
            return [_ANALYSIS]
        prompts.append(prompt)
        props = list(schema["items"]["properties"].keys())
        return [{p: f"{p}-{i}" for p in props} for i in range(count)]

    client = AsyncMock()
    client.generate_structured = AsyncMock(side_effect=gen)
    with patch_clients(client):
        await generate_dataset({Customer: 3, Review: 5}, seed=1)

    # The Review prompt (generating 'comment') should carry its referenced
    # customer's data under a "related_records.customer" block.
    review_prompts = [p for p in prompts if "Review model" in p]
    assert review_prompts
    assert all('"related_records":' in p for p in review_prompts)
    assert all('"customer":' in p for p in review_prompts)
    # The parent Customer prompt has no foreign keys, so no related_records data
    # is folded into its partial records (the JSON key is absent).
    customer_prompts = [p for p in prompts if "Customer model" in p]
    assert customer_prompts
    assert all('"related_records":' not in p for p in customer_prompts)


@pytest.mark.asyncio
async def test_uuid_primary_key_strategy() -> None:
    class Doc(BaseModel):
        id: Annotated[str, PrimaryKey(strategy="uuid")]
        title: str

    with patch_clients(make_fake_client()):
        dataset = await generate_dataset({Doc: 5}, seed=3)

    ids = [d.id for d in dataset[Doc]]
    assert len(set(ids)) == 5
    assert all(isinstance(i, str) and len(i) == 32 for i in ids)


def test_self_reference_via_string_forward_ref() -> None:
    """A model can reference itself using a string forward reference."""

    class Employee(BaseModel):
        id: Annotated[int, PrimaryKey()]
        manager_id: Annotated[
            int | None, ForeignKey("Employee", nullable=True, null_probability=0.3)
        ] = None

    with patch_clients(make_fake_client()):
        dataset = generate_dataset_sync({Employee: 15}, seed=4)

    employees = dataset[Employee]
    ids = {e.id for e in employees}
    managers = [e.manager_id for e in employees]
    assert any(m is None for m in managers)  # at least one top-level employee
    assert all(m in ids for m in managers if m is not None)  # valid references


@pytest.mark.asyncio
async def test_nullable_foreign_key() -> None:
    class Ticket(BaseModel):
        id: Annotated[int, PrimaryKey()]
        customer_id: Annotated[
            int | None, ForeignKey(Customer, nullable=True, null_probability=0.5)
        ] = None

    with patch_clients(make_fake_client()):
        dataset = await generate_dataset({Customer: 5, Ticket: 60}, seed=11)

    values = [t.customer_id for t in dataset[Ticket]]
    assert any(v is None for v in values)
    assert any(v is not None for v in values)
    customer_ids = {c.id for c in dataset[Customer]}
    assert all(v in customer_ids for v in values if v is not None)


@pytest.mark.asyncio
async def test_seed_reproducible() -> None:
    with patch_clients(make_fake_client()):
        a = await generate_dataset({Customer: 8, Order: 20}, seed=99)
    with patch_clients(make_fake_client()):
        b = await generate_dataset({Customer: 8, Order: 20}, seed=99)

    assert [o.customer_id for o in a[Order]] == [o.customer_id for o in b[Order]]


@pytest.mark.asyncio
async def test_missing_parent_raises() -> None:
    with patch_clients(make_fake_client()):
        with pytest.raises(ValueError, match="not included in the dataset"):
            await generate_dataset({Order: 5})  # Customer missing


def test_cyclic_dependency_raises() -> None:
    class A(BaseModel):
        id: Annotated[int, PrimaryKey()]

    class B(BaseModel):
        id: Annotated[int, PrimaryKey()]
        a_id: Annotated[int, ForeignKey(A)]

    # Introduce A -> B at the annotation level to form a cycle.
    A.__annotations__["b_id"] = Annotated[int, ForeignKey(B)]

    with pytest.raises(ValueError, match="[Cc]yclic"):
        _resolve_generation_order([A, B])


def test_sync_wrapper_and_to_dataframes() -> None:
    with patch_clients(make_fake_client()):
        dataset = generate_dataset_sync({Customer: 4, Order: 6}, seed=5)

    frames = dataset.to_dataframes()
    assert set(frames) == {"Customer", "Order"}
    assert len(frames["Customer"]) == 4
    assert len(frames["Order"]) == 6
