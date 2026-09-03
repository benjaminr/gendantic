"""Offline tests for relational (multi-model) generation with a mocked client.

The mocked LLM seam is provided by the ``make_client`` / ``patch_clients``
fixtures in conftest.py; ``fake_values`` below is the default field-generation
callback returning ``"<prop>-<i>"`` for every requested property.
"""

from typing import Annotated, Any

import pytest
from pydantic import BaseModel, Field

from gendantic import (
    ForeignKey,
    ForeignKeySpec,
    Normal,
    PrimaryKey,
    generate_dataset,
    generate_dataset_sync,
)
from gendantic.relational import _resolve_generation_order


def fake_values(
    schema: dict[str, Any], prompt: str, count: int
) -> list[dict[str, Any]]:
    """Return LLM-field values for whatever properties the schema requests."""
    props = list(schema["items"]["properties"].keys())
    return [{p: f"{p}-{i}" for p in props} for i in range(count)]


class Customer(BaseModel):
    id: Annotated[int, PrimaryKey()]
    name: str


class Order(BaseModel):
    id: Annotated[int, PrimaryKey()]
    customer_id: Annotated[int, ForeignKey(Customer)]
    amount: Annotated[float, Normal(mean=200, std=50)]


@pytest.mark.asyncio
async def test_referential_integrity_and_counts(make_client, patch_clients) -> None:
    with patch_clients(make_client(fake_values)):
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
async def test_generation_order_is_topological(make_client, patch_clients) -> None:
    """Child declared first still generates after its parent."""
    with patch_clients(make_client(fake_values)):
        dataset = await generate_dataset({Order: 20, Customer: 5}, seed=1)

    customer_ids = {c.id for c in dataset[Customer]}
    assert all(o.customer_id in customer_ids for o in dataset[Order])


@pytest.mark.asyncio
async def test_primary_keys_not_asked_of_llm(make_client, patch_clients) -> None:
    seen: list[list[str]] = []

    def fields(schema: dict[str, Any], prompt: str, count: int) -> list[dict[str, Any]]:
        props = sorted(schema["items"]["properties"].keys())
        seen.append(props)
        return [{p: f"{p}{i}" for p in props} for i in range(count)]

    with patch_clients(make_client(fields)):
        await generate_dataset({Customer: 3, Order: 3}, seed=7)

    # Customer only needs 'name' from the LLM; Order has no LLM fields at all
    # (id is PK, customer_id is FK, amount is a distribution).
    assert ["name"] in seen
    assert all("id" not in props and "customer_id" not in props for props in seen)


@pytest.mark.asyncio
async def test_relational_validation_failure_raises(make_client, patch_clients) -> None:
    """A dropped row would break integrity, so relational generation raises.

    Unlike standalone generation (which tops up), relational rows carry
    engine-assigned keys that cannot be regenerated, so a validation failure is
    fatal rather than silently returning a short, integrity-broken table.
    """

    class Tagged(BaseModel):
        id: Annotated[int, PrimaryKey()]
        label: str = Field(min_length=3)  # LLM field with a constraint

    def too_short(
        schema: dict[str, Any], prompt: str, count: int
    ) -> list[dict[str, Any]]:
        return [{"label": "x"} for _ in range(count)]

    with patch_clients(make_client(too_short)):
        with pytest.raises(ValueError, match="referential integrity"):
            await generate_dataset({Tagged: 5}, seed=1)


@pytest.mark.asyncio
async def test_foreign_key_context_passed_to_llm(make_client, patch_clients) -> None:
    """A child's LLM prompt includes the attributes of its referenced parent."""

    class Review(BaseModel):
        id: Annotated[int, PrimaryKey()]
        customer_id: Annotated[int, ForeignKey(Customer)]
        comment: str  # LLM-generated -> triggers a prompt carrying FK context

    prompts: list[str] = []

    def fields(schema: dict[str, Any], prompt: str, count: int) -> list[dict[str, Any]]:
        prompts.append(prompt)
        props = list(schema["items"]["properties"].keys())
        return [{p: f"{p}-{i}" for p in props} for i in range(count)]

    with patch_clients(make_client(fields)):
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
async def test_uuid_primary_key_strategy(make_client, patch_clients) -> None:
    class Doc(BaseModel):
        id: Annotated[str, PrimaryKey(strategy="uuid")]
        title: str

    with patch_clients(make_client(fake_values)):
        dataset = await generate_dataset({Doc: 5}, seed=3)

    ids = [d.id for d in dataset[Doc]]
    assert len(set(ids)) == 5
    assert all(isinstance(i, str) and len(i) == 32 for i in ids)


def test_self_reference_via_string_forward_ref(make_client, patch_clients) -> None:
    """A model can reference itself using a string forward reference."""

    class Employee(BaseModel):
        id: Annotated[int, PrimaryKey()]
        manager_id: Annotated[
            int | None, ForeignKey("Employee", nullable=True, null_probability=0.3)
        ] = None

    with patch_clients(make_client(fake_values)):
        dataset = generate_dataset_sync({Employee: 15}, seed=4)

    employees = dataset[Employee]
    ids = {e.id for e in employees}
    managers = [e.manager_id for e in employees]
    assert any(m is None for m in managers)  # at least one top-level employee
    assert all(m in ids for m in managers if m is not None)  # valid references


@pytest.mark.asyncio
async def test_nullable_foreign_key(make_client, patch_clients) -> None:
    class Ticket(BaseModel):
        id: Annotated[int, PrimaryKey()]
        customer_id: Annotated[
            int | None, ForeignKey(Customer, nullable=True, null_probability=0.5)
        ] = None

    with patch_clients(make_client(fake_values)):
        dataset = await generate_dataset({Customer: 5, Ticket: 60}, seed=11)

    values = [t.customer_id for t in dataset[Ticket]]
    assert any(v is None for v in values)
    assert any(v is not None for v in values)
    customer_ids = {c.id for c in dataset[Customer]}
    assert all(v in customer_ids for v in values if v is not None)


@pytest.mark.asyncio
async def test_seed_reproducible(make_client, patch_clients) -> None:
    with patch_clients(make_client(fake_values)):
        a = await generate_dataset({Customer: 8, Order: 20}, seed=99)
    with patch_clients(make_client(fake_values)):
        b = await generate_dataset({Customer: 8, Order: 20}, seed=99)

    assert [o.customer_id for o in a[Order]] == [o.customer_id for o in b[Order]]


@pytest.mark.asyncio
async def test_missing_parent_raises(make_client, patch_clients) -> None:
    with patch_clients(make_client(fake_values)):
        with pytest.raises(ValueError, match="not included in the dataset"):
            await generate_dataset({Order: 5})  # Customer missing


def test_cyclic_dependency_raises() -> None:
    # A -> B via a string forward reference (B is not yet defined), B -> A
    # directly: together they form a cycle no generation order can satisfy.
    class A(BaseModel):
        id: Annotated[int, PrimaryKey()]
        b_id: Annotated[int, ForeignKey("B")]

    class B(BaseModel):
        id: Annotated[int, PrimaryKey()]
        a_id: Annotated[int, ForeignKey(A)]

    with pytest.raises(ValueError, match="[Cc]yclic"):
        _resolve_generation_order([A, B])


@pytest.mark.asyncio
async def test_composite_primary_key_join_table(make_client, patch_clients) -> None:
    """A join table whose PK is its two foreign keys stays referentially sound."""

    class Prod(BaseModel):
        id: Annotated[int, PrimaryKey()]

    class OrderItem(BaseModel):
        order_id: Annotated[int, ForeignKey(Customer)]
        product_id: int
        quantity: Annotated[int, Normal(mean=2, std=1)]
        __primary_key__ = ("order_id", "product_id")
        __foreign_keys__ = [
            ForeignKeySpec(columns="product_id", model="Prod"),
        ]

    with patch_clients(make_client(fake_values)):
        dataset = await generate_dataset({Customer: 5, Prod: 4, OrderItem: 12}, seed=1)

    items = dataset[OrderItem]
    customer_ids = {c.id for c in dataset[Customer]}
    product_ids = {p.id for p in dataset[Prod]}
    assert len(items) == 12
    # Referential integrity on both key columns.
    assert all(i.order_id in customer_ids for i in items)
    assert all(i.product_id in product_ids for i in items)
    # The composite primary key (order_id, product_id) is unique per row.
    pairs = [(i.order_id, i.product_id) for i in items]
    assert len(set(pairs)) == len(pairs)


@pytest.mark.asyncio
async def test_composite_key_caps_at_available_combinations(
    make_client, patch_clients
) -> None:
    """Requesting more join rows than distinct combinations caps the count."""

    class Prod(BaseModel):
        id: Annotated[int, PrimaryKey()]

    class OrderItem(BaseModel):
        order_id: Annotated[int, ForeignKey(Customer)]
        product_id: int
        __primary_key__ = ("order_id", "product_id")
        __foreign_keys__ = [ForeignKeySpec(columns="product_id", model="Prod")]

    with patch_clients(make_client(fake_values)):
        # 2 customers x 2 products = 4 possible distinct pairs, but 10 requested.
        dataset = await generate_dataset({Customer: 2, Prod: 2, OrderItem: 10}, seed=2)

    items = dataset[OrderItem]
    assert len(items) == 4  # capped at the number of distinct combinations
    pairs = [(i.order_id, i.product_id) for i in items]
    assert len(set(pairs)) == 4


@pytest.mark.asyncio
async def test_composite_foreign_key_to_composite_primary_key(
    make_client, patch_clients
) -> None:
    """A child can reference a parent's composite primary key across two columns."""

    class Building(BaseModel):
        site: Annotated[int, PrimaryKey()]
        floor: Annotated[int, PrimaryKey()]

    class Room(BaseModel):
        id: Annotated[int, PrimaryKey()]
        b_site: int
        b_floor: int
        __foreign_keys__ = [
            ForeignKeySpec(
                columns=("b_site", "b_floor"),
                model="Building",
                references=("site", "floor"),
            )
        ]

    with patch_clients(make_client(fake_values)):
        dataset = await generate_dataset({Building: 6, Room: 20}, seed=3)

    buildings = {(b.site, b.floor) for b in dataset[Building]}
    rooms = dataset[Room]
    assert len(rooms) == 20
    assert all((r.b_site, r.b_floor) in buildings for r in rooms)
    assert len({r.id for r in rooms}) == 20  # surrogate PK still unique


def test_sync_wrapper_and_to_dataframes(make_client, patch_clients) -> None:
    with patch_clients(make_client(fake_values)):
        dataset = generate_dataset_sync({Customer: 4, Order: 6}, seed=5)

    frames = dataset.to_dataframes()
    assert set(frames) == {"Customer", "Order"}
    assert len(frames["Customer"]) == 4
    assert len(frames["Order"]) == 6
