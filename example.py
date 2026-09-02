"""
Gendantic Example - Intelligent Synthetic Data Generation

This example demonstrates:
1. Statistical distributions with Annotated types
2. LLM-generated semantic fields
3. Dynamic model generation from descriptions
4. Reproducibility with seeds
5. Context-aware generation
"""

import asyncio
from typing import Annotated, Optional

from pydantic import BaseModel, Field

from gendantic import (
    Beta,
    Categorical,
    ForeignKey,
    LogNormal,
    Normal,
    Poisson,
    PrimaryKey,
    Uniform,
    generate_dataset,
    generate_model_from_description,
    generate_synthetic_data,
    generate_synthetic_data_batch,
)


# Example 1: Model with statistical distributions
class Employee(BaseModel):
    """Employee with mixed distribution-sampled and LLM-generated fields."""

    # LLM-generated fields (semantic content)
    first_name: str = Field(min_length=2, max_length=30)
    last_name: str = Field(min_length=2, max_length=30)
    job_title: str
    bio: str = Field(description="Brief professional bio")

    # Distribution-sampled fields (statistical guarantees)
    age: Annotated[int, Uniform(min=22, max=65)]
    salary: Annotated[float, Normal(mean=75000, std=20000)] = Field(ge=30000)
    years_experience: Annotated[int, Uniform(min=0, max=40)]
    department: Annotated[
        str,
        Categorical(
            weights={
                "Engineering": 0.35,
                "Product": 0.20,
                "Sales": 0.20,
                "Marketing": 0.15,
                "HR": 0.10,
            }
        ),
    ]
    performance_rating: Annotated[float, Beta(alpha=5, beta=2)] = Field(
        ge=0.0, le=1.0, description="Performance score 0-1"
    )


# Example 2: Sales data with various distributions
class SalesRecord(BaseModel):
    """Sales record demonstrating different distribution types."""

    # LLM generates realistic content
    customer_name: str
    product_name: str
    notes: Optional[str] = None

    # Numpy samples from distributions
    deal_value: Annotated[float, LogNormal(mean=9, sigma=1.5)]  # Right-skewed
    units_sold: Annotated[int, Poisson(lam=5)]  # Count data
    discount_pct: Annotated[float, Beta(alpha=2, beta=8)]  # Mostly low discounts
    region: Annotated[
        str,
        Categorical(weights={"North": 0.3, "South": 0.25, "East": 0.25, "West": 0.2}),
    ]


async def demo_distributions():
    """Demonstrate statistical distribution sampling."""
    print("=" * 60)
    print("1. Statistical Distributions")
    print("=" * 60)
    print("\nGenerating 5 employees with distribution-sampled fields...")
    print("(salary: Normal, age: Uniform, department: Categorical)\n")

    employees = await generate_synthetic_data(Employee, count=5, seed=42)

    for emp in employees:
        print(f"{emp.first_name} {emp.last_name}")
        print(f"  {emp.job_title} | {emp.department}")
        print(f"  Age: {emp.age} | Experience: {emp.years_experience}y")
        print(f"  Salary: £{emp.salary:,.0f} | Rating: {emp.performance_rating:.2f}")
        print(f"  Bio: {emp.bio[:60]}...")
        print()


async def demo_reproducibility():
    """Demonstrate reproducible generation with seeds."""
    print("=" * 60)
    print("2. Reproducibility with Seeds")
    print("=" * 60)
    print("\nGenerating with seed=123 twice - distribution fields match:\n")

    batch1 = await generate_synthetic_data(Employee, count=3, seed=123)
    batch2 = await generate_synthetic_data(Employee, count=3, seed=123)

    for i, (e1, e2) in enumerate(zip(batch1, batch2, strict=False)):
        print(f"Record {i + 1}:")
        print(
            f"  Batch 1: age={e1.age}, salary=£{e1.salary:,.0f}, dept={e1.department}"
        )
        print(
            f"  Batch 2: age={e2.age}, salary=£{e2.salary:,.0f}, dept={e2.department}"
        )
        match = (
            e1.age == e2.age
            and e1.salary == e2.salary
            and e1.department == e2.department
        )
        print(f"  Match: {'Yes' if match else 'No'}")
        print()


async def demo_dynamic_model():
    """Demonstrate dynamic model generation from descriptions."""
    print("=" * 60)
    print("3. Dynamic Model Generation")
    print("=" * 60)
    print("\nGenerating a model from natural language description...\n")

    description = """
    A customer support ticket with:
    - Priority level (high, medium, low with realistic proportions)
    - Category (billing, technical, general inquiry)
    - Customer satisfaction score (0-10)
    - Resolution time in hours
    - A description of the issue
    """

    Model, source_code = await generate_model_from_description(
        description, model_name="SupportTicket"
    )

    print("Generated model code:")
    print("-" * 40)
    print(source_code)
    print("-" * 40)

    print("\nGenerating 3 tickets using the model:\n")
    tickets = await generate_synthetic_data(Model, count=3)

    for i, ticket in enumerate(tickets, 1):
        print(f"Ticket {i}:")
        for field_name in Model.model_fields:
            value = getattr(ticket, field_name)
            if isinstance(value, float):
                print(f"  {field_name}: {value:.2f}")
            else:
                value_str = (
                    str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
                )
                print(f"  {field_name}: {value_str}")
        print()


async def demo_context_aware():
    """Demonstrate context-aware generation."""
    print("=" * 60)
    print("4. Context-Aware Generation")
    print("=" * 60)
    print("\nSame model, different contexts produce different realistic data:\n")

    contexts = [
        "Silicon Valley AI startup",
        "Traditional London investment bank",
    ]

    for context in contexts:
        print(f"Context: {context}")
        print("-" * 40)

        employees = await generate_synthetic_data(Employee, count=2, context=context)

        for emp in employees:
            print(f"  {emp.first_name} {emp.last_name} - {emp.job_title}")
            print(f"  £{emp.salary:,.0f} | {emp.department}")
        print()


async def demo_batch_generation():
    """Demonstrate batch generation across contexts."""
    print("=" * 60)
    print("5. Batch Generation")
    print("=" * 60)
    print("\nGenerating employees for multiple offices concurrently:\n")

    contexts = [
        "New York headquarters",
        "London office",
        "Tokyo branch",
    ]

    batches = await generate_synthetic_data_batch(Employee, contexts, count=2, seed=42)

    for context, employees in zip(contexts, batches, strict=False):
        print(f"{context}:")
        for emp in employees:
            print(
                f"  {emp.first_name} {emp.last_name} - {emp.job_title} (£{emp.salary:,.0f})"
            )
        print()


# Example 3: Related models for relational generation
class Store(BaseModel):
    """A retail store (parent table)."""

    id: Annotated[int, PrimaryKey()]
    name: str  # LLM-generated
    region: Annotated[
        str,
        Categorical(weights={"North": 0.3, "South": 0.25, "East": 0.25, "West": 0.2}),
    ]


class Purchase(BaseModel):
    """A purchase made at a store (child table with a foreign key)."""

    id: Annotated[str, PrimaryKey(strategy="uuid")]
    store_id: Annotated[int, ForeignKey(Store)]
    item: str  # LLM-generated
    amount: Annotated[float, LogNormal(mean=3.0, sigma=0.6)]


async def demo_relational():
    """Demonstrate multi-model generation with referential integrity."""
    print("=" * 60)
    print("6. Relational Generation (foreign keys)")
    print("=" * 60)
    print("\nGenerating related Stores and Purchases with valid foreign keys:\n")

    dataset = await generate_dataset({Store: 5, Purchase: 20}, seed=42)
    stores = dataset[Store]
    purchases = dataset[Purchase]

    stores_by_id = {s.id: s for s in stores}
    # Referential integrity holds by construction:
    assert all(p.store_id in stores_by_id for p in purchases)

    print(f"Generated {len(stores)} stores and {len(purchases)} purchases.")
    print("Every purchase references a real store:\n")
    for purchase in purchases[:5]:
        store = stores_by_id[purchase.store_id]
        print(
            f"  {purchase.item} (£{purchase.amount:,.2f}) "
            f"@ {store.name} [{store.region}]"
        )
    print()


async def main():
    """Run all demonstrations."""
    print("\nGendantic - Intelligent Synthetic Data Generation")
    print("=" * 60)
    print()

    try:
        await demo_distributions()
        await demo_reproducibility()
        await demo_dynamic_model()
        await demo_context_aware()
        await demo_batch_generation()
        await demo_relational()

        print("=" * 60)
        print("All demonstrations complete!")
        print("=" * 60)

    except ValueError as e:
        print(f"\nError: {e}")
        print("\nTo run this example, point gendantic at a LiteLLM proxy:")
        print("  export LITELLM_API_BASE='http://localhost:4000'")
        print("  export LITELLM_MODEL='gpt-4o-mini'")


if __name__ == "__main__":
    asyncio.run(main())
