"""Relational generation with **LLM-backed** semantic fields (live proxy).

Unlike ``relational_quickstart.py`` (which is fully offline), this example calls
a real LLM for the semantic fields — customer names, product names and review
text — while numpy still samples the statistical fields and the engine still
guarantees referential integrity across the three related tables:

    Customer  1--*  Review  *--1  Product

It needs a LiteLLM proxy. Configure it with the standard environment variables
before running:

    export LITELLM_API_BASE="https://your-litellm-proxy/v1"
    export LITELLM_API_KEY="your-proxy-key"        # if the proxy requires auth
    export LITELLM_MODEL="openai/gpt-4o-mini"       # openai/ prefix routes the
                                                    # SDK at an OpenAI-compatible
                                                    # proxy endpoint

Run it with:

    uv run python examples/relational_llm.py
"""

import os
import sys
from typing import Annotated

from pydantic import BaseModel, Field

from gendantic import (
    Categorical,
    ForeignKey,
    LogNormal,
    Normal,
    PrimaryKey,
    generate_dataset_sync,
)


class Customer(BaseModel):
    """A retail customer."""

    id: Annotated[int, PrimaryKey()]
    name: str  # LLM-generated
    tier: Annotated[
        str, Categorical(weights={"free": 0.6, "pro": 0.3, "enterprise": 0.1})
    ]
    lifetime_value: Annotated[float, LogNormal(mean=6.0, sigma=1.0)]


class Product(BaseModel):
    """A catalogue product."""

    id: Annotated[str, PrimaryKey(strategy="uuid")]
    name: str  # LLM-generated
    category: Annotated[
        str,
        Categorical(weights={"Home": 0.4, "Electronics": 0.35, "Outdoors": 0.25}),
    ]
    price: Annotated[float, Normal(mean=40.0, std=12.0)] = Field(ge=1.0)


class Review(BaseModel):
    """A customer's review of a product."""

    id: Annotated[int, PrimaryKey()]
    customer_id: Annotated[int, ForeignKey(Customer)]
    product_id: Annotated[str, ForeignKey(Product)]
    rating: Annotated[
        int,
        Categorical(weights={"1": 0.1, "2": 0.1, "3": 0.2, "4": 0.3, "5": 0.3}),
    ]
    comment: str  # LLM-generated, coherent with the sampled rating


def main() -> None:
    if not os.environ.get("LITELLM_API_BASE"):
        sys.exit(
            "No LiteLLM proxy configured. Set LITELLM_API_BASE (and, if needed, "
            "LITELLM_API_KEY / LITELLM_MODEL) before running this example."
        )

    print(
        f"Using proxy {os.environ['LITELLM_API_BASE']} "
        f"model {os.environ.get('LITELLM_MODEL', '(default)')}\n"
    )

    dataset = generate_dataset_sync(
        {Customer: 6, Product: 5, Review: 12},
        seed=42,
        context="A UK outdoor-and-home retailer",
    )

    customers = dataset[Customer]
    products = dataset[Product]
    reviews = dataset[Review]

    print(
        f"Generated {len(customers)} customers, "
        f"{len(products)} products, {len(reviews)} reviews\n"
    )

    # Referential integrity holds even though names/comments came from the LLM.
    customer_ids = {c.id for c in customers}
    product_ids = {p.id for p in products}
    assert all(r.customer_id in customer_ids for r in reviews)
    assert all(r.product_id in product_ids for r in reviews)
    print("Referential integrity verified: all foreign keys resolve.\n")

    by_customer = {c.id: c for c in customers}
    by_product = {p.id: p for p in products}
    print("Sample reviews:")
    for r in reviews[:5]:
        customer = by_customer[r.customer_id]
        product = by_product[r.product_id]
        print(f"  {customer.name} rated {product.name!r} {r.rating}/5: {r.comment}")


if __name__ == "__main__":
    main()
