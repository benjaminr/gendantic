"""Relational generation quickstart (runs offline, no LLM proxy required).

Every non-key field here is backed by a statistical distribution, so gendantic
samples the whole dataset with numpy and never calls an LLM. This makes the
example fully runnable out of the box while still demonstrating referential
integrity across three related tables:

    Customer  1--*  Order  *--1  Product

Run it with:

    uv run python examples/relational_quickstart.py
"""

from typing import Annotated

from pydantic import BaseModel, Field

from gendantic import (
    Categorical,
    ForeignKey,
    LogNormal,
    Normal,
    Poisson,
    PrimaryKey,
    generate_dataset_sync,
)


class Customer(BaseModel):
    id: Annotated[int, PrimaryKey()]
    tier: Annotated[
        str,
        Categorical(weights={"free": 0.6, "pro": 0.3, "enterprise": 0.1}),
    ]
    lifetime_value: Annotated[float, LogNormal(mean=6.0, sigma=1.0)]


class Product(BaseModel):
    id: Annotated[str, PrimaryKey(strategy="uuid")]
    price: Annotated[float, Normal(mean=40.0, std=12.0)] = Field(ge=1.0)


class Order(BaseModel):
    id: Annotated[str, PrimaryKey(strategy="uuid")]
    customer_id: Annotated[int, ForeignKey(Customer)]
    product_id: Annotated[str, ForeignKey(Product)]
    quantity: Annotated[int, Poisson(lam=3)] = Field(ge=1)


def main() -> None:
    dataset = generate_dataset_sync(
        {Customer: 20, Product: 8, Order: 100},
        seed=42,
    )

    customers = dataset[Customer]
    products = dataset[Product]
    orders = dataset[Order]

    print(
        f"Generated {len(customers)} customers, "
        f"{len(products)} products, {len(orders)} orders\n"
    )

    # Referential integrity: every order points at a real customer and product.
    customer_ids = {c.id for c in customers}
    product_ids = {p.id for p in products}
    assert all(o.customer_id in customer_ids for o in orders)
    assert all(o.product_id in product_ids for o in orders)
    assert len(customer_ids) == len(customers)  # unique primary keys
    print("Referential integrity verified: all foreign keys resolve.\n")

    print("Sample orders:")
    for order in orders[:5]:
        print(
            f"  order {order.id[:8]}… "
            f"customer={order.customer_id} "
            f"product={order.product_id[:8]}… "
            f"qty={order.quantity}"
        )


if __name__ == "__main__":
    main()
