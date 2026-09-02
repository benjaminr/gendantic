"""Database round-trip: reflect a schema, generate data, load it back.

This example uses a throwaway **SQLite** database so it runs with no database
server, but the exact same code works against PostgreSQL — just pass a
``postgresql+psycopg://user:pw@host/db`` URL instead of the SQLite one.

It demonstrates the full round-trip:

1. ``reflect_schema`` turns the live tables into gendantic-annotated models,
   including a composite-key join table (``order_items``).
2. ``generate_dataset_sync`` produces synthetic rows with referential integrity;
   the LLM fills the semantic text columns (customer/product names).
3. ``load_dataset`` bulk-inserts the rows back, parent-first.

Needs a LiteLLM proxy for the text columns. Set the standard environment
variables before running:

    export LITELLM_API_BASE="https://your-litellm-proxy/v1"
    export LITELLM_API_KEY="your-proxy-key"        # if the proxy requires auth
    export LITELLM_MODEL="openai/gpt-4o-mini"

Run it with:

    uv run --extra db python examples/db_binding.py
"""

import os
import sys
import tempfile
from pathlib import Path

from sqlalchemy import (
    Column,
    ForeignKey,
    Integer,
    MetaData,
    Numeric,
    String,
    Table,
    create_engine,
    func,
    select,
)

from gendantic import generate_dataset_sync
from gendantic.db import load_dataset, reflect_schema


def build_schema(engine: object) -> None:
    """Create a small shop schema: customers, products, orders, order_items."""
    metadata = MetaData()
    Table(
        "customers",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("name", String(100)),
    )
    Table(
        "products",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("title", String(200)),
        Column("price", Numeric(10, 2)),
    )
    Table(
        "orders",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("customer_id", Integer, ForeignKey("customers.id"), nullable=False),
    )
    # Join table: the primary key IS its two foreign keys.
    Table(
        "order_items",
        metadata,
        Column("order_id", Integer, ForeignKey("orders.id"), primary_key=True),
        Column("product_id", Integer, ForeignKey("products.id"), primary_key=True),
        Column("quantity", Integer),
    )
    metadata.create_all(engine)


def main() -> None:
    if not os.environ.get("LITELLM_API_BASE"):
        sys.exit(
            "No LiteLLM proxy configured. Set LITELLM_API_BASE (and, if needed, "
            "LITELLM_API_KEY / LITELLM_MODEL) before running this example."
        )

    tmp = Path(tempfile.mkdtemp()) / "shop.db"
    url = f"sqlite:///{tmp}"
    engine = create_engine(url)
    build_schema(engine)
    print(f"Created SQLite database at {tmp}\n")

    # 1. Reflect the schema into gendantic-annotated Pydantic models.
    models = reflect_schema(engine)
    print("Reflected models:", ", ".join(sorted(models)))
    print(f"  order_items primary key: {models['order_items'].__primary_key__}\n")

    # 2. Generate synthetic rows with referential integrity.
    dataset = generate_dataset_sync(
        {
            models["customers"]: 10,
            models["products"]: 6,
            models["orders"]: 25,
            models["order_items"]: 40,
        },
        seed=42,
    )

    # 3. Load the data back into the database.
    inserted = load_dataset(dataset, engine)
    print("Inserted rows:", inserted, "\n")

    # Read a few rows back out with a join to prove it landed coherently.
    metadata = MetaData()
    metadata.reflect(bind=engine)
    customers = metadata.tables["customers"]
    orders = metadata.tables["orders"]
    with engine.connect() as conn:
        total = conn.execute(select(func.count()).select_from(orders)).scalar()
        print(f"{total} orders in the database. First five with their customer:")
        rows = conn.execute(
            select(orders.c.id, customers.c.name)
            .select_from(orders.join(customers, orders.c.customer_id == customers.c.id))
            .limit(5)
        ).all()
        for order_id, customer_name in rows:
            print(f"  order {order_id} -> {customer_name}")


if __name__ == "__main__":
    main()
