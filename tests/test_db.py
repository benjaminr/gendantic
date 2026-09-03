"""Offline tests for the database binding, exercised against SQLite.

SQLAlchemy makes the same reflection/insert code work on SQLite and PostgreSQL,
so these run without a real Postgres server. LLM fields are filled by a typed
mock so no network calls are made.
"""

from typing import Any

import pytest

sa = pytest.importorskip("sqlalchemy")

from sqlalchemy import (  # noqa: E402
    Column,
    Enum,
    Integer,
    MetaData,
    Numeric,
    String,
    Table,
    create_engine,
    func,
    select,
)
from sqlalchemy import ForeignKey as SAForeignKey  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from gendantic import generate_dataset_sync  # noqa: E402
from gendantic.db import infer_distributions, load_dataset, reflect_schema  # noqa: E402
from gendantic.db.reflect import _build_model  # noqa: E402
from gendantic.distributions import Categorical, Normal  # noqa: E402
from gendantic.relational import _primary_key_columns  # noqa: E402


def _typed_value(prop: dict[str, Any]) -> Any:
    """Return a value matching a JSON-schema property's declared type."""
    kind = prop.get("type")
    if kind is None and "anyOf" in prop:
        kinds = [s.get("type") for s in prop["anyOf"]]
        kind = next((k for k in kinds if k and k != "null"), "string")
    if kind == "integer":
        return 1
    if kind == "number":
        return 1.5
    if kind == "boolean":
        return True
    return "text"


def typed_values(schema: dict[str, Any], prompt: str, count: int) -> list[dict]:
    """Field-generation callback returning type-correct values per property."""
    props = schema["items"]["properties"]
    return [{name: _typed_value(p) for name, p in props.items()} for _ in range(count)]


def _make_engine() -> Any:
    """An in-memory SQLite engine that persists across connections."""
    return create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


def _create_shop_schema(engine: Any) -> None:
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
        Column("customer_id", Integer, SAForeignKey("customers.id"), nullable=False),
    )
    Table(
        "order_items",
        metadata,
        Column("order_id", Integer, SAForeignKey("orders.id"), primary_key=True),
        Column("product_id", Integer, SAForeignKey("products.id"), primary_key=True),
        Column("quantity", Integer),
    )
    metadata.create_all(engine)


def test_reflect_detects_keys_including_composite_join_table() -> None:
    engine = _make_engine()
    _create_shop_schema(engine)

    models = reflect_schema(engine)
    assert set(models) == {"customers", "products", "orders", "order_items"}

    # Single-column primary key.
    customer_pk = _primary_key_columns(models["customers"])
    assert [c.field for c in customer_pk] == ["id"]

    # Join table: composite primary key made of its two foreign keys.
    order_item = models["order_items"]
    assert order_item.__primary_key__ == ("order_id", "product_id")
    pk_fields = [c.field for c in _primary_key_columns(order_item)]
    assert pk_fields == ["order_id", "product_id"]


def test_round_trip_generate_and_load(make_client, patch_clients) -> None:
    engine = _make_engine()
    _create_shop_schema(engine)
    models = reflect_schema(engine)

    with patch_clients(make_client(typed_values)):
        dataset = generate_dataset_sync(
            {
                models["customers"]: 8,
                models["products"]: 5,
                models["orders"]: 15,
                models["order_items"]: 20,
            },
            seed=1,
        )

    inserted = load_dataset(dataset, engine)
    assert inserted == {
        "customers": 8,
        "products": 5,
        "orders": 15,
        "order_items": 20,
    }

    metadata = MetaData()
    metadata.reflect(bind=engine)
    customers = metadata.tables["customers"]
    orders = metadata.tables["orders"]
    order_items = metadata.tables["order_items"]
    products = metadata.tables["products"]

    with engine.connect() as conn:
        assert conn.execute(select(func.count()).select_from(customers)).scalar() == 8
        assert (
            conn.execute(select(func.count()).select_from(order_items)).scalar() == 20
        )

        customer_ids = set(conn.execute(select(customers.c.id)).scalars())
        product_ids = set(conn.execute(select(products.c.id)).scalars())
        order_ids = set(conn.execute(select(orders.c.id)).scalars())

        # Foreign keys in the loaded rows point at real parents.
        order_customer_ids = set(conn.execute(select(orders.c.customer_id)).scalars())
        assert order_customer_ids <= customer_ids

        item_order_ids = set(conn.execute(select(order_items.c.order_id)).scalars())
        item_product_ids = set(conn.execute(select(order_items.c.product_id)).scalars())
        assert item_order_ids <= order_ids
        assert item_product_ids <= product_ids


def test_infer_distributions_from_existing_data() -> None:
    engine = _make_engine()
    metadata = MetaData()
    readings = Table(
        "readings",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("sensor", String(20)),
        Column("value", Numeric(10, 2)),
    )
    metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(
            readings.insert(),
            [
                {"id": i, "sensor": "A" if i % 2 else "B", "value": 10.0 + i}
                for i in range(1, 21)
            ],
        )

    specs = infer_distributions(engine, "readings")

    assert "id" not in specs  # primary key skipped
    assert isinstance(specs["value"], Normal)  # numeric column -> Normal
    assert isinstance(specs["sensor"], Categorical)  # low-cardinality -> Categorical
    assert set(specs["sensor"].weights) == {"A", "B"}


def test_enum_and_length_mapping() -> None:
    """Enum columns become Categorical; VARCHAR(n) gets a max_length."""
    metadata = MetaData()
    widget = Table(
        "widgets",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("status", Enum("new", "active", "closed", name="status_enum")),
        Column("code", String(50)),
    )

    model = _build_model(widget, {widget: "Widget"})

    status_annotation = model.__annotations__["status"]
    markers = sa.util.to_list(status_annotation.__metadata__)
    categoricals = [m for m in markers if isinstance(m, Categorical)]
    assert categoricals, "enum column should map to a Categorical"
    assert set(categoricals[0].weights) == {"new", "active", "closed"}

    # VARCHAR(50) -> max_length constraint on the Pydantic field.
    assert model.model_fields["code"].metadata  # length constraint present
