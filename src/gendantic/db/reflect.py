"""Reflect a live database schema into gendantic-annotated Pydantic models.

``reflect_schema`` introspects an existing database with SQLAlchemy and produces
one Pydantic model per table, annotated so it drops straight into
:func:`gendantic.generate_dataset`:

- primary keys become ``PrimaryKey`` (single-column) or ``__primary_key__``
  (composite);
- foreign keys become ``ForeignKey`` (single-column) or ``ForeignKeySpec`` in
  ``__foreign_keys__`` (composite), preserving referential structure;
- enum columns become ``Categorical``; ``VARCHAR(n)`` gets a ``max_length``;
- nullable columns become ``Optional``; remaining columns are left for the LLM.

Example:
    from gendantic.db import reflect_schema
    from gendantic import generate_dataset_sync

    models = reflect_schema("postgresql+psycopg://user:pw@host/shop")
    dataset = generate_dataset_sync(
        {models["customers"]: 100, models["orders"]: 500}, seed=42
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Optional

from pydantic import BaseModel, Field, create_model

from ..distributions import Categorical
from ..relational import ForeignKey, ForeignKeySpec, PrimaryKey
from ._engine import as_engine, camel_case
from .types import enum_values, python_type_for, string_max_length

try:
    from sqlalchemy import MetaData
except ImportError as exc:  # pragma: no cover - exercised via message assertion
    raise ImportError(
        "gendantic.db requires the 'db' extra. Install it with: "
        "pip install 'gendantic[db]'"
    ) from exc

if TYPE_CHECKING:
    from sqlalchemy import Column, Table
    from sqlalchemy.engine import Engine


def reflect_schema(
    source: "str | Engine",
    *,
    schema: str | None = None,
    tables: list[str] | None = None,
) -> dict[str, type[BaseModel]]:
    """Reflect a database schema into gendantic-annotated Pydantic models.

    Args:
        source: A SQLAlchemy database URL, or an already-created ``Engine``.
        schema: Optional schema/namespace to reflect (e.g. ``"public"``).
        tables: Optional list of table names to restrict reflection to.

    Returns:
        A mapping of table name to its generated Pydantic model class. Foreign
        keys reference other models by class name, so the whole mapping can be
        passed straight to :func:`gendantic.generate_dataset`.
    """
    engine = as_engine(source)
    metadata = MetaData()
    metadata.reflect(bind=engine, schema=schema, only=tables)

    all_tables = list(metadata.tables.values())
    name_for_table = {table: camel_case(table.name) for table in all_tables}

    models: dict[str, type[BaseModel]] = {}
    for table in all_tables:
        models[table.name] = _build_model(table, name_for_table)
    return models


def _pk_strategy(column: "Column[Any]", py_type: type) -> str:
    """Choose a primary-key generation strategy for a reflected column."""
    if py_type is str:
        return "uuid"
    if py_type is int and column.autoincrement in (True, "auto"):
        return "sequential"
    return "auto"


def _build_model(
    table: "Table",
    name_for_table: "dict[Table, str]",
) -> type[BaseModel]:
    """Build one Pydantic model from a reflected table."""
    class_name = name_for_table[table]
    pk_names = [c.name for c in table.primary_key.columns]

    # Group foreign keys by constraint so composite keys stay together.
    single_fk: dict[str, tuple[str, str, bool]] = {}
    composite_fks: list[ForeignKeySpec] = []
    for constraint in table.foreign_key_constraints:
        local_cols = [c.name for c in constraint.columns]
        remote_cols = [element.column.name for element in constraint.elements]
        target_name = name_for_table[constraint.referred_table]
        nullable = any(table.columns[c].nullable for c in local_cols)
        if len(local_cols) == 1:
            single_fk[local_cols[0]] = (target_name, remote_cols[0], nullable)
        else:
            composite_fks.append(
                ForeignKeySpec(
                    columns=tuple(local_cols),
                    model=target_name,
                    references=tuple(remote_cols),
                    nullable=nullable,
                )
            )

    fields: dict[str, Any] = {}
    for column in table.columns:
        py_type = python_type_for(column)
        is_pk = column.name in pk_names
        markers: list[Any] = []

        if is_pk and len(pk_names) == 1:
            markers.append(PrimaryKey(strategy=_pk_strategy(column, py_type)))

        if column.name in single_fk:
            target_name, remote_col, nullable = single_fk[column.name]
            markers.append(
                ForeignKey(
                    target_name,
                    field=remote_col,
                    nullable=bool(column.nullable) and not is_pk,
                )
            )

        values = enum_values(column)
        if values and not markers:
            weight = 1.0 / len(values)
            markers.append(Categorical(weights=dict.fromkeys(values, weight)))

        # Keep the marker at the outermost Annotated level; represent nullability
        # via an inner Optional so gendantic's key extraction still sees it.
        core: Any = Optional[py_type] if (column.nullable and not is_pk) else py_type
        annotation: Any = Annotated[(core, *markers)] if markers else core

        default: Any = None if (column.nullable and not is_pk) else ...
        max_length = string_max_length(column) if not markers else None
        if max_length is not None:
            default = Field(default=default, max_length=max_length)

        fields[column.name] = (annotation, default)

    model: type[BaseModel] = create_model(class_name, **fields)
    model.__tablename__ = table.name  # type: ignore[attr-defined]
    if len(pk_names) > 1:
        model.__primary_key__ = tuple(pk_names)  # type: ignore[attr-defined]
    if composite_fks:
        model.__foreign_keys__ = composite_fks  # type: ignore[attr-defined]
    return model
