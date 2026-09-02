"""Shared helpers for the db binding: engine coercion and naming."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Union

try:
    from sqlalchemy import create_engine
    from sqlalchemy.engine import Engine
except ImportError as exc:  # pragma: no cover - exercised via message assertion
    raise ImportError(
        "gendantic.db requires the 'db' extra. Install it with: "
        "pip install 'gendantic[db]'"
    ) from exc

if TYPE_CHECKING:
    EngineSource = Union[str, Engine]


def as_engine(source: "str | Engine") -> Engine:
    """Coerce a database URL or an existing Engine into an Engine."""
    if isinstance(source, str):
        return create_engine(source)
    return source


def camel_case(name: str) -> str:
    """Turn a snake_case table name into a CamelCase model name.

    ``order_items`` -> ``OrderItems``. Leading digits are prefixed with ``T`` so
    the result is a valid identifier.
    """
    parts = [p for p in name.replace(" ", "_").split("_") if p]
    camel = "".join(word[:1].upper() + word[1:] for word in parts) or "Table"
    if camel[0].isdigit():
        camel = "T" + camel
    return camel


def qualified_name(schema: str | None, table_name: str) -> str:
    """Build the key SQLAlchemy uses for a table in ``MetaData.tables``."""
    return f"{schema}.{table_name}" if schema else table_name


def table_name_of(model: Any) -> str:
    """Return the source table name recorded on a reflected model."""
    return str(getattr(model, "__tablename__", model.__name__.lower()))
