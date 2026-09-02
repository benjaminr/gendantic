"""SQL column type -> Python type mapping for schema reflection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sqlalchemy import Column


def python_type_for(column: "Column[Any]") -> type:
    """Best-effort Python type for a reflected SQLAlchemy column.

    SQLAlchemy type objects expose ``.python_type`` for the common cases
    (``int``, ``float``, ``str``, ``bool``, ``datetime``, ``Decimal``,
    ``uuid.UUID`` ...). Anything without a usable mapping falls back to ``str``,
    which the LLM can then fill in.
    """
    try:
        py = column.type.python_type
    except (NotImplementedError, AttributeError):
        return str
    return py if isinstance(py, type) else str


def enum_values(column: "Column[Any]") -> list[str] | None:
    """Return the allowed values for an ``Enum`` column, or ``None``.

    Covers SQLAlchemy's generic ``Enum`` as well as Postgres native enums, both
    of which expose an ``enums`` attribute on the type.
    """
    values = getattr(column.type, "enums", None)
    if values:
        return [str(v) for v in values]
    return None


def string_max_length(column: "Column[Any]") -> int | None:
    """Return the declared length of a ``VARCHAR(n)`` column, if any."""
    length = getattr(column.type, "length", None)
    return int(length) if isinstance(length, int) and length > 0 else None
