"""Load a generated :class:`~gendantic.Dataset` back into a database."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import MetaData, func, select

from ..relational import Dataset, _primary_key_columns, _resolve_generation_order
from ._engine import as_engine, qualified_name, table_name_of

if TYPE_CHECKING:
    from sqlalchemy import Table
    from sqlalchemy.engine import Connection, Engine


def load_dataset(
    dataset: Dataset,
    source: "str | Engine",
    *,
    schema: str | None = None,
    reset_sequences: bool = True,
) -> dict[str, int]:
    """Bulk-insert a generated dataset into an existing database.

    Rows are inserted parent-first (topologically sorted by foreign keys) so
    referential integrity holds. The primary-key values gendantic generated are
    inserted explicitly, keeping foreign keys consistent.

    Args:
        dataset: The dataset produced by :func:`gendantic.generate_dataset`.
        source: A SQLAlchemy database URL, or an already-created ``Engine``.
        schema: Optional schema the target tables live in.
        reset_sequences: On PostgreSQL, advance each table's identity sequence
            past the inserted primary keys so future inserts don't collide.

    Returns:
        A mapping of table name to the number of rows inserted.
    """
    engine = as_engine(source)
    order = _resolve_generation_order(list(dataset))

    metadata = MetaData()
    metadata.reflect(bind=engine, schema=schema)

    is_postgres = engine.dialect.name == "postgresql"
    inserted: dict[str, int] = {}
    with engine.begin() as conn:
        for model in order:
            records = dataset[model]
            if not records:
                continue
            table_name = table_name_of(model)
            table = metadata.tables[qualified_name(schema, table_name)]
            rows = [record.model_dump() for record in records]
            conn.execute(table.insert(), rows)
            inserted[table_name] = len(rows)

            if reset_sequences and is_postgres:
                _reset_sequences(conn, model, table, schema)

    return inserted


def _reset_sequences(
    conn: "Connection",
    model: type,
    table: "Table",
    schema: str | None,
) -> None:
    """Advance identity sequences on integer primary keys past inserted rows.

    Built from SQLAlchemy expression constructs (not raw SQL) so the table and
    column identifiers are handled safely.
    """
    table_ident = f"{schema}.{table.name}" if schema else table.name
    for pk_column in _primary_key_columns(model):
        if pk_column.base_type is not int:
            continue
        col = table.c[pk_column.field]
        sequence = func.pg_get_serial_sequence(table_ident, pk_column.field)
        current_max = select(func.coalesce(func.max(col), 1)).scalar_subquery()
        conn.execute(select(func.setval(sequence, current_max)))
