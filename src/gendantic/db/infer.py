"""Infer distributions for a table by sampling its existing rows.

When a database already holds real data, fitting distributions to it makes the
synthetic data *statistically resemble* production rather than merely being
type-valid: numeric columns get a fitted ``Normal``, and low-cardinality columns
get a ``Categorical`` weighted by observed frequency.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any

from sqlalchemy import MetaData, select

from ..distributions import Categorical, DistributionSpec, Normal
from ._engine import as_engine, qualified_name
from .types import python_type_for

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

_NUMERIC = (int, float, Decimal)


def infer_distributions(
    source: "str | Engine",
    table: str,
    *,
    schema: str | None = None,
    sample_size: int = 5000,
    max_categories: int = 25,
) -> dict[str, DistributionSpec]:
    """Fit a distribution to each suitable column of a table from its data.

    Primary-key and foreign-key columns are skipped (their values come from
    referential generation, not distributions).

    Args:
        source: A SQLAlchemy database URL, or an already-created ``Engine``.
        table: Name of the table to sample.
        schema: Optional schema the table lives in.
        sample_size: Maximum number of rows to sample.
        max_categories: Columns with more distinct values than this are not
            treated as categorical.

    Returns:
        A mapping of column name to a fitted ``DistributionSpec``. Attach these
        to a model (e.g. via ``extend_model_with_distributions``) or build a
        model that references them.
    """
    engine = as_engine(source)
    metadata = MetaData()
    metadata.reflect(bind=engine, schema=schema, only=[table])
    tbl = metadata.tables[qualified_name(schema, table)]

    candidates = [
        column
        for column in tbl.columns
        if not column.primary_key and not column.foreign_keys
    ]
    if not candidates:
        return {}

    with engine.connect() as conn:
        rows = conn.execute(select(*candidates).limit(sample_size)).mappings().all()

    specs: dict[str, DistributionSpec] = {}
    for column in candidates:
        values = [row[column.name] for row in rows if row[column.name] is not None]
        if not values:
            continue
        spec = _fit_column(python_type_for(column), values, max_categories)
        if spec is not None:
            specs[column.name] = spec
    return specs


def _fit_column(
    py_type: type, values: list[Any], max_categories: int
) -> DistributionSpec | None:
    """Fit a single column's values to a Normal or Categorical, if sensible.

    Numeric columns are fitted to a ``Normal``; non-numeric columns with few
    distinct values become a ``Categorical`` weighted by observed frequency.
    High-cardinality text is left for the LLM (returns ``None``).
    """
    if py_type in _NUMERIC:
        numbers = [float(v) for v in values]
        mean = sum(numbers) / len(numbers)
        variance = sum((n - mean) ** 2 for n in numbers) / len(numbers)
        std = variance**0.5
        return Normal(mean=mean, std=std if std > 0 else 1.0)

    distinct = {str(v) for v in values}
    if len(distinct) <= max_categories:
        counts: dict[str, int] = {}
        for value in values:
            counts[str(value)] = counts.get(str(value), 0) + 1
        total = sum(counts.values())
        weights = {key: count / total for key, count in counts.items()}
        return Categorical(weights=weights)

    return None
