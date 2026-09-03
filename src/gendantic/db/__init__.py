"""Database bindings: reflect a schema into models, load generated data back.

Requires the ``db`` extra:

    pip install 'gendantic[db]'

Typical round-trip:

    from gendantic.db import reflect_schema, load_dataset
    from gendantic import generate_dataset_sync

    models = reflect_schema("postgresql+psycopg://user:pw@host/shop")
    dataset = generate_dataset_sync(
        {models["customers"]: 100, models["orders"]: 500}, seed=42
    )
    load_dataset(dataset, "postgresql+psycopg://user:pw@host/shop")
"""

# Single guard for the whole binding: Python runs this package __init__ before
# importing any db submodule (whichever import path is used), so a missing 'db'
# extra becomes one clear message rather than a raw "No module named sqlalchemy".
try:
    import sqlalchemy as _sqlalchemy  # noqa: F401  (imported only to fail fast)
except ImportError as exc:  # pragma: no cover - only hit without the 'db' extra
    raise ImportError(
        "gendantic.db requires the 'db' extra. Install it with: "
        "pip install 'gendantic[db]'"
    ) from exc

from .infer import infer_distributions
from .load import load_dataset
from .reflect import reflect_schema

__all__ = [
    "reflect_schema",
    "load_dataset",
    "infer_distributions",
]
