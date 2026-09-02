"""Helpers for exporting generated records to other formats."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    import pandas as pd


def to_dataframe(records: Sequence[BaseModel]) -> "pd.DataFrame":
    """Convert generated model instances into a pandas DataFrame.

    Each record becomes a row via ``model_dump()``; nested models and
    collections are kept as-is in the corresponding cell.

    Args:
        records: Model instances produced by ``generate_synthetic_data`` (or any
            sequence of Pydantic models).

    Returns:
        A ``pandas.DataFrame`` with one row per record.

    Raises:
        ImportError: If pandas is not installed. Install the optional extra with
            ``pip install 'gendantic[pandas]'``.

    Examples:
        employees = generate_synthetic_data_sync(Employee, count=100, seed=42)
        df = to_dataframe(employees)
    """
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover - exercised via message assertion
        raise ImportError(
            "to_dataframe requires pandas. Install it with: "
            "pip install 'gendantic[pandas]'"
        ) from e

    return pd.DataFrame([record.model_dump() for record in records])
