"""Tests for the to_dataframe export helper."""

from pydantic import BaseModel

from gendantic import to_dataframe


class Row(BaseModel):
    name: str
    age: int
    score: float


def test_to_dataframe_shape_and_columns() -> None:
    records = [
        Row(name="Alice", age=30, score=1.5),
        Row(name="Bob", age=41, score=2.0),
    ]
    df = to_dataframe(records)

    assert list(df.columns) == ["name", "age", "score"]
    assert len(df) == 2
    assert df.loc[0, "name"] == "Alice"
    assert df.loc[1, "age"] == 41


def test_to_dataframe_empty() -> None:
    df = to_dataframe([])
    assert len(df) == 0


def test_to_dataframe_preserves_nested_values() -> None:
    class Nested(BaseModel):
        tags: list[str]

    df = to_dataframe([Nested(tags=["a", "b"])])
    assert df.loc[0, "tags"] == ["a", "b"]
