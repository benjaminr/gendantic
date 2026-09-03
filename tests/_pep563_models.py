"""Models declared under PEP 563 postponed evaluation of annotations.

With ``from __future__ import annotations`` every annotation on the class is a
*string* in ``__annotations__``; only Pydantic's ``model_fields`` holds the
evaluated types and ``Annotated`` metadata. Used by the field-introspection
tests to prove gendantic reads markers from the evaluated form.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel

from gendantic import ForeignKey, Normal, PrimaryKey, Uniform


class Team(BaseModel):
    id: Annotated[int, PrimaryKey()]
    size: Annotated[int, Uniform(min=2, max=12)]


class Member(BaseModel):
    id: Annotated[int, PrimaryKey()]
    team_id: Annotated[int, ForeignKey(Team)]
    score: Annotated[float, Normal(mean=50, std=10)]
    name: str
