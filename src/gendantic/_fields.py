"""Field introspection shared by the sampler, relational engine and model tools.

gendantic attaches its markers (distribution specs, ``Conditional``,
``PrimaryKey``, ``ForeignKey``) to fields through ``Annotated``. Reading them
back off ``model_class.__annotations__`` only sees the annotations written on
that exact class, so markers declared on a parent model are missed, and it
sees raw strings under ``from __future__ import annotations``. Pydantic has
already resolved both of those cases by the time the class exists: every field
(inherited or not) appears in ``model_fields`` with its evaluated type on
``FieldInfo.annotation`` and the ``Annotated`` extras on ``FieldInfo.metadata``.
This module is the single place that reads them.
"""

from __future__ import annotations

import types
from collections.abc import Iterator
from typing import Annotated, Any, Union, get_args, get_origin

from pydantic import BaseModel


def unwrap_optional(tp: Any) -> Any:
    """Strip ``Optional``/``X | None`` from a type, returning the inner type.

    ``Optional[int]`` -> ``int``; ``int | None`` -> ``int``. Unions with more
    than one non-``None`` member, and everything else, are returned unchanged.
    """
    origin = get_origin(tp)
    if origin is Union or origin is types.UnionType:
        members = [a for a in get_args(tp) if a is not type(None)]
        if len(members) == 1:
            return members[0]
    return tp


def iter_fields(
    model_class: type[BaseModel],
) -> Iterator[tuple[str, Any, tuple[Any, ...]]]:
    """Yield ``(field_name, base_type, markers)`` for every field of a model.

    ``base_type`` is the field's declared type with any ``Optional`` wrapper
    removed (so ``Annotated[Optional[int], Uniform(...)]`` reports ``int`` and is
    sampled as an integer). ``markers`` are the ``Annotated`` extras: gendantic
    specs and keys, plus Pydantic's own constraint objects (``Ge``, ``Le``, ...),
    which callers filter with ``isinstance``. Both the outer form
    ``Annotated[Optional[T], marker]`` and the inner form
    ``Optional[Annotated[T, marker]]`` are recognised.

    Fields are yielded in Pydantic's field order, which includes fields
    inherited from parent models.
    """
    for field_name, field_info in model_class.model_fields.items():
        base_type: Any = field_info.annotation
        markers: list[Any] = list(field_info.metadata)

        inner = unwrap_optional(base_type)
        if get_origin(inner) is Annotated:
            # ``Optional[Annotated[T, ...]]``: Pydantic leaves the inner
            # Annotated intact, so lift its metadata out ourselves.
            args = get_args(inner)
            inner = args[0]
            markers.extend(args[1:])
        yield field_name, inner, tuple(markers)


def first_marker(markers: tuple[Any, ...], marker_type: type | tuple[type, ...]) -> Any:
    """Return the first marker that is an instance of ``marker_type``, or None."""
    for marker in markers:
        if isinstance(marker, marker_type):
            return marker
    return None
