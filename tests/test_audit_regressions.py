"""Regression tests for defects found in the post-0.1.0 code audit.

Each test names the behaviour it protects; the mocked-LLM fixtures come from
conftest.py.
"""

from enum import Enum
from typing import Annotated, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, Field

import gendantic
from gendantic import (
    Categorical,
    Conditional,
    Normal,
    PrimaryKey,
    Range,
    Uniform,
    fidelity_report,
    generate_dataset,
    generate_synthetic_data,
)
from gendantic.llm import LiteLLMClient
from gendantic.llm_driven_analyser import LLMDrivenModelAnalyser
from gendantic.model_generator import (
    _execute_model_code,
    _get_basic_model_source_repr,
    _get_model_source_repr,
    _validate_code_safety,
)
from gendantic.sampler import DistributionSampler

# --- Packaging ---------------------------------------------------------------


def test_package_exposes_version() -> None:
    assert isinstance(gendantic.__version__, str)
    assert gendantic.__version__
    assert "__version__" in gendantic.__all__


# --- Generation input validation ---------------------------------------------


class Pure(BaseModel):
    x: Annotated[float, Normal(mean=0, std=1)]


async def test_negative_count_raises() -> None:
    with pytest.raises(ValueError, match="count must be >= 0"):
        await generate_synthetic_data(Pure, count=-1)


async def test_zero_count_returns_empty() -> None:
    assert await generate_synthetic_data(Pure, count=0) == []


# --- fidelity_report never raises for categorical fields ---------------------


class Dept(BaseModel):
    # Sums to 0.995: accepted by Categorical (0.01 tolerance) but not exactly 1.
    d: Annotated[str, Categorical(weights={"a": 0.5, "b": 0.495})]


def test_fidelity_categorical_with_approximate_weights_does_not_raise() -> None:
    specs = LLMDrivenModelAnalyser.extract_distribution_specs_with_types(Dept)
    rows = DistributionSampler(seed=1).sample_fields(specs, 1000)
    report = fidelity_report(rows, Dept)
    assert len(report.fields) == 1
    assert report.fields[0].passed


def test_fidelity_categorical_with_unknown_category_fails_not_raises() -> None:
    rows = [{"d": "a"}] * 300 + [{"d": "not-a-category"}] * 200
    report = fidelity_report(rows, Dept)
    assert not report.passed
    result = report.fields[0]
    assert result.test == "chi2"
    assert result.p_value == 0.0
    assert not result.passed


# --- $defs travel with the partial schema ------------------------------------


class Status(str, Enum):
    open = "open"
    closed = "closed"


class Ticket(BaseModel):
    priority: Annotated[int, Uniform(min=1, max=5)]
    status: Status  # LLM-generated; rendered as a $ref into $defs


async def test_partial_schema_carries_defs_for_enum_fields(
    make_client, patch_clients
) -> None:
    seen: list[dict[str, Any]] = []

    def values(schema: dict[str, Any], prompt: str, count: int) -> list[dict[str, Any]]:
        seen.append(schema)
        return [{"status": "open"} for _ in range(count)]

    with patch_clients(make_client(values)):
        records = await generate_synthetic_data(Ticket, count=3, seed=0)

    assert len(records) == 3
    assert all(r.status is Status.open for r in records)
    (schema,) = seen
    assert schema["items"]["properties"]["status"] == {"$ref": "#/$defs/Status"}
    assert "Status" in schema["items"]["$defs"]


async def test_client_hoists_defs_to_schema_root() -> None:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = '{"items": [{"status": "open"}]}'
    acompletion = AsyncMock(return_value=response)

    client = LiteLLMClient(api_base="http://proxy.test", model="m")
    item_schema = {
        "type": "object",
        "properties": {"status": {"$ref": "#/$defs/Status"}},
        "required": ["status"],
        "$defs": {"Status": {"type": "string", "enum": ["open", "closed"]}},
    }
    with patch("gendantic.llm.litellm.acompletion", acompletion):
        items = await client.generate_structured(
            {"type": "array", "items": item_schema}, "prompt", count=1
        )

    assert items == [{"status": "open"}]
    sent = acompletion.call_args.kwargs["response_format"]["json_schema"]["schema"]
    # Definitions live at the document root, where "#/$defs/..." resolves...
    assert sent["$defs"] == item_schema["$defs"]
    # ...and not (also) nested inside the item schema.
    assert "$defs" not in sent["properties"]["items"]["items"]
    # The LLM-facing prompt still shows the definitions alongside the schema.
    system_message = acompletion.call_args.kwargs["messages"][0]["content"]
    assert "$defs" in system_message


# --- generate_dataset honours max_concurrency --------------------------------


class Parent(BaseModel):
    id: Annotated[int, PrimaryKey()]
    v: Annotated[float, Normal(mean=0, std=1)]


class Child(BaseModel):
    id: Annotated[int, PrimaryKey()]
    w: Annotated[float, Normal(mean=0, std=1)]


async def test_generate_dataset_shares_one_semaphore_across_models() -> None:
    calls: list[dict[str, Any]] = []

    async def fake_generate(model_class, count, context, seed, **kwargs):
        calls.append(kwargs)
        return [model_class(**row) for row in _rows(model_class, count, kwargs)]

    def _rows(model_class, count, kwargs):
        value_field = "v" if model_class is Parent else "w"
        return [{**pre, value_field: 0.0} for pre in kwargs["prefilled"][:count]]

    with patch(
        "gendantic.generator._generate_with_distribution_sampling", fake_generate
    ):
        dataset = await generate_dataset({Parent: 2, Child: 3}, max_concurrency=2)

    assert len(dataset[Parent]) == 2 and len(dataset[Child]) == 3
    semaphores = {id(c["semaphore"]) for c in calls}
    assert len(semaphores) == 1, "every model should draw on the same budget"
    assert calls[0]["semaphore"]._value == 2


async def test_generate_dataset_rejects_non_positive_max_concurrency() -> None:
    with pytest.raises(ValueError, match="max_concurrency must be >= 1"):
        await generate_dataset({Parent: 1}, max_concurrency=0)


# --- Model source round-trips keep bounds and newer specs --------------------


class Staff(BaseModel):
    """Staff record."""

    dept: Annotated[str, Categorical(weights={"Eng": 0.5, "Ops": 0.5})]
    age: Annotated[int, Uniform(min=18, max=65)]
    salary: Annotated[
        float,
        Conditional(
            on="dept",
            cases={"Eng": Normal(mean=9, std=1)},
            default=Normal(mean=5, std=1),
        ),
    ]
    bonus: Annotated[float, Normal(mean=1, std=1)] = Field(ge=0, le=10)
    spend: Annotated[
        float,
        Conditional(
            on="age",
            cases={Range(max=30): Normal(mean=1, std=1)},
            default=Normal(mean=2, std=1),
        ),
    ]
    nickname: str = "none"


def test_model_source_repr_keeps_field_bounds_and_conditionals() -> None:
    source = _get_model_source_repr(Staff, {})
    assert (
        "bonus: Annotated[float, Normal(mean=1, std=1)] = Field(ge=0, le=10)" in source
    )
    assert "Conditional(on='dept'" in source
    assert "Range(min=None, max=30)" in source
    assert "nickname: str = 'none'" in source

    # The rendered source is accepted by the sandbox and executes back into a
    # model that carries the same bounds.
    _validate_code_safety(source)
    rebuilt = _execute_model_code(source)
    specs = LLMDrivenModelAnalyser.extract_distribution_specs_with_types(rebuilt)
    assert specs["bonus"][2] == {"ge": 0.0, "le": 10.0, "gt": None, "lt": None}
    assert isinstance(specs["salary"][0], Conditional)


def test_basic_source_repr_keeps_default_alongside_constraints() -> None:
    class Basic(BaseModel):
        score: int = Field(default=5, ge=0, le=10)

    source = _get_basic_model_source_repr(Basic)
    assert "score: int = Field(default=5, ge=0, le=10)" in source


def test_source_repr_covers_inherited_fields() -> None:
    class Base(BaseModel):
        age: Annotated[int, Uniform(min=18, max=65)]

    class Derived(Base):
        name: str

    source = _get_model_source_repr(Derived, {})
    assert "age: Annotated[int, Uniform(min=18, max=65)]" in source
    assert "name: str" in source
