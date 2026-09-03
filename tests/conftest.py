"""Shared test fixtures for the offline (mocked-LLM) test suite.

Every LLM-dependent test mocks the single seam
``get_client().generate_structured(schema, prompt, count)``. The generation
pipeline issues two kinds of calls against that seam:

* a *model-analysis* call, whose schema carries a ``model_analysis`` block, and
* *field-generation* calls, which request the actual record fields.

The fixtures here centralise the analysis-shortcut and the ``AsyncMock`` wiring
so each test only supplies its own field-generation logic.
"""

import json
from contextlib import ExitStack, contextmanager
from typing import Any, Callable, Iterator
from unittest.mock import AsyncMock, patch

import pytest

# A canned analysis payload matching LLMDrivenModelAnalyser._ANALYSIS_SCHEMA.
ANALYSIS: dict[str, Any] = {
    "model_analysis": {
        "purpose": "test",
        "domain": "testing",
        "use_case": "unit-test",
        "data_patterns": "synthetic",
    },
    "generation_guidance": {
        "overall_strategy": "s",
        "field_relationships": "r",
        "data_quality_approach": "q",
        "cultural_considerations": "c",
    },
}

# A field-generation callback: (schema, prompt, count) -> list of record dicts.
FieldValuesFn = Callable[[dict[str, Any], str, int], list[dict[str, Any]]]


def is_analysis_call(schema: dict[str, Any]) -> bool:
    """True when a generate_structured call is the model-analysis request."""
    return "model_analysis" in json.dumps(schema)


@pytest.fixture
def make_client() -> Callable[[FieldValuesFn], Any]:
    """Return a factory building a mock client from a field-generation callback.

    The callback receives ``(schema, prompt, count)`` and returns the record
    rows for field-generation calls; analysis calls are answered automatically
    with :data:`ANALYSIS`.
    """

    def _make(field_values_fn: FieldValuesFn) -> Any:
        def gen(
            schema: dict[str, Any], prompt: str, count: int = 1
        ) -> list[dict[str, Any]]:
            if is_analysis_call(schema):
                return [ANALYSIS]
            return field_values_fn(schema, prompt, count)

        client = AsyncMock()
        client.generate_structured = AsyncMock(side_effect=gen)
        return client

    return _make


@pytest.fixture
def patch_clients() -> Callable[[Any], Any]:
    """Return a context manager that patches get_client in every module using it."""

    @contextmanager
    def _patch(client: Any) -> Iterator[None]:
        with ExitStack() as stack:
            for mod in ("generator", "llm_driven_analyser", "model_generator"):
                stack.enter_context(
                    patch(f"gendantic.{mod}.get_client", return_value=client)
                )
            yield

    return _patch
