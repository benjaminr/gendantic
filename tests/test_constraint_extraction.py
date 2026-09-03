"""Field constraints must reach the analysis prompt.

Pydantic v2 stores ``Field(ge=..., max_length=..., pattern=...)`` constraints as
metadata objects (``annotated_types.Ge``, ``MaxLen``, ...) rather than as
attributes on ``FieldInfo``, so these tests guard against a regression where the
constraint extractor silently returned ``{}`` for every field.
"""

from pydantic import BaseModel, Field

from gendantic.llm_driven_analyser import LLMDrivenModelAnalyser


class Constrained(BaseModel):
    """A model exercising numeric and string constraints."""

    age: int = Field(ge=0, le=120)
    score: float = Field(gt=0.0, lt=1.0, multiple_of=0.1)
    name: str = Field(min_length=2, max_length=50, pattern=r"^[A-Z]")
    plain: int


def test_numeric_constraints_are_extracted() -> None:
    fields = Constrained.model_fields
    assert LLMDrivenModelAnalyser._extract_field_constraints_raw(fields["age"]) == {
        "ge": 0,
        "le": 120,
    }
    assert LLMDrivenModelAnalyser._extract_field_constraints_raw(fields["score"]) == {
        "gt": 0.0,
        "lt": 1.0,
        "multiple_of": 0.1,
    }


def test_string_constraints_are_extracted() -> None:
    constraints = LLMDrivenModelAnalyser._extract_field_constraints_raw(
        Constrained.model_fields["name"]
    )
    assert constraints == {
        "min_length": 2,
        "max_length": 50,
        "pattern": "^[A-Z]",
    }


def test_unconstrained_field_yields_empty() -> None:
    assert (
        LLMDrivenModelAnalyser._extract_field_constraints_raw(
            Constrained.model_fields["plain"]
        )
        == {}
    )


def test_constraints_render_into_analysis_prompt() -> None:
    pydantic_info = LLMDrivenModelAnalyser._extract_pure_pydantic_info(Constrained)
    prompt = LLMDrivenModelAnalyser._build_analysis_prompt(
        Constrained, pydantic_info, context="HR", count=3
    )

    # Real constraint values reach the LLM, not a blanket "None".
    assert '"ge": 0' in prompt
    assert '"le": 120' in prompt
    assert '"max_length": 50' in prompt
    assert '"pattern": "^[A-Z]"' in prompt
