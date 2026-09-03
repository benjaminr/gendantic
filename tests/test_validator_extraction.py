"""Validator metadata must reach both the analysis and generation prompts.

Pydantic v2 stores validators on ``__pydantic_decorators__`` rather than as
attributes discoverable via ``dir()``, so these tests guard against a
regression where validator context silently never reached the LLM.
"""

from pydantic import BaseModel, field_validator, model_validator

from gendantic.generator import (
    _extract_critical_validation_requirements,
    _get_validator_hints_from_docstrings,
)
from gendantic.llm_driven_analyser import LLMDrivenModelAnalyser


class Employee(BaseModel):
    """An employee record."""

    email: str
    name: str
    age: int

    @field_validator("email")
    @classmethod
    def check_company_email(cls, v: str) -> str:
        """Emails must end with @acme.com"""
        return v

    @model_validator(mode="after")
    def check_consistency(self) -> "Employee":
        """Name and email local part should agree."""
        return self


def test_field_specific_validators_are_found() -> None:
    validators = LLMDrivenModelAnalyser._extract_field_specific_validators(
        "email", Employee
    )
    assert [v["validator_name"] for v in validators] == ["check_company_email"]
    assert validators[0]["mode"] == "after"
    assert "acme.com" in validators[0]["description"]

    # A field with no validator returns nothing.
    assert LLMDrivenModelAnalyser._extract_field_specific_validators("name", Employee) == []


def test_extract_validators_info_covers_field_and_model_validators() -> None:
    info = LLMDrivenModelAnalyser._extract_validators_info(Employee)

    assert info["check_company_email"]["type"] == "field_validator"
    assert info["check_company_email"]["fields"] == ["email"]
    assert info["check_consistency"]["type"] == "model_validator"
    assert info["check_consistency"]["mode"] == "after"


def test_validators_render_into_analysis_prompt() -> None:
    pydantic_info = LLMDrivenModelAnalyser._extract_pure_pydantic_info(Employee)
    prompt = LLMDrivenModelAnalyser._build_analysis_prompt(
        Employee, pydantic_info, context="HR", count=3
    )

    assert "check_company_email" in prompt
    assert "check_consistency" in prompt
    assert "acme.com" in prompt


def test_validator_hints_reach_generation_prompt_inputs() -> None:
    hints = _get_validator_hints_from_docstrings(Employee)
    assert "check_company_email" in hints
    assert "acme.com" in hints
    assert "check_consistency" in hints


def test_critical_email_domain_requirement_is_extracted() -> None:
    requirements = _extract_critical_validation_requirements(Employee)
    assert "CRITICAL EMAIL DOMAIN" in requirements
    assert "acme.com" in requirements


def test_model_without_validators_yields_empty() -> None:
    class Plain(BaseModel):
        x: int

    assert LLMDrivenModelAnalyser._extract_validators_info(Plain) == {}
    assert _get_validator_hints_from_docstrings(Plain) == ""
    assert _extract_critical_validation_requirements(Plain) == ""
