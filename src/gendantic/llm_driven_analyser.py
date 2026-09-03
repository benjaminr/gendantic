import json
import logging
from typing import Any

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from ._fields import first_marker, iter_fields
from .distributions import Conditional, DistributionSpec
from .llm import get_client
from .prompts import load_prompt

logger = logging.getLogger("gendantic")


class LLMDrivenModelAnalyser:
    @classmethod
    def extract_distribution_specs(
        cls, model_class: type[BaseModel]
    ) -> dict[str, DistributionSpec | Conditional]:
        """
        Extract distribution specifications from Annotated type hints.

        Scans the model's fields (including those inherited from parent
        models) for DistributionSpec (and Conditional) instances within
        Annotated types.

        Args:
            model_class: Pydantic model class to extract specs from

        Returns:
            Dict mapping field names to their DistributionSpec/Conditional
            instances. Fields without distribution specs are not included.

        Example:
            class Employee(BaseModel):
                salary: Annotated[int, Normal(mean=50000, std=15000)]
                name: str  # No distribution

            specs = LLMDrivenModelAnalyser.extract_distribution_specs(Employee)
            # Returns: {"salary": Normal(mean=50000, std=15000)}
        """
        specs: dict[str, DistributionSpec | Conditional] = {}
        for field_name, _base_type, markers in iter_fields(model_class):
            # Only one distribution per field: the first marker wins.
            spec = first_marker(markers, (DistributionSpec, Conditional))
            if spec is not None:
                specs[field_name] = spec
        return specs

    @classmethod
    def extract_distribution_specs_with_types(
        cls, model_class: type[BaseModel]
    ) -> dict[
        str, tuple[DistributionSpec | Conditional, type, dict[str, float | None]]
    ]:
        """
        Extract distribution specifications with their target types and constraints.

        Like extract_distribution_specs but also returns the base type
        (e.g., int or float) from the Annotated hint, and Field constraints
        (ge, le, gt, lt) for clipping sampled values. An ``Optional`` wrapper
        on the base type is stripped, so ``Annotated[Optional[int], ...]``
        is sampled (and rounded) as an ``int``.

        Returns:
            Dict mapping field names to (spec, target_type, constraints) tuples,
            where spec is a DistributionSpec or a Conditional.
            constraints dict has keys: 'ge', 'le', 'gt', 'lt' with float values or None.
        """
        specs: dict[
            str, tuple[DistributionSpec | Conditional, type, dict[str, float | None]]
        ] = {}

        for field_name, base_type, markers in iter_fields(model_class):
            spec = first_marker(markers, (DistributionSpec, Conditional))
            if spec is None:
                continue
            # Pydantic v2 stores Field(ge=..., le=..., gt=..., lt=...) as
            # annotated_types Ge/Le/Gt/Lt objects alongside our markers.
            constraints: dict[str, float | None] = {
                "ge": None,
                "le": None,
                "gt": None,
                "lt": None,
            }
            for meta in markers:
                meta_type = type(meta).__name__
                if meta_type == "Ge" and hasattr(meta, "ge"):
                    constraints["ge"] = float(meta.ge)
                elif meta_type == "Le" and hasattr(meta, "le"):
                    constraints["le"] = float(meta.le)
                elif meta_type == "Gt" and hasattr(meta, "gt"):
                    constraints["gt"] = float(meta.gt)
                elif meta_type == "Lt" and hasattr(meta, "lt"):
                    constraints["lt"] = float(meta.lt)
            specs[field_name] = (spec, base_type, constraints)

        return specs

    @classmethod
    async def analyse_model_for_generation(
        cls,
        model_class: type[BaseModel],
        context: str = "general",
        count: int = 10,
    ) -> dict[str, Any]:
        """
        Use LLM to analyse Pydantic model and determine optimal generation strategy.

        This method extracts pure Pydantic metadata and lets the LLM interpret it
        intelligently rather than using hardcoded rules. If the LLM call fails,
        a deterministic fallback analysis is returned.
        """
        # Extract pure technical Pydantic information
        pydantic_info = cls._extract_pure_pydantic_info(model_class)

        # Let LLM analyse and interpret
        analysis = await cls._get_llm_analysis(
            model_class, pydantic_info, context, count
        )

        return analysis

    @classmethod
    def _extract_pure_pydantic_info(
        cls, model_class: type[BaseModel]
    ) -> dict[str, Any]:
        """Extract pure Pydantic metadata without interpretation."""

        schema = model_class.model_json_schema()
        fields = model_class.model_fields

        # Only the data actually rendered into the analysis prompt is collected
        # here (see _build_analysis_prompt): the JSON schema plus, per field, its
        # type info, constraints, description, schema fragment and validators.
        model_info: dict[str, Any] = {
            "schema": schema,
            "fields": {},
        }

        for field_name, field_info in fields.items():
            field_data = {
                "type_info": cls._extract_field_type_info(
                    field_name, model_class, field_info
                ),
                "constraints": cls._extract_field_constraints_raw(field_info),
                "description": getattr(field_info, "description", None),
                "schema_info": schema.get("properties", {}).get(field_name, {}),
                "field_validators": cls._extract_field_specific_validators(
                    field_name, model_class
                ),
            }

            model_info["fields"][field_name] = field_data

        # Extract validators and computed fields
        model_info["validators"] = cls._extract_validators_info(model_class)
        model_info["computed_fields"] = cls._extract_computed_fields_info(model_class)

        return model_info

    @classmethod
    def _extract_field_type_info(
        cls, field_name: str, model_class: type[BaseModel], field_info: FieldInfo
    ) -> dict[str, Any]:
        """Extract raw type information."""
        annotations = getattr(model_class, "__annotations__", {})
        annotation = annotations.get(field_name)

        return {
            "annotation": str(annotation) if annotation else None,
            "is_required": field_info.is_required(),
            "default": field_info.default if field_info.default is not ... else None,
        }

    # Constraint attributes carried on Pydantic v2 field metadata objects
    # (annotated_types.Ge/Le/Gt/Lt/MinLen/MaxLen/MultipleOf and the general
    # metadata that holds ``pattern``). Each object exposes the constraint under
    # the matching attribute name.
    _CONSTRAINT_ATTRS = (
        "gt",
        "ge",
        "lt",
        "le",
        "multiple_of",
        "min_length",
        "max_length",
        "pattern",
    )

    @classmethod
    def _extract_field_constraints_raw(cls, field_info: FieldInfo) -> dict[str, Any]:
        """Extract Pydantic v2 field constraints from ``field_info.metadata``.

        In Pydantic v2, ``Field(ge=..., max_length=..., pattern=...)``
        constraints are stored as metadata objects (``annotated_types.Ge``,
        ``MaxLen``, ...), not as attributes on ``FieldInfo``. Returns a flat
        ``{constraint: value}`` dict (e.g. ``{"ge": 0, "le": 120}``); empty when
        the field is unconstrained.
        """
        constraints: dict[str, Any] = {}
        for meta in field_info.metadata:
            for attr in cls._CONSTRAINT_ATTRS:
                value = getattr(meta, attr, None)
                if value is not None:
                    constraints[attr] = value
        return constraints

    @classmethod
    def _extract_field_specific_validators(
        cls, field_name: str, model_class: type[BaseModel]
    ) -> list[dict[str, Any]]:
        """Extract validators that apply to this specific field."""

        field_validators = []

        # Look for field validators that mention this field. Pydantic v2 records
        # them on __pydantic_decorators__, not as attributes on the class.
        for (
            name,
            decorator,
        ) in model_class.__pydantic_decorators__.field_validators.items():
            if field_name in decorator.info.fields:
                field_validators.append(
                    {
                        "validator_name": name,
                        "description": decorator.func.__doc__ or "",
                        "mode": decorator.info.mode,
                    }
                )

        return field_validators

    @classmethod
    def _extract_validators_info(cls, model_class: type[BaseModel]) -> dict[str, Any]:
        """Extract validator information including docstrings.

        Pydantic v2 stores validator metadata on ``__pydantic_decorators__``
        rather than as attributes discoverable via ``dir()``.
        """
        decorators = model_class.__pydantic_decorators__
        validators: dict[str, Any] = {}

        for name, field_validator in decorators.field_validators.items():
            validators[name] = {
                "type": "field_validator",
                "fields": list(field_validator.info.fields),
                "mode": field_validator.info.mode,
                "description": field_validator.func.__doc__,
            }

        for name, model_validator in decorators.model_validators.items():
            validators[name] = {
                "type": "model_validator",
                "mode": model_validator.info.mode,
                "description": model_validator.func.__doc__,
            }

        return validators

    @classmethod
    def _extract_computed_fields_info(
        cls, model_class: type[BaseModel]
    ) -> dict[str, Any]:
        """Extract computed field information."""
        computed = {}

        # Get computed fields from model
        if hasattr(model_class, "__pydantic_computed_fields__"):
            for name, info in model_class.__pydantic_computed_fields__.items():
                computed[name] = {
                    "return_type": str(info.return_type)
                    if hasattr(info, "return_type")
                    else None,
                    "description": getattr(info, "description", None),
                }

        return computed

    # JSON schema for the structured analysis response. Kept strict-mode
    # compatible (every object closes additionalProperties and lists all
    # required keys) so it works with providers that enforce strict schemas.
    _ANALYSIS_SCHEMA: dict[str, Any] = {
        "type": "object",
        "properties": {
            "model_analysis": {
                "type": "object",
                "properties": {
                    "purpose": {"type": "string"},
                    "domain": {"type": "string"},
                    "use_case": {"type": "string"},
                    "data_patterns": {"type": "string"},
                },
                "required": ["purpose", "domain", "use_case", "data_patterns"],
                "additionalProperties": False,
            },
            "generation_guidance": {
                "type": "object",
                "properties": {
                    "overall_strategy": {"type": "string"},
                    "field_relationships": {"type": "string"},
                    "data_quality_approach": {"type": "string"},
                    "cultural_considerations": {"type": "string"},
                },
                "required": [
                    "overall_strategy",
                    "field_relationships",
                    "data_quality_approach",
                    "cultural_considerations",
                ],
                "additionalProperties": False,
            },
        },
        "required": ["model_analysis", "generation_guidance"],
        "additionalProperties": False,
    }

    @classmethod
    async def _get_llm_analysis(
        cls,
        model_class: type[BaseModel],
        pydantic_info: dict[str, Any],
        context: str,
        count: int,
    ) -> dict[str, Any]:
        """Get comprehensive LLM analysis of the model.

        Routes through the shared LiteLLM client's structured-output path
        (honouring LITELLM_MODEL). Falls back to a deterministic analysis if
        the LLM call fails or returns an unexpected shape.
        """
        client = get_client()

        # Build comprehensive prompt with pure Pydantic data
        prompt = cls._build_analysis_prompt(model_class, pydantic_info, context, count)

        try:
            results = await client.generate_structured(
                schema={"type": "array", "items": cls._ANALYSIS_SCHEMA},
                prompt=prompt,
                count=1,
            )
            if (
                results
                and isinstance(results[0], dict)
                and "model_analysis" in results[0]
            ):
                return results[0]
            return cls._fallback_analysis(model_class)
        except Exception as e:
            logger.warning("LLM model analysis failed, using fallback: %s", e)
            return cls._fallback_analysis(model_class)

    @classmethod
    def _build_analysis_prompt(
        cls,
        model_class: type[BaseModel],
        pydantic_info: dict[str, Any],
        context: str,
        count: int,
    ) -> str:
        """Build a comprehensive analysis prompt for the LLM."""

        model_name = model_class.__name__
        docstring = model_class.__doc__ or "No documentation provided"

        # Format the Pydantic information clearly
        fields_info = []
        for field_name, field_data in pydantic_info["fields"].items():
            field_desc = f"""
**{field_name}**:
  - Type: {field_data["type_info"]["annotation"]}
  - Required: {field_data["type_info"]["is_required"]}
  - Default: {field_data["type_info"]["default"]}
  - Constraints: {json.dumps(field_data["constraints"], indent=2) if field_data["constraints"] else "None"}
  - Description: {field_data["description"]}
  - Field Validators: {json.dumps(field_data["field_validators"], indent=2) if field_data["field_validators"] else "None"}
  - Schema: {json.dumps(field_data["schema_info"], indent=2) if field_data["schema_info"] else "None"}"""
            fields_info.append(field_desc)

        validators_info = (
            json.dumps(pydantic_info["validators"], indent=2)
            if pydantic_info["validators"]
            else "None"
        )
        computed_info = (
            json.dumps(pydantic_info["computed_fields"], indent=2)
            if pydantic_info["computed_fields"]
            else "None"
        )

        # Load and format prompt
        prompt = load_prompt("model_analysis").format(
            count=count,
            model_name=model_name,
            docstring=docstring,
            context=context,
            fields_info="\n".join(fields_info),
            validators_info=validators_info,
            computed_info=computed_info,
            schema_json=json.dumps(pydantic_info.get("schema", {}), indent=2),
        )

        return prompt

    @classmethod
    def _fallback_analysis(cls, model_class: type[BaseModel]) -> dict[str, Any]:
        """Provide a basic fallback analysis if LLM fails."""
        return {
            "model_analysis": {
                "purpose": f"Model representing {model_class.__name__} data",
                "domain": "general",
                "use_case": "data_storage",
                "data_patterns": "basic_fields",
            },
            "generation_guidance": {
                "overall_strategy": "Generate basic realistic data respecting Pydantic constraints",
                "field_relationships": "Consider field types and constraints",
                "data_quality_approach": "High quality with variation",
                "cultural_considerations": "Diverse and inclusive",
            },
        }
