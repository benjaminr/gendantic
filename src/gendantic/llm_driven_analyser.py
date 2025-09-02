import json
from typing import Any

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from .llm import get_client
from .prompts import load_prompt


class LLMDrivenModelAnalyser:
    @classmethod
    def analyse_model_for_generation(
        cls,
        model_class: type[BaseModel],
        context: str = "general",
        count: int = 10,
    ) -> dict[str, Any]:
        """
        Use LLM to analyse Pydantic model and determine optimal generation strategy.
        
        This method extracts pure Pydantic metadata and lets the LLM interpret it
        intelligently rather than using hardcoded rules.
        """
        # Extract pure technical Pydantic information
        pydantic_info = cls._extract_pure_pydantic_info(model_class)
        
        # Let LLM analyse and interpret
        analysis = cls._get_llm_analysis(
            model_class, pydantic_info, context, count
        )
        
        return analysis

    @classmethod
    def _extract_pure_pydantic_info(cls, model_class: type[BaseModel]) -> dict[str, Any]:
        """Extract pure Pydantic metadata without interpretation."""
        
        schema = model_class.model_json_schema()
        fields = model_class.model_fields
        
        # Get model-level information
        model_info = {
            "model_name": model_class.__name__,
            "model_docstring": model_class.__doc__,
            "model_config": getattr(model_class, "model_config", {}),
            "schema": schema,
            "fields": {},
        }
        
        # Extract field information without interpretation
        for field_name, field_info in fields.items():
            field_data = {
                "name": field_name,
                "type_info": cls._extract_field_type_info(field_name, model_class, field_info),
                "constraints": cls._extract_field_constraints_raw(field_info),
                "validation": cls._extract_validation_info_raw(field_info),
                "metadata": cls._extract_field_metadata_raw(field_info),
                "schema_info": schema.get("properties", {}).get(field_name, {}),
                "field_validators": cls._extract_field_specific_validators(field_name, model_class),
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
            "python_type": type(annotation).__name__ if annotation else None,
            "is_required": field_info.is_required(),
            "default": field_info.default if field_info.default is not ... else None,
        }

    @classmethod
    def _extract_field_constraints_raw(cls, field_info: FieldInfo) -> dict[str, Any]:
        """Extract raw Pydantic constraints without interpretation."""
        constraints = {}
        
        # Numeric constraints
        for attr in ["gt", "ge", "lt", "le", "multiple_of"]:
            if hasattr(field_info, attr):
                value = getattr(field_info, attr)
                if value is not None:
                    constraints[attr] = value
        
        # String constraints
        for attr in ["min_length", "max_length", "pattern"]:
            if hasattr(field_info, attr):
                value = getattr(field_info, attr)
                if value is not None:
                    constraints[attr] = value
        
        return constraints

    @classmethod
    def _extract_validation_info_raw(cls, field_info: FieldInfo) -> dict[str, Any]:
        """Extract raw validation information."""
        return {
            "required": field_info.is_required(),
            "alias": getattr(field_info, "alias", None),
            "description": getattr(field_info, "description", None),
            "deprecated": getattr(field_info, "deprecated", None),
            "frozen": getattr(field_info, "frozen", None),
        }

    @classmethod
    def _extract_field_metadata_raw(cls, field_info: FieldInfo) -> dict[str, Any]:
        """Extract raw field metadata."""
        metadata = {}
        
        if hasattr(field_info, "json_schema_extra"):
            metadata["json_schema_extra"] = field_info.json_schema_extra
        
        if hasattr(field_info, "examples"):
            metadata["examples"] = field_info.examples
        
        return metadata

    @classmethod
    def _extract_field_specific_validators(
        cls, field_name: str, model_class: type[BaseModel]
    ) -> list[dict[str, Any]]:
        """Extract validators that apply to this specific field."""
        
        field_validators = []
        
        # Look for field validators that mention this field
        for attr_name in dir(model_class):
            attr = getattr(model_class, attr_name)
            if hasattr(attr, "__pydantic_validator__"):
                validator_info = getattr(attr, "__pydantic_validator__", {})
                validator_fields = validator_info.get("fields", [])
                
                if field_name in validator_fields:
                    field_validators.append({
                        "validator_name": attr_name,
                        "description": getattr(attr, "__doc__", ""),
                        "mode": validator_info.get("mode", None),
                    })
        
        return field_validators

    @classmethod
    def _extract_validators_info(cls, model_class: type[BaseModel]) -> dict[str, Any]:
        """Extract validator information including docstrings."""
        validators = {}
        
        # Get field validators with more details
        for attr_name in dir(model_class):
            attr = getattr(model_class, attr_name)
            if hasattr(attr, "__pydantic_validator__"):
                validator_info = getattr(attr, "__pydantic_validator__", {})
                validators[attr_name] = {
                    "type": "field_validator",
                    "fields": validator_info.get("fields", []),
                    "mode": validator_info.get("mode", None),
                    "description": getattr(attr, "__doc__", None),
                    "source_code": f"Method: {attr_name}",
                }
        
        return validators

    @classmethod
    def _extract_computed_fields_info(cls, model_class: type[BaseModel]) -> dict[str, Any]:
        """Extract computed field information."""
        computed = {}
        
        # Get computed fields from model
        if hasattr(model_class, "__pydantic_computed_fields__"):
            for name, info in model_class.__pydantic_computed_fields__.items():
                computed[name] = {
                    "return_type": str(info.return_type) if hasattr(info, "return_type") else None,
                    "description": getattr(info, "description", None),
                }
        
        return computed

    @classmethod
    def _get_llm_analysis(
        cls,
        model_class: type[BaseModel],
        pydantic_info: dict[str, Any],
        context: str,
        count: int,
    ) -> dict[str, Any]:
        """Get comprehensive LLM analysis of the model."""
        
        client = get_client()
        
        # Build comprehensive prompt with pure Pydantic data
        prompt = cls._build_analysis_prompt(model_class, pydantic_info, context, count)
        
        # Define the analysis schema we want back
        analysis_schema = {
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
                },
                "field_generation_strategies": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "distribution_type": {"type": "string"},
                            "parameters": {"type": "object"},
                            "reasoning": {"type": "string"},
                            "correlations": {"type": "array", "items": {"type": "string"}},
                            "quality_hints": {"type": "object"},
                        },
                    },
                },
                "generation_guidance": {
                    "type": "object",
                    "properties": {
                        "overall_strategy": {"type": "string"},
                        "field_relationships": {"type": "string"},
                        "data_quality_approach": {"type": "string"},
                        "cultural_considerations": {"type": "string"},
                    },
                },
            },
        }
        
        try:
            # For analysis, we don't want structured generation of model data
            # Instead, let's use a simple JSON request for analysis
            if hasattr(client, 'client') and hasattr(client.client, 'chat'):
                # OpenAI client
                import json
                response = client.client.chat.completions.create(
                    model="gpt-4o-mini",
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": "You are a data modeling expert. Respond only with valid JSON matching the requested analysis format."},
                        {"role": "user", "content": prompt + "\n\nRespond with JSON containing model_analysis, field_generation_strategies, and generation_guidance objects."}
                    ],
                )
                content = response.choices[0].message.content or "{}"
                result = json.loads(content)
                if isinstance(result, dict) and any(key in result for key in ["model_analysis", "field_generation_strategies", "generation_guidance"]):
                    return result
                else:
                    return cls._fallback_analysis(model_class)
            else:
                # Anthropic fallback
                return cls._fallback_analysis(model_class)
                
        except Exception as e:
            print(f"LLM analysis failed: {e}")
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
  - Type: {field_data['type_info']['annotation']}
  - Required: {field_data['type_info']['is_required']}
  - Default: {field_data['type_info']['default']}
  - Constraints: {json.dumps(field_data['constraints'], indent=2) if field_data['constraints'] else 'None'}
  - Description: {field_data['validation'].get('description', 'None')}
  - Field Validators: {json.dumps(field_data['field_validators'], indent=2) if field_data['field_validators'] else 'None'}
  - Schema: {json.dumps(field_data['schema_info'], indent=2) if field_data['schema_info'] else 'None'}"""
            fields_info.append(field_desc)
        
        validators_info = json.dumps(pydantic_info["validators"], indent=2) if pydantic_info["validators"] else "None"
        computed_info = json.dumps(pydantic_info["computed_fields"], indent=2) if pydantic_info["computed_fields"] else "None"
        
        # Load and format prompt
        prompt = load_prompt("model_analysis").format(
            count=count,
            model_name=model_name,
            docstring=docstring,
            context=context,
            fields_info='\n'.join(fields_info),
            validators_info=validators_info,
            computed_info=computed_info,
            schema_json=json.dumps(pydantic_info.get('schema', {}), indent=2)
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
            "field_generation_strategies": {},
            "generation_guidance": {
                "overall_strategy": "Generate basic realistic data respecting Pydantic constraints",
                "field_relationships": "Consider field types and constraints",
                "data_quality_approach": "High quality with variation",
                "cultural_considerations": "Diverse and inclusive",
            },
        }