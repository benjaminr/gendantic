import asyncio
import inspect
import re
from typing import Any, List

from pydantic import BaseModel, ValidationError

from .llm import get_client
from .llm_driven_analyser import LLMDrivenModelAnalyser
from .prompts import load_prompt


async def generate(
    model_class: type[BaseModel],
    count: int = 10,
    *,
    provider: str | None = None,
    context: str = "general",
) -> list[BaseModel]:
    """
    Generate synthetic data using LLM analysis of Pydantic models.
    
    Args:
        model_class: The Pydantic model class to generate data for
        count: Number of records to generate
        provider: LLM provider ("openai" or "anthropic"), auto-detected if None
        context: Optional context about the intended use case for better generation

    Returns:
        List of model instances with synthetic data

    Examples:
        # Basic generation
        users = await generate(User, count=100)
        
        # Context-aware generation
        employees = await generate(
            Employee,
            count=50,
            context="UK tech startup with remote-first culture"
        )
    """

    if context == "general":
        context = f"Modern business using {model_class.__name__} data model"
    return await _generate_with_llm_intelligence_async(model_class, count, provider, context)


async def generate_batch(
    model_class: type[BaseModel],
    contexts: List[str],
    count: int = 10,
    *,
    provider: str | None = None,
) -> List[list[BaseModel]]:
    """
    Generate multiple batches with different contexts concurrently.
    
    Args:
        model_class: The Pydantic model class to generate data for
        contexts: List of contexts for different batches
        count: Number of records per batch
        provider: LLM provider ("openai" or "anthropic"), auto-detected if None

    Returns:
        List of lists, each containing model instances for one context

    Examples:
        contexts = ["UK bank", "US startup", "German manufacturing"]
        batches = await generate_batch(Employee, contexts, count=20)
    """
    
    tasks = [
        generate(model_class, count, provider=provider, context=context)
        for context in contexts
    ]
    return await asyncio.gather(*tasks)


async def _generate_with_llm_intelligence_async(
    model_class: type[BaseModel],
    count: int,
    provider: str | None,
    context: str,
) -> list[BaseModel]:
    """LLM-driven generation."""
    
    # Get LLM analysis of the model
    analysis = LLMDrivenModelAnalyser.analyse_model_for_generation(
        model_class, context=context, count=count
    )
    
    # Build generation prompt from analysis
    prompt = _build_generation_prompt_from_analysis(
        model_class, analysis, count, context
    )
    
    # Generate data asynchronously
    client = get_client(provider)
    
    try:
        schema = model_class.model_json_schema()
        raw_data = await client.generate_structured(
            schema=schema, prompt=prompt, count=count
        )
    except Exception as e:
        raise ValueError(f"Failed to generate data: {e}") from e
    
    # Validate and return
    return _validate_and_convert(model_class, raw_data)



def _build_generation_prompt_from_analysis(
    model_class: type[BaseModel],
    analysis: dict[str, Any],
    count: int,
    context: str,
) -> str:
    """Build generation prompt from LLM analysis results."""
    
    model_analysis = analysis.get("model_analysis", {})
    field_strategies = analysis.get("field_generation_strategies", {})
    guidance = analysis.get("generation_guidance", {})
    
    # Handle different LLM response formats (string vs dict)
    if isinstance(model_analysis, str):
        purpose = model_analysis  # Use the entire description as purpose
        domain = "business"  # Default to business domain
        use_case = "data storage"
    else:
        # Extract insights from dictionary format
        purpose = (model_analysis.get("purpose") or 
                   model_analysis.get("description") or 
                   f"{model_class.__name__} data")
        domain = (model_analysis.get("domain") or 
                  model_analysis.get("business_domain") or 
                  "general")
        use_case = model_analysis.get("use_case", "data storage")
    
    # Build field guidance
    field_guidance = []
    for field_name, strategy in field_strategies.items():
        reasoning = strategy.get("reasoning", "Standard generation")
        correlations = strategy.get("correlations", [])
        
        field_desc = f"**{field_name}**: {reasoning}"
        if correlations:
            field_desc += f" (correlates with: {', '.join(correlations)})"
        
        field_guidance.append(field_desc)
    
    # Extract validator requirements from analysis and model
    validator_requirements = []
    for field_name, strategy in field_strategies.items():
        params = strategy.get("parameters", {})
        if "email" in field_name and "domains" in params:
            domains = params["domains"]
            if len(domains) == 1:
                validator_requirements.append(f"- **{field_name}**: MUST use domain '{domains[0]}' (validator enforced)")
    
    # Also check if there are any validation requirements we should highlight
    schema = model_class.model_json_schema()
    for field_name in schema.get("properties", {}):
        if "email" in field_name.lower() and hasattr(model_class, f"{field_name}_must_be_company_domain"):
            # Heuristic: if there's a validator method, emphasize domain requirements
            validator_requirements.append(f"- **{field_name}**: Check for specific domain requirements in validators")
    
    # Special case for common validator patterns
    for attr_name in dir(model_class):
        if "validator" in attr_name and "email" in attr_name:
            validator_requirements.append("- **CRITICAL**: Pay attention to email domain validator requirements")
    
    # Extract critical validator requirements
    critical_requirements = _extract_critical_validation_requirements(model_class)
    validator_hints = _get_validator_hints_from_docstrings(model_class)
    
    # Build inline prompt
    generation_strategy = (
        guidance.get('overall_strategy', 'Generate realistic data respecting all constraints')
        if isinstance(guidance, dict) else str(guidance)
    )
    field_relationships = (
        guidance.get('field_relationships', 'Consider natural correlations between fields')
        if isinstance(guidance, dict) else 'Consider natural correlations between fields'
    )
    field_guidance_text = (
        '\n'.join(field_guidance) 
        if field_guidance 
        else 'Generate appropriate values for all fields based on their types and constraints.'
    )
    additional_requirements = '\n'.join(validator_requirements) if validator_requirements else ''
    
    prompt = load_prompt("data_generation").format(
        count=count,
        model_class=model_class.__name__,
        critical_requirements=critical_requirements,
        validator_hints=validator_hints,
        additional_requirements=additional_requirements,
        purpose=purpose,
        domain=domain,
        use_case=use_case,
        context=context,
        generation_strategy=generation_strategy,
        field_relationships=field_relationships,
        field_guidance=field_guidance_text
    )
    
    return prompt


def _validate_and_convert(
    model_class: type[BaseModel], raw_data: list[dict[str, Any]]
) -> list[BaseModel]:
    """Validate and convert raw data to model instances."""
    
    validated_data = []
    for item in raw_data:
        try:
            instance = model_class(**item)
            validated_data.append(instance)
        except ValidationError as e:
            print(f"Validation error for item {item}: {e}")
            continue

    if not validated_data:
        raise ValueError("No valid data was generated")

    return validated_data


def _get_validator_hints_from_docstrings(model_class: type[BaseModel]) -> str:
    """Extract validator hints from model docstrings and validators."""
    hints = []
    
    # Check for field validators with specific docstrings
    for attr_name in dir(model_class):
        attr = getattr(model_class, attr_name)
        if hasattr(attr, "__pydantic_validator__") or "validator" in attr_name:
            doc = getattr(attr, "__doc__", "")
            if doc:
                hints.append(f"- **{attr_name}**: {doc.strip()}")
    
    return "\n".join(hints) if hints else ""


def _extract_critical_validation_requirements(model_class: type[BaseModel]) -> str:
    """Extract critical validation requirements by examining the model."""
    
    requirements = []
    
    # Direct inspection of specific validation patterns
    for attr_name in dir(model_class):
        attr = getattr(model_class, attr_name)
        
        # Check for field validators
        if hasattr(attr, "__pydantic_validator__") or callable(attr) and "validator" in attr_name:
            doc = getattr(attr, "__doc__", "")
            
            # Extract specific domain requirements from docstrings and method names
            if "email" in attr_name and ("company" in attr_name or "domain" in doc.lower()):
                if "@" in doc:
                    # Extract domain from docstring
                    domain_match = re.search(r'@([a-zA-Z0-9.-]+\.com)', doc)
                    if domain_match:
                        domain = domain_match.group(1)
                        requirements.append(f"- **CRITICAL EMAIL DOMAIN**: ALL emails must end with @{domain} (validator enforced)")
                elif "company" in doc.lower() or "work" in doc.lower():
                    requirements.append(f"- **CRITICAL EMAIL DOMAIN**: ALL emails must use company domain (validator enforced)")
            
            # Generic validator documentation
            if doc and len(doc.strip()) > 10:
                field_match = re.search(r'(\w+)', attr_name)
                field_name = field_match.group(1) if field_match else "field"
                requirements.append(f"- **{field_name.upper()} VALIDATION**: {doc.strip()}")
    
    # Also check source code for validation error messages
    try:
        source = inspect.getsource(model_class)
        if "@" in source and "ValueError" in source:
            # Extract domain requirements from validation error messages
            domain_matches = re.findall(r'@([a-zA-Z0-9.-]+\.com)', source)
            for domain in set(domain_matches):
                requirements.append(f"- **MANDATORY EMAIL DOMAIN**: ALL generated emails MUST use @{domain}")
    except (OSError, TypeError):
        pass
    
    return "\n".join(requirements) if requirements else ""
