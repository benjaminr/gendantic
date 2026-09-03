"""
Dynamic Pydantic model generation from natural language descriptions.

The LLM generates actual Python code for the model, including Annotated
types with distribution specifications where appropriate.

Security: Generated code is validated via AST parsing before execution
to ensure only safe constructs (class definitions, type annotations) are used.
"""

import ast
from typing import Any

from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from .llm import get_client
from .prompts import load_prompt


class CodeValidationError(Exception):
    """Raised when generated code contains unsafe constructs."""

    pass


async def generate_model_from_description(
    description: str,
    *,
    model_name: str | None = None,
) -> tuple[type, str]:
    """
    Generate a Pydantic model class from a natural language description.

    The LLM analyses the description and generates Python code for a model,
    intelligently choosing which fields should have statistical distributions
    and which should be LLM-generated.

    Security: The generated code is validated via AST parsing to ensure
    it only contains safe constructs (class definitions, type annotations).
    No arbitrary code execution is allowed.

    Args:
        description: Natural language description of the data model.
        model_name: Optional name for the model class.

    Returns:
        Tuple of (model_class, source_code)

    Raises:
        CodeValidationError: If the generated code contains unsafe constructs

    Example:
        Model, code = await generate_model_from_description(
            "A customer support ticket with priority, category, and description"
        )
    """
    # Get model code from LLM
    code = await _get_model_code_from_llm(description, model_name)

    # Validate the code is safe before execution
    _validate_code_safety(code)

    # Execute the code to create the model class
    model_class = _execute_model_code(code)

    return model_class, code


async def extend_model_with_correlations(
    model_class: type[BaseModel],
) -> tuple[type, str]:
    """
    Extend an existing Pydantic model with LLM-suggested correlations.

    Analyses a model that has distribution-annotated fields and asks the LLM
    to suggest appropriate correlations and copula types based on the
    field semantics and domain knowledge.

    Args:
        model_class: An existing Pydantic BaseModel class with distribution specs

    Returns:
        Tuple of (extended_model_class, source_code) with __correlations__ added

    Raises:
        ValueError: If no distribution fields found or LLM fails
        CodeValidationError: If generated code contains unsafe constructs

    Example:
        class Employee(BaseModel):
            age: Annotated[int, Uniform(min=22, max=65)]
            experience: Annotated[int, Uniform(min=0, max=40)]
            salary: Annotated[float, Normal(mean=75000, std=20000)]
            name: str

        ExtendedEmployee, code = await extend_model_with_correlations(Employee)
        # ExtendedEmployee now has __correlations__ with suggested
        # correlations between age, experience, and salary
    """
    from .distributions import DistributionSpec
    from .llm_driven_analyser import LLMDrivenModelAnalyser

    # Extract distribution specs from the model. Conditional fields are sampled
    # per-group and cannot participate in copula correlations, so exclude them.
    dist_specs = {
        name: spec
        for name, spec in LLMDrivenModelAnalyser.extract_distribution_specs(
            model_class
        ).items()
        if isinstance(spec, DistributionSpec)
    }

    if not dist_specs:
        raise ValueError(
            f"Model {model_class.__name__} has no distribution-annotated fields. "
            "extend_model_with_correlations() requires at least 2 fields with "
            "distribution specs."
        )

    if len(dist_specs) < 2:
        raise ValueError(
            f"Model {model_class.__name__} has only {len(dist_specs)} distribution field(s). "
            "extend_model_with_correlations() requires at least 2 fields to create "
            "correlations."
        )

    # Get the model source code representation
    model_code = _get_model_source_repr(model_class, dist_specs)

    # Format distribution fields info for the prompt
    dist_fields_info = "\n".join(
        f"- {name}: {spec.distribution_type} distribution"
        for name, spec in dist_specs.items()
    )

    # Get correlation suggestions from LLM
    correlations = await _get_correlations_from_llm(model_code, dist_fields_info)

    # Generate extended model code with correlations
    extended_code = _generate_extended_model_code(model_class, model_code, correlations)

    # Validate and execute
    _validate_code_safety(extended_code)
    extended_model = _execute_model_code(extended_code)

    return extended_model, extended_code


async def extend_model_with_distributions(
    model_class: type[BaseModel],
) -> tuple[type, str]:
    """
    Extend a basic Pydantic model with LLM-suggested statistical distributions.

    Analyses a model's fields and suggests appropriate distributions based on
    field names, types, and constraints. Numeric fields get statistical distributions,
    categorical fields get Categorical, and semantic fields remain LLM-generated.

    Args:
        model_class: A basic Pydantic BaseModel class without distribution specs

    Returns:
        Tuple of (extended_model_class, source_code) with Annotated distribution types

    Raises:
        ValueError: If LLM fails to suggest distributions

    Example:
        class Employee(BaseModel):
            name: str
            age: int
            salary: float
            department: str

        # LLM adds appropriate distributions
        DistEmployee, code = await extend_model_with_distributions(Employee)
        # Result:
        # class Employee(BaseModel):
        #     name: str  # Kept as LLM-generated
        #     age: Annotated[int, Uniform(min=22, max=65)]
        #     salary: Annotated[float, Normal(mean=75000, std=20000)]
        #     department: Annotated[str, Categorical(weights={...})]
    """
    # Get the model source code representation
    model_code = _get_basic_model_source_repr(model_class)

    # Get extended model code from LLM
    code = await _get_distribution_code_from_llm(model_code)

    # Validate and execute
    _validate_code_safety(code)
    extended_model = _execute_model_code(code)

    return extended_model, code


def _get_basic_model_source_repr(model_class: type[BaseModel]) -> str:
    """Generate a source code representation of a basic model (no distributions)."""
    lines = [f"class {model_class.__name__}(BaseModel):"]

    # Add docstring if present
    if model_class.__doc__:
        lines.append(f'    """{model_class.__doc__}"""')

    # Add field annotations
    for field_name, field_info in model_class.model_fields.items():
        annotation = model_class.__annotations__.get(field_name, "Any")
        type_str = _type_to_string(annotation)

        # Pydantic v2 stores constraints in metadata as Ge(ge=0), Le(le=100), etc.
        constraints = []
        constraint_map = {"ge": "Ge", "le": "Le", "gt": "Gt", "lt": "Lt"}
        for constraint_name, class_name in constraint_map.items():
            for meta in field_info.metadata:
                if type(meta).__name__ == class_name:
                    value = getattr(meta, constraint_name, None)
                    if value is not None:
                        constraints.append(f"{constraint_name}={value}")
                    break

        if constraints:
            lines.append(
                f"    {field_name}: {type_str} = Field({', '.join(constraints)})"
            )
        elif (
            field_info.default is not PydanticUndefined
            and field_info.default is not ...
        ):
            lines.append(f"    {field_name}: {type_str} = {field_info.default!r}")
        else:
            lines.append(f"    {field_name}: {type_str}")

    return "\n".join(lines)


async def _get_distribution_code_from_llm(
    model_code: str,
) -> str:
    """Ask LLM to generate model code with distributions added."""
    client = get_client()

    prompt = load_prompt("distribution_extension").format(model_code=model_code)

    response_schema = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code for the model with distribution annotations",
            },
        },
        "required": ["code"],
    }

    try:
        result = await client.generate_structured(
            schema=response_schema,
            prompt=prompt,
            count=1,
        )
        response = result[0] if isinstance(result, list) else result
        return str(response.get("code", ""))
    except Exception as e:
        raise ValueError(f"Failed to generate model with distributions: {e}") from e


def _get_model_source_repr(
    model_class: type[BaseModel],
    dist_specs: dict[str, Any],
) -> str:
    """Generate a source code representation of the model."""
    lines = [f"class {model_class.__name__}(BaseModel):"]

    # Add docstring if present
    if model_class.__doc__:
        lines.append(f'    """{model_class.__doc__}"""')

    # Add field annotations
    for field_name, field_info in model_class.model_fields.items():
        annotation = model_class.__annotations__.get(field_name, "Any")
        annotation_str = _annotation_to_string(annotation)

        # Check for actual default value (not PydanticUndefined which means required)
        if field_info.default is not PydanticUndefined:
            lines.append(f"    {field_name}: {annotation_str} = {field_info.default!r}")
        else:
            lines.append(f"    {field_name}: {annotation_str}")

    return "\n".join(lines)


def _annotation_to_string(annotation: Any) -> str:
    """Convert a type annotation to its string representation."""
    from typing import Annotated, get_args, get_origin

    if get_origin(annotation) is Annotated:
        args = get_args(annotation)
        base_type = args[0]
        metadata = args[1:]

        base_str = _type_to_string(base_type)
        # Distribution specs and other markers round-trip through repr().
        metadata_strs = [repr(m) for m in metadata]

        return f"Annotated[{base_str}, {', '.join(metadata_strs)}]"

    return _type_to_string(annotation)


def _type_to_string(t: Any) -> str:
    """Convert a type to its string representation."""
    from typing import get_args, get_origin

    origin = get_origin(t)
    if origin is not None:
        args = get_args(t)
        origin_name = getattr(origin, "__name__", str(origin))
        if args:
            args_str = ", ".join(_type_to_string(a) for a in args)
            return f"{origin_name}[{args_str}]"
        return origin_name

    if hasattr(t, "__name__"):
        return str(t.__name__)

    return str(t)


async def _get_correlations_from_llm(
    model_code: str,
    dist_fields_info: str,
) -> list[dict[str, Any]]:
    """Ask LLM to suggest correlations for the model."""
    client = get_client()

    prompt = load_prompt("model_extension").format(
        model_code=model_code,
        distribution_fields=dist_fields_info,
    )

    response_schema = {
        "type": "object",
        "properties": {
            "correlations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "field1": {"type": "string"},
                        "field2": {"type": "string"},
                        "correlation": {"type": "number"},
                        "copula_type": {"type": "string"},
                    },
                    "required": ["field1", "field2", "correlation"],
                },
            },
            "reasoning": {"type": "string"},
        },
        "required": ["correlations"],
    }

    try:
        result = await client.generate_structured(
            schema=response_schema,
            prompt=prompt,
            count=1,
        )
        response = result[0] if isinstance(result, list) else result
        return list(response.get("correlations", []))
    except Exception as e:
        raise ValueError(f"Failed to get correlation suggestions: {e}") from e


def _generate_extended_model_code(
    model_class: type[BaseModel],
    model_code: str,
    correlations: list[dict[str, Any]],
) -> str:
    """Generate code for the extended model with correlations."""
    if not correlations:
        # No correlations suggested, return original model code
        return model_code

    # Build correlations code
    corr_specs = []
    for corr in correlations:
        field1 = corr["field1"]
        field2 = corr["field2"]
        value = corr["correlation"]
        copula = corr.get("copula_type")

        if copula and copula != "gaussian":
            corr_specs.append(f'        ("{field1}", "{field2}", {value}, "{copula}"),')
        else:
            corr_specs.append(f'        ("{field1}", "{field2}", {value}),')

    correlations_code = "    __correlations__ = Correlations(\n"
    correlations_code += "\n".join(corr_specs)
    correlations_code += "\n    )"

    # Insert correlations into model code
    lines = model_code.split("\n")
    # Find where to insert (before the last line or at the end of class)
    insert_idx = len(lines)
    for i, line in enumerate(lines):
        if line.strip() and not line.startswith(" ") and i > 0:
            insert_idx = i
            break

    lines.insert(insert_idx, "")
    lines.insert(insert_idx + 1, correlations_code)

    return "\n".join(lines)


async def _get_model_code_from_llm(
    description: str,
    model_name: str | None,
) -> str:
    """Use LLM to generate Python code for the model."""
    client = get_client()

    prompt = load_prompt("model_generation").format(
        description=description,
        model_name_hint=f'Name the model "{model_name}".'
        if model_name
        else "Choose an appropriate model name.",
    )

    response_schema = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Complete Python code for the Pydantic model",
            },
        },
        "required": ["code"],
    }

    try:
        result = await client.generate_structured(
            schema=response_schema,
            prompt=prompt,
            count=1,
        )
        response = result[0] if isinstance(result, list) else result
        return str(response.get("code", ""))
    except Exception as e:
        raise ValueError(f"Failed to generate model code: {e}") from e


# Allowed AST node types for safe code. Any node whose type is not in this set
# is rejected by ``_validate_node`` before the code is executed.
ALLOWED_NODE_TYPES = {
    # Module structure
    ast.Module,
    ast.ClassDef,
    ast.FunctionDef,  # For validators
    ast.AsyncFunctionDef,
    # Expressions
    ast.Expr,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Constant,
    ast.Attribute,
    ast.Subscript,
    ast.Slice,
    ast.BinOp,
    ast.UnaryOp,
    ast.Compare,
    ast.BoolOp,
    ast.IfExp,  # ternary: `a if cond else b`
    ast.JoinedStr,  # f-strings (e.g. in validator error messages)
    ast.FormattedValue,
    # Operators
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.Pow,
    ast.FloorDiv,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.Is,
    ast.IsNot,
    ast.Or,
    ast.And,
    ast.Not,
    ast.USub,
    ast.BitOr,  # union type hints, e.g. `int | None`
    # Containers
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Set,
    # Assignment / annotations
    ast.Assign,  # e.g. `__correlations__ = Correlations(...)`
    ast.AnnAssign,
    ast.arg,
    ast.arguments,
    # Calls (for Field(), Normal(), etc.)
    ast.Call,
    ast.keyword,
    # Control flow (for validators)
    ast.Return,
    ast.If,
    ast.Pass,
    ast.Raise,
    # Other
    ast.alias,
    ast.withitem,
}

# Allowed names that can be called
ALLOWED_CALL_NAMES = {
    # Pydantic
    "BaseModel",
    "Field",
    "field_validator",
    "model_validator",
    # Typing
    "Annotated",
    "Optional",
    "List",
    "Dict",
    "Union",
    # Distributions
    "Normal",
    "Uniform",
    "Categorical",
    "LogNormal",
    "Exponential",
    "Poisson",
    "Beta",
    "Binomial",
    # Correlations
    "Correlations",
    "CopulaType",
    # Builtins for validators
    "ValueError",
    "len",
    "str",
    "int",
    "float",
    "bool",
    "isinstance",
}

# Explicitly forbidden patterns
FORBIDDEN_NAMES = {
    "exec",
    "eval",
    "compile",
    "open",
    "input",
    "__import__",
    "getattr",
    "setattr",
    "delattr",
    "globals",
    "locals",
    "vars",
    "dir",
    "type",
    "object",
    "os",
    "sys",
    "subprocess",
    "importlib",
    "builtins",
    "__builtins__",
    "__code__",
    "__globals__",
}


def _validate_code_safety(code: str) -> None:
    """
    Validate that the generated code only contains safe constructs.

    Uses AST parsing to inspect the code structure without executing it.

    Raises:
        CodeValidationError: If unsafe constructs are found
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise CodeValidationError(f"Generated code has syntax error: {e}") from e

    # Walk the AST and validate each node
    for node in ast.walk(tree):
        _validate_node(node)


def _validate_node(node: ast.AST) -> None:
    """Validate a single AST node."""
    # Check for Import statements - not allowed (we provide imports)
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        raise CodeValidationError(
            "Import statements not allowed in generated code. "
            "All necessary imports are provided automatically."
        )

    # Check for forbidden names
    if isinstance(node, ast.Name):
        if node.id in FORBIDDEN_NAMES:
            raise CodeValidationError(f"Forbidden name '{node.id}' in generated code")

    # Check for forbidden attribute access
    if isinstance(node, ast.Attribute):
        if node.attr in FORBIDDEN_NAMES:
            raise CodeValidationError(
                f"Forbidden attribute access '.{node.attr}' in generated code"
            )
        # Block dunder attributes except __doc__
        if node.attr.startswith("__") and node.attr != "__doc__":
            raise CodeValidationError(
                f"Dunder attribute access '.{node.attr}' not allowed"
            )

    # Validate function calls
    if isinstance(node, ast.Call):
        _validate_call(node)

    # Enforce the allowlist: reject any node type not explicitly permitted.
    if type(node) not in ALLOWED_NODE_TYPES:
        raise CodeValidationError(
            f"Disallowed syntax '{type(node).__name__}' in generated code"
        )


def _validate_call(node: ast.Call) -> None:
    """Validate a function call node."""
    # Get the function name being called
    if isinstance(node.func, ast.Name):
        func_name = node.func.id
        if func_name in FORBIDDEN_NAMES:
            raise CodeValidationError(f"Forbidden function call '{func_name}'")
        if func_name not in ALLOWED_CALL_NAMES:
            raise CodeValidationError(
                f"Call to '{func_name}' not allowed in generated code"
            )
    elif isinstance(node.func, ast.Attribute):
        # Method calls like self.something() - allow for validators
        attr_name = node.func.attr
        if attr_name in FORBIDDEN_NAMES:
            raise CodeValidationError(f"Forbidden method call '.{attr_name}'")


def _execute_model_code(code: str) -> type:
    """
    Execute the validated code and return the model class.

    The code has already been validated by _validate_code_safety().
    """
    # Import everything we need directly (not via exec)
    from datetime import date, datetime
    from typing import Annotated, Dict, List, Optional, Union

    from pydantic import BaseModel, Field, field_validator, model_validator

    from gendantic import (
        Beta,
        Binomial,
        Categorical,
        CopulaType,
        Correlations,
        Exponential,
        LogNormal,
        Normal,
        Poisson,
        Uniform,
    )

    # Get __build_class__ from builtins (needed for class definitions)
    if isinstance(__builtins__, dict):
        build_class = __builtins__["__build_class__"]
    else:
        build_class = __builtins__.__build_class__

    # Build namespace with all required objects pre-imported
    namespace: dict[str, Any] = {
        # Module-level attributes required by Pydantic
        "__name__": "__generated__",
        "__module__": "__generated__",
        # Restricted builtins
        "__builtins__": {
            "__build_class__": build_class,
            "True": True,
            "False": False,
            "None": None,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "set": set,
            "len": len,
            "isinstance": isinstance,
            "ValueError": ValueError,
            "TypeError": TypeError,
            "classmethod": classmethod,
        },
        # Typing
        "Annotated": Annotated,
        "Optional": Optional,
        "List": List,
        "Dict": Dict,
        "Union": Union,
        # Datetime
        "date": date,
        "datetime": datetime,
        # Pydantic
        "BaseModel": BaseModel,
        "Field": Field,
        "field_validator": field_validator,
        "model_validator": model_validator,
        # Distributions
        "Normal": Normal,
        "Uniform": Uniform,
        "Categorical": Categorical,
        "LogNormal": LogNormal,
        "Exponential": Exponential,
        "Poisson": Poisson,
        "Beta": Beta,
        "Binomial": Binomial,
        # Correlations
        "Correlations": Correlations,
        "CopulaType": CopulaType,
    }

    try:
        # Execute the generated model code in the restricted namespace
        exec(code, namespace)

        # Find the model class (should be a BaseModel subclass)
        for name, obj in namespace.items():
            if (
                isinstance(obj, type)
                and name not in ("BaseModel",)
                and hasattr(obj, "model_fields")
                and not name.startswith("_")
            ):
                return obj

        raise ValueError("No Pydantic model class found in generated code")

    except CodeValidationError:
        raise
    except SyntaxError as e:
        raise CodeValidationError(
            f"Generated code has syntax error: {e}\n\nCode:\n{code}"
        ) from e
    except Exception as e:
        raise ValueError(
            f"Failed to execute generated code: {e}\n\nCode:\n{code}"
        ) from e
