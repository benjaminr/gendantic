"""Tests for dynamic model generation and code safety validation."""

import pytest

from gendantic import CodeValidationError
from gendantic.model_generator import _execute_model_code, _validate_code_safety


class TestCodeSafetyValidation:
    """Test the AST-based code safety validation."""

    def test_valid_simple_model(self):
        """Test that a simple valid model passes validation."""
        code = '''
class Employee(BaseModel):
    """An employee record."""
    name: str
    age: int
'''
        _validate_code_safety(code)  # Should not raise

    def test_valid_model_with_distributions(self):
        """Test that a model with distribution annotations passes validation."""
        code = '''
class Employee(BaseModel):
    """An employee record."""
    name: str
    salary: Annotated[int, Normal(mean=50000, std=15000)]
    department: Annotated[str, Categorical(weights={"Eng": 0.5, "Sales": 0.5})]
'''
        _validate_code_safety(code)  # Should not raise

    def test_valid_model_with_field_constraints(self):
        """Test that Field() constraints are allowed."""
        code = '''
class Person(BaseModel):
    """A person."""
    name: str = Field(min_length=1, max_length=100)
    age: int = Field(ge=0, le=150)
'''
        _validate_code_safety(code)  # Should not raise

    def test_valid_model_with_validator(self):
        """Test that field validators are allowed."""
        code = '''
class User(BaseModel):
    """A user."""
    email: str

    @field_validator("email")
    @classmethod
    def validate_email(cls, v):
        if "@" not in v:
            raise ValueError("Invalid email")
        return v
'''
        _validate_code_safety(code)  # Should not raise

    def test_reject_import_statement(self):
        """Test that import statements are rejected."""
        code = """
import os

class Model(BaseModel):
    name: str
"""
        with pytest.raises(CodeValidationError, match="Import statements not allowed"):
            _validate_code_safety(code)

    def test_reject_from_import(self):
        """Test that from...import statements are rejected."""
        code = """
from os import system

class Model(BaseModel):
    name: str
"""
        with pytest.raises(CodeValidationError, match="Import statements not allowed"):
            _validate_code_safety(code)

    def test_reject_exec_call(self):
        """Test that exec() calls are rejected."""
        code = """
class Model(BaseModel):
    name: str = exec("print('evil')")
"""
        with pytest.raises(CodeValidationError, match="Forbidden"):
            _validate_code_safety(code)

    def test_reject_eval_call(self):
        """Test that eval() calls are rejected."""
        code = """
class Model(BaseModel):
    name: str

    def bad(self):
        return eval("1+1")
"""
        with pytest.raises(CodeValidationError, match="Forbidden"):
            _validate_code_safety(code)

    def test_reject_open_call(self):
        """Test that open() calls are rejected."""
        code = """
class Model(BaseModel):
    name: str

    def bad(self):
        return open("/etc/passwd")
"""
        with pytest.raises(CodeValidationError, match="Forbidden"):
            _validate_code_safety(code)

    def test_reject_dunder_attribute(self):
        """Test that dunder attribute access is rejected."""
        code = """
class Model(BaseModel):
    name: str

    def bad(self):
        return self.__class__.__bases__
"""
        with pytest.raises(CodeValidationError, match="Dunder attribute"):
            _validate_code_safety(code)

    def test_reject_globals_access(self):
        """Test that globals() is rejected."""
        code = """
class Model(BaseModel):
    name: str

    def bad(self):
        return globals()
"""
        with pytest.raises(CodeValidationError, match="Forbidden"):
            _validate_code_safety(code)

    def test_reject_builtins_access(self):
        """Test that __builtins__ access is rejected."""
        code = """
class Model(BaseModel):
    name: str

x = __builtins__
"""
        with pytest.raises(CodeValidationError, match="Forbidden"):
            _validate_code_safety(code)


class TestModelCodeExecution:
    """Test the safe execution of validated model code."""

    def test_execute_simple_model(self):
        """Test executing a simple model definition."""
        code = '''
class TestModel(BaseModel):
    """A test model."""
    name: str
    value: int
'''
        Model = _execute_model_code(code)

        assert Model.__name__ == "TestModel"
        assert "name" in Model.model_fields
        assert "value" in Model.model_fields

    def test_execute_model_with_distributions(self):
        """Test executing a model with distribution annotations."""
        code = '''
class Employee(BaseModel):
    """Employee model."""
    salary: Annotated[int, Normal(mean=50000, std=15000)]
    age: Annotated[int, Uniform(min=18, max=65)]
'''
        Model = _execute_model_code(code)

        assert Model.__name__ == "Employee"
        assert "salary" in Model.model_fields
        assert "age" in Model.model_fields

    def test_execute_model_with_optional_fields(self):
        """Test executing a model with optional fields."""
        code = '''
class Person(BaseModel):
    """A person."""
    name: str
    nickname: Optional[str] = None
'''
        Model = _execute_model_code(code)

        # Should be able to create instance without nickname
        instance = Model(name="John")
        assert instance.name == "John"
        assert instance.nickname is None

    def test_execute_model_with_field_constraints(self):
        """Test that Field constraints are applied."""
        code = '''
class Bounded(BaseModel):
    """Model with constraints."""
    value: int = Field(ge=0, le=100)
'''
        Model = _execute_model_code(code)

        # Valid value should work
        instance = Model(value=50)
        assert instance.value == 50

        # Invalid value should raise
        with pytest.raises(Exception):  # Pydantic ValidationError
            Model(value=150)

    def test_syntax_error_raises_validation_error(self):
        """Test that syntax errors are caught."""
        code = """
class Bad(BaseModel)
    name: str  # Missing colon after BaseModel
"""
        with pytest.raises(CodeValidationError, match="syntax error"):
            _validate_code_safety(code)

    def test_no_model_found_raises_error(self):
        """Test error when no model class is found."""
        code = """
x = 1
y = 2
"""
        _validate_code_safety(code)  # This passes (just assignments)

        with pytest.raises(ValueError, match="No Pydantic model class found"):
            _execute_model_code(code)

    def test_execute_model_with_correlations(self):
        """Test executing a model with correlations."""
        code = '''
class Employee(BaseModel):
    """Employee model with correlations."""
    age: Annotated[int, Uniform(min=22, max=65)]
    salary: Annotated[float, Normal(mean=75000, std=20000)]

    __correlations__ = Correlations(
        ("age", "salary", 0.5),
    )
'''
        Model = _execute_model_code(code)

        assert Model.__name__ == "Employee"
        assert hasattr(Model, "__correlations__")
        from gendantic import Correlations

        assert isinstance(Model.__correlations__, Correlations)


class TestExtendModelHelpers:
    """Test helper functions for extend_model."""

    def test_annotation_to_string_simple(self):
        """Test converting simple type annotations to strings."""
        from gendantic.model_generator import _annotation_to_string

        assert _annotation_to_string(str) == "str"
        assert _annotation_to_string(int) == "int"
        assert _annotation_to_string(float) == "float"

    def test_annotation_to_string_annotated(self):
        """Test converting Annotated types to strings."""
        from typing import Annotated

        from gendantic import Normal
        from gendantic.model_generator import _annotation_to_string

        ann = Annotated[int, Normal(mean=50000, std=15000)]
        result = _annotation_to_string(ann)

        assert "Annotated" in result
        assert "int" in result
        assert "Normal" in result

    def test_get_model_source_repr(self):
        """Test generating source representation of a model."""
        from typing import Annotated

        from pydantic import BaseModel

        from gendantic import Normal, Uniform
        from gendantic.model_generator import _get_model_source_repr

        class TestEmployee(BaseModel):
            """An employee."""

            age: Annotated[int, Uniform(min=22, max=65)]
            salary: Annotated[float, Normal(mean=75000, std=20000)]
            name: str

        dist_specs = {
            "age": Uniform(min=22, max=65),
            "salary": Normal(mean=75000, std=20000),
        }
        source = _get_model_source_repr(TestEmployee, dist_specs)

        assert "class TestEmployee(BaseModel)" in source
        assert "age:" in source
        assert "salary:" in source
        assert "name:" in source

    def test_generate_extended_model_code(self):
        """Test generating code for extended model with correlations."""
        from pydantic import BaseModel

        from gendantic.model_generator import _generate_extended_model_code

        class TestModel(BaseModel):
            pass

        model_code = """class TestModel(BaseModel):
    age: Annotated[int, Uniform(min=22, max=65)]
    salary: Annotated[float, Normal(mean=75000, std=20000)]"""

        correlations = [
            {"field1": "age", "field2": "salary", "correlation": 0.5},
        ]

        extended = _generate_extended_model_code(TestModel, model_code, correlations)

        assert "__correlations__" in extended
        assert "Correlations(" in extended
        assert '"age"' in extended
        assert '"salary"' in extended
        assert "0.5" in extended

    def test_generate_extended_model_code_with_copula(self):
        """Test generating code with explicit copula types."""
        from pydantic import BaseModel

        from gendantic.model_generator import _generate_extended_model_code

        class TestModel(BaseModel):
            pass

        model_code = """class TestModel(BaseModel):
    performance: Annotated[float, Beta(alpha=5, beta=2)]
    bonus: Annotated[float, Normal(mean=5000, std=2000)]"""

        correlations = [
            {
                "field1": "performance",
                "field2": "bonus",
                "correlation": 0.7,
                "copula_type": "gumbel",
            },
        ]

        extended = _generate_extended_model_code(TestModel, model_code, correlations)

        assert "__correlations__" in extended
        assert '"gumbel"' in extended

    def test_generate_extended_model_code_no_correlations(self):
        """Test that empty correlations returns original code."""
        from pydantic import BaseModel

        from gendantic.model_generator import _generate_extended_model_code

        class TestModel(BaseModel):
            pass

        model_code = """class TestModel(BaseModel):
    age: int"""

        extended = _generate_extended_model_code(TestModel, model_code, [])

        assert extended == model_code


class TestExtendModelValidation:
    """Test validation for extend_model."""

    @pytest.mark.asyncio
    async def test_extend_model_no_distributions_error(self):
        """Test that extend_model raises error when no distribution fields."""
        from pydantic import BaseModel

        from gendantic import extend_model_with_correlations

        class PlainModel(BaseModel):
            name: str
            age: int

        with pytest.raises(ValueError, match="no distribution-annotated fields"):
            await extend_model_with_correlations(PlainModel)

    @pytest.mark.asyncio
    async def test_extend_model_single_distribution_error(self):
        """Test that extend_model requires at least 2 distribution fields."""
        from typing import Annotated

        from pydantic import BaseModel

        from gendantic import Normal, extend_model_with_correlations

        class SingleDistModel(BaseModel):
            salary: Annotated[float, Normal(mean=75000, std=20000)]
            name: str

        with pytest.raises(ValueError, match="only 1 distribution field"):
            await extend_model_with_correlations(SingleDistModel)


class TestExtendModelWithDistributions:
    """Test the extend_model_with_distributions helper functions."""

    def test_get_basic_model_source_repr(self):
        """Test generating source representation of a basic model."""
        from pydantic import BaseModel, Field

        from gendantic.model_generator import _get_basic_model_source_repr

        class BasicEmployee(BaseModel):
            name: str
            age: int = Field(ge=18, le=100)
            salary: float

        source = _get_basic_model_source_repr(BasicEmployee)

        assert "class BasicEmployee(BaseModel)" in source
        assert "name: str" in source
        assert "age: int" in source
        assert "salary: float" in source

    def test_get_basic_model_source_repr_with_constraints(self):
        """Test that Field constraints are preserved in source repr."""
        from pydantic import BaseModel, Field

        from gendantic.model_generator import _get_basic_model_source_repr

        class ConstrainedModel(BaseModel):
            value: int = Field(ge=0, le=100)

        source = _get_basic_model_source_repr(ConstrainedModel)

        assert "ge=0" in source
        assert "le=100" in source
