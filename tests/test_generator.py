"""Tests for the main generator functionality."""

from datetime import datetime
from typing import Optional

import pytest
from pydantic import BaseModel, Field

from gendantic import generate


class SimpleUser(BaseModel):
    """Simple user model for testing."""

    name: str
    age: int = Field(ge=18, le=100)
    email: str


class ComplexUser(BaseModel):
    """More complex user model for testing."""

    first_name: str = Field(min_length=2, max_length=50)
    last_name: str = Field(min_length=2, max_length=50)
    age: int = Field(ge=18, le=100)
    email: str
    is_active: bool = True
    score: Optional[float] = Field(None, ge=0.0, le=100.0)
    created_at: Optional[datetime] = None


def test_generate_simple_model():
    """Test generating data for a simple model."""
    # Note: This test will be skipped if no API keys are available
    try:
        users = generate(SimpleUser, count=3)

        assert len(users) <= 3  # May be less due to validation failures
        assert all(isinstance(user, SimpleUser) for user in users)

        for user in users:
            assert user.name
            assert 18 <= user.age <= 100
            assert "@" in user.email

    except ValueError as e:
        if "No LLM provider configured" in str(e):
            pytest.skip("No API keys configured for testing")
        else:
            raise


def test_generate_with_overrides():
    """Test generating data with field overrides."""
    try:
        users = generate(SimpleUser, count=2, age_range=(25, 35))

        for user in users:
            assert 25 <= user.age <= 35

    except ValueError as e:
        if "No LLM provider configured" in str(e):
            pytest.skip("No API keys configured for testing")
        else:
            raise


def test_generate_complex_model():
    """Test generating data for a more complex model."""
    try:
        users = generate(ComplexUser, count=2)

        for user in users:
            assert 2 <= len(user.first_name) <= 50
            assert 2 <= len(user.last_name) <= 50
            assert 18 <= user.age <= 100
            assert "@" in user.email
            if user.score is not None:
                assert 0.0 <= user.score <= 100.0

    except ValueError as e:
        if "No LLM provider configured" in str(e):
            pytest.skip("No API keys configured for testing")
        else:
            raise
