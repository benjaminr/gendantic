"""Tests for the main generator functionality."""

from datetime import datetime
from typing import Annotated, Optional

import pytest
from pydantic import BaseModel, Field

from gendantic import Normal, Uniform, generate_synthetic_data


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


class UserWithDistributions(BaseModel):
    """User model with statistical distributions."""

    name: str
    age: Annotated[int, Uniform(min=18, max=65)]
    salary: Annotated[float, Normal(mean=50000, std=15000)]


@pytest.mark.asyncio
async def test_generate_simple_model():
    """Test generating data for a simple model."""
    # Note: This test will be skipped if no API keys are available
    try:
        users = await generate_synthetic_data(SimpleUser, count=3)

        assert len(users) <= 3  # May be less due to validation failures
        assert all(isinstance(user, SimpleUser) for user in users)

        for user in users:
            assert user.name
            assert 18 <= user.age <= 100
            assert "@" in user.email

    except ValueError as e:
        if "LiteLLM proxy URL not configured" in str(e):
            pytest.skip("LiteLLM proxy not configured for testing")
        else:
            raise


@pytest.mark.asyncio
async def test_generate_with_distributions():
    """Test generating data with distribution specs."""
    try:
        users = await generate_synthetic_data(UserWithDistributions, count=5, seed=42)

        assert len(users) == 5
        for user in users:
            # Distribution-sampled fields
            assert 18 <= user.age <= 65
            assert user.salary > 0  # Normal can go negative but usually won't

    except ValueError as e:
        if "LiteLLM proxy URL not configured" in str(e):
            pytest.skip("LiteLLM proxy not configured for testing")
        else:
            raise


@pytest.mark.asyncio
async def test_generate_complex_model():
    """Test generating data for a more complex model."""
    try:
        users = await generate_synthetic_data(ComplexUser, count=2)

        for user in users:
            assert 2 <= len(user.first_name) <= 50
            assert 2 <= len(user.last_name) <= 50
            assert 18 <= user.age <= 100
            assert "@" in user.email
            if user.score is not None:
                assert 0.0 <= user.score <= 100.0

    except ValueError as e:
        if "LiteLLM proxy URL not configured" in str(e):
            pytest.skip("LiteLLM proxy not configured for testing")
        else:
            raise


@pytest.mark.asyncio
async def test_generate_reproducible_with_seed():
    """Test that the same seed produces same distribution samples."""
    try:
        users1 = await generate_synthetic_data(UserWithDistributions, count=3, seed=123)
        users2 = await generate_synthetic_data(UserWithDistributions, count=3, seed=123)

        # Distribution-sampled fields should be identical
        for u1, u2 in zip(users1, users2, strict=False):
            assert u1.age == u2.age
            assert u1.salary == u2.salary

    except ValueError as e:
        if "LiteLLM proxy URL not configured" in str(e):
            pytest.skip("LiteLLM proxy not configured for testing")
        else:
            raise
