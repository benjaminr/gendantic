# Gendantic

Intelligent synthetic data generation using Pydantic models and LLMs.

## Quick Start

```python
import asyncio
from pydantic import BaseModel, Field
from gendantic import generate

class User(BaseModel):
    name: str = Field(min_length=2)
    age: int = Field(ge=18, le=100)
    email: str

async def main():
    users = await generate(User, count=5)
    for user in users:
        print(f"{user.name} ({user.age}) - {user.email}")

# Set OPENAI_API_KEY environment variable
asyncio.run(main())
```

## Features

- **LLM-Driven**: Intelligent analysis of your Pydantic models for realistic data generation
- **Async-First**: High-performance async API with batch generation support
- **Context-Aware**: Generate data tailored to specific contexts
- **Validator Compliant**: Automatically respects all Pydantic field validators
- **Multi-Provider**: Support for OpenAI and Anthropic models
- **Zero Configuration**: Works out of the box with minimal setup

## Installation

```bash
pip install gendantic
```

Or for development:

```bash
git clone <repository>
cd gendantic
uv venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv add -e .
```

## Usage

### Basic Usage

```python
import asyncio
from pydantic import BaseModel, EmailStr, Field, field_validator
from gendantic import generate

class Employee(BaseModel):
    first_name: str = Field(min_length=2, max_length=30)
    last_name: str = Field(min_length=2, max_length=30)
    email: EmailStr
    salary: int = Field(ge=30000, le=200000)
    department: str
    
    @field_validator("email")
    @classmethod
    def email_must_be_company_domain(cls, v):
        if not v.endswith("@mycompany.com"):
            raise ValueError("Must use company email")
        return v

async def main():
    # Generate employees - automatically respects validators
    employees = await generate(Employee, count=10)
    print(f"Generated {len(employees)} employees")

asyncio.run(main())
```

### Context-Aware Generation

```python
# Generate data for specific business contexts
fintech_employees = await generate(
    Employee, 
    count=20, 
    context="Fast-growing London fintech startup"
)

bank_employees = await generate(
    Employee,
    count=15,
    context="Traditional UK high-street bank"
)
```

### Batch Generation

```python
from gendantic import generate_batch

# Generate multiple contexts concurrently
contexts = [
    "Tech startup in London",
    "Manufacturing company in Manchester", 
    "Consulting firm in Edinburgh"
]

batches = await generate_batch(Employee, contexts, count=5)
# Returns 3 lists of 5 employees each
```

## Configuration

Set your API key as an environment variable or in a `.env` file:

```bash
# For OpenAI (recommended)
export OPENAI_API_KEY="your-api-key"

# For Anthropic
export ANTHROPIC_API_KEY="your-api-key"
```

**Or create a `.env` file:**
```env
OPENAI_API_KEY=your-api-key
```

If both keys are set, OpenAI will be used by default. Use the `provider` parameter to specify:

```python
# Use specific provider
employees = await generate(Employee, count=5, provider="anthropic")
```

## Key Features

### Intelligent Model Analysis
Gendantic uses LLMs to intelligently analyse your Pydantic models, understanding field relationships, constraints, and business context to generate realistic data.

### Validator Compliance
All generated data automatically passes your Pydantic field validators. No more validation errors from synthetic data.

### Context-Aware Generation
Provide business context to generate more realistic data patterns:

```python
# Different contexts produce different realistic patterns
startup_data = await generate(Employee, context="Silicon Valley startup")
bank_data = await generate(Employee, context="Traditional London bank")
```

### High Performance
Async-first design with concurrent batch processing for generating large datasets efficiently.

## Example

Run the included example:

```bash
uv run python example.py
```

## Development

```bash
# Install development dependencies
uv add --dev pytest mypy ruff

# Run tests
uv run pytest

# Type checking
uv run mypy src/

# Format and lint
uv run ruff format
uv run ruff check
```

## License

MIT