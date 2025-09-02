import asyncio
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, EmailStr, Field, field_validator

from gendantic import generate, generate_batch


class Department(str, Enum):
    """Employee department options."""
    ENGINEERING = "engineering"
    PRODUCT = "product" 
    MARKETING = "marketing"
    SALES = "sales"


class Employee(BaseModel):
    """Employee at a modern company."""
    
    first_name: str = Field(min_length=2, max_length=30)
    last_name: str = Field(min_length=2, max_length=30)
    email: EmailStr = Field(description="Work email address")
    department: Department
    job_title: str
    salary: int = Field(ge=30000, le=200000, description="Annual salary in GBP")
    years_experience: int = Field(ge=0, le=40)
    is_manager: bool = False
    start_date: datetime
    performance_rating: Optional[float] = Field(ge=1.0, le=5.0, default=None)
    
    @field_validator("email")
    @classmethod
    def email_must_be_company_domain(cls, v):
        """Work emails must use company domain."""
        if not v.endswith("@mycompany.com"):
            raise ValueError("Work email must use @mycompany.com domain")
        return v


async def main():
    """Demonstrate Gendantic's intelligent synthetic data generation."""
    
    print("🚀 Gendantic - LLM-Driven Synthetic Data Generation")
    print("=" * 60)
    
    try:
        # 1. Basic generation - LLM automatically analyses the model
        print("\nBasic Generation")
        print("The LLM automatically analyses your Pydantic model:")
        
        basic_employees = await generate(Employee, count=3)
        
        print(f"Generated {len(basic_employees)} employees")
        for emp in basic_employees:
            print(f"  • {emp.first_name} {emp.last_name} - {emp.job_title}")
            print(f"    {emp.email} | {emp.department.value.title()}")
            print(f"    £{emp.salary:,} | {emp.years_experience}y experience")
            print()
        
        # 2. Context-aware generation - much more realistic
        print("Context-Aware Generation")
        print("Provide business context for more realistic data:")
        
        fintech_employees = await generate(
            Employee, 
            count=3, 
            context="Fast-growing London fintech startup with diverse international team"
        )
        
        print("London fintech startup employees:")
        for emp in fintech_employees:
            print(f"  • {emp.first_name} {emp.last_name} - {emp.job_title}")
            print(f"    {emp.email} | {emp.department.value.title()}")
            print(f"    £{emp.salary:,} | Rating: {emp.performance_rating or 'N/A'}/5")
            print()
        
        # 3. Demonstrate async generation
        print("Async Generation")
        print("Generate data asynchronously:")
        
        async_employees = await generate(Employee, count=2, context="Modern UK startup")
        
        print("Async generated employees:")
        for emp in async_employees:
            print(f"  • {emp.first_name} {emp.last_name} - {emp.job_title}")
            print(f"    {emp.email} | £{emp.salary:,}")
            print()
        
    except ValueError as e:
        print(f"Error: {e}")
        print("\nTo run this example, set either:")
        print("- OPENAI_API_KEY environment variable")
        print("- ANTHROPIC_API_KEY environment variable")


if __name__ == "__main__":
    asyncio.run(main())
