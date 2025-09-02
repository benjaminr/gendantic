import asyncio
import json
import os
from abc import ABC, abstractmethod
from typing import Any, List

import anthropic
import openai
from dotenv import load_dotenv

from .prompts import load_prompt


# Load .env file if it exists
load_dotenv()


class LLMClient(ABC):
    """Abstract base class for async LLM clients."""

    @abstractmethod
    async def generate_structured(
        self, schema: dict[str, Any], prompt: str, count: int = 1
    ) -> list[dict[str, Any]]:
        """Generate structured data matching the schema asynchronously."""
        pass
    
    async def generate_batch(
        self, schema: dict[str, Any], prompts: List[str], count: int = 1
    ) -> List[list[dict[str, Any]]]:
        """Generate multiple structured data batches concurrently."""
        tasks = [
            self.generate_structured(schema=schema, prompt=prompt, count=count)
            for prompt in prompts
        ]
        return await asyncio.gather(*tasks)


class OpenAIClient(LLMClient):
    """OpenAI async client using structured outputs."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

    async def generate_structured(
        self, schema: dict[str, Any], prompt: str, count: int = 1
    ) -> list[dict[str, Any]]:
        """Generate structured data using OpenAI async client."""
        async_client = openai.AsyncOpenAI(api_key=self.api_key)
        
        try:
            response = await async_client.chat.completions.create(
                model="gpt-4o-mini",
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": load_prompt("llm_system").format(
                            count=count,
                            schema=json.dumps(schema)
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )

            response_text = response.choices[0].message.content
            if not response_text:
                raise ValueError("Empty response from OpenAI")

            parsed_response = json.loads(response_text)
            if "items" not in parsed_response:
                raise ValueError("Response missing 'items' field")

            items = parsed_response["items"]
            if not isinstance(items, list):
                raise ValueError("Response 'items' field is not a list")
            return items
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {e}") from e
        finally:
            await async_client.close()


class AnthropicClient(LLMClient):
    """Anthropic async client using JSON mode."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

    async def generate_structured(
        self, schema: dict[str, Any], prompt: str, count: int = 1
    ) -> list[dict[str, Any]]:
        """Generate structured data using Anthropic async client."""
        async_client = anthropic.AsyncAnthropic(api_key=self.api_key)
        
        try:
            response = await async_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4000,
                system=f"Generate {count} synthetic data records as a JSON array following this schema: {json.dumps(schema)}. Respond with only the JSON array, no explanation.",
                messages=[{"role": "user", "content": prompt}],
            )

            # Parse JSON response
            text_content = None
            for content_block in response.content:
                if hasattr(content_block, "text"):
                    text_content = content_block.text
                    break

            if text_content is None:
                raise ValueError("No text content found in response")

            parsed_response = json.loads(text_content)
            if not isinstance(parsed_response, list):
                raise ValueError("Response is not a JSON array")
            return parsed_response
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {e}") from e
        finally:
            await async_client.close()


def get_client(provider: str | None = None) -> LLMClient:
    """Get appropriate LLM client based on provider or available API keys."""
    if provider == "openai":
        return OpenAIClient()
    elif provider == "anthropic":
        return AnthropicClient()

    # Auto-detect based on available API keys
    if os.getenv("OPENAI_API_KEY"):
        return OpenAIClient()
    elif os.getenv("ANTHROPIC_API_KEY"):
        return AnthropicClient()
    else:
        raise ValueError(
            "No LLM provider configured. Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable, "
            "or specify provider parameter."
        )
