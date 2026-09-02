"""LLM client using LiteLLM proxy for unified model access."""

import json
import os
from typing import Any

import litellm
from dotenv import load_dotenv

from .prompts import load_prompt

# Load .env file if it exists
load_dotenv()


def _make_schema_strict(schema: Any) -> Any:
    """Recursively enforce OpenAI/Azure strict structured-output rules.

    Strict mode (used by Azure-routed OpenAI models) requires every object to
    set ``additionalProperties: false`` and to list *all* of its properties in
    ``required``. This walks the schema — including nested objects, array items,
    ``$defs`` and the ``anyOf``/``allOf``/``oneOf`` combinators — and applies
    those rules so a plain Pydantic JSON schema is accepted by strict endpoints.
    """
    if isinstance(schema, dict):
        result = {k: _make_schema_strict(v) for k, v in schema.items()}
        if result.get("type") == "object" and "properties" in result:
            result["additionalProperties"] = False
            result["required"] = list(result["properties"].keys())
        return result
    if isinstance(schema, list):
        return [_make_schema_strict(item) for item in schema]
    return schema


class LiteLLMClient:
    """LiteLLM client for calling models through a LiteLLM proxy.

    Configuration via environment variables:
        LITELLM_API_BASE: Base URL for the LiteLLM proxy (e.g., http://localhost:4000)
        LITELLM_API_KEY: API key for the proxy (if required)
        LITELLM_MODEL: Default model to use (e.g., gpt-4o-mini, claude-3-5-sonnet-20241022)
    """

    def __init__(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ):
        self.api_base = api_base or os.getenv("LITELLM_API_BASE")
        self.api_key = api_key or os.getenv("LITELLM_API_KEY", "")
        self.model = model or os.getenv("LITELLM_MODEL", "gpt-4o-mini")

        if not self.api_base:
            raise ValueError(
                "LiteLLM proxy URL not configured. "
                "Set LITELLM_API_BASE environment variable (e.g., http://localhost:4000)"
            )

    async def generate_structured(
        self, schema: dict[str, Any], prompt: str, count: int = 1
    ) -> list[dict[str, Any]]:
        """Generate structured data matching the schema asynchronously.

        Uses LiteLLM's structured output support for reliable JSON responses.
        """
        response_text = None
        try:
            # Build a wrapper schema that ensures we get {"items": [...]}
            # The input schema is expected to be {"type": "array", "items": {...}}
            item_schema = _make_schema_strict(schema.get("items", schema))

            wrapped_schema = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_response",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "items": {
                                "type": "array",
                                "items": item_schema,
                            }
                        },
                        "required": ["items"],
                        "additionalProperties": False,
                    },
                },
            }

            response = await litellm.acompletion(
                model=self.model,
                api_base=self.api_base,
                api_key=self.api_key,
                response_format=wrapped_schema,
                messages=[
                    {
                        "role": "system",
                        "content": load_prompt("llm_system").format(
                            count=count, schema=json.dumps(item_schema)
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )

            response_text = response.choices[0].message.content
            if not response_text or not response_text.strip():
                raise ValueError(
                    f"Empty response from LLM. "
                    f"Model: {self.model}, API base: {self.api_base}"
                )

            parsed_response = json.loads(response_text)
            if "items" not in parsed_response:
                raise ValueError(
                    f"Response missing 'items' field. Got keys: {list(parsed_response.keys())}"
                )

            items = parsed_response["items"]
            if not isinstance(items, list):
                raise ValueError(
                    f"Response 'items' field is not a list, got {type(items).__name__}"
                )
            return items

        except json.JSONDecodeError as e:
            preview = repr(response_text[:200]) if response_text else "None"
            raise ValueError(
                f"Failed to parse JSON response: {e}. Response preview: {preview}"
            ) from e
        except litellm.exceptions.APIConnectionError as e:
            raise ValueError(
                f"Could not connect to LiteLLM proxy at {self.api_base}. "
                f"Is the proxy running? Error: {e}"
            ) from e
        except litellm.exceptions.AuthenticationError as e:
            raise ValueError(
                f"Authentication failed for LiteLLM proxy. "
                f"Check LITELLM_API_KEY. Error: {e}"
            ) from e


# Module-level client instance (lazy initialization)
_client: LiteLLMClient | None = None


def get_client(
    api_base: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> LiteLLMClient:
    """Get the LiteLLM client instance.

    Args:
        api_base: LiteLLM proxy base URL (or set LITELLM_API_BASE)
        api_key: API key for proxy (or set LITELLM_API_KEY)
        model: Model identifier (or set LITELLM_MODEL)

    Returns:
        LiteLLMClient instance
    """
    global _client

    # If custom params provided, create new client
    if api_base or api_key or model:
        return LiteLLMClient(api_base=api_base, api_key=api_key, model=model)

    # Otherwise use/create singleton
    if _client is None:
        _client = LiteLLMClient()
    return _client
