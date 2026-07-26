from __future__ import annotations

import json
import logging
from typing import Any, Optional

import httpx

from src.config import settings

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised when LLM API call fails."""


class LLMClient:
    """OpenAI-compatible LLM client (works with DeepSeek, OpenAI, etc.)."""

    def __init__(self, api_key=None, base_url=None, model=None) -> None:
        self.api_key = api_key or settings.openai_api_key
        self.base_url = (base_url or settings.llm_base_url).rstrip("/")
        self.model = model or settings.llm_model
        self.max_tokens = settings.llm_max_tokens
        self.temperature = settings.llm_temperature

    def chat(
        self,
        system: str,
        user: str,
        response_format: Optional[dict] = None,
    ) -> str:
        """Send a chat completion request and return the raw assistant content."""
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if response_format is not None:
            body["response_format"] = response_format

        logger.info(
            "LLM request: model=%s, system=%d chars, user=%d chars",
            self.model,
            len(system),
            len(user),
        )

        try:
            resp = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=180,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as e:
            detail = e.response.text[:500] if e.response else str(e)
            raise LLMError(f"API error ({e.response.status_code}): {detail}") from e
        except httpx.RequestError as e:
            raise LLMError(f"Request failed: {e}") from e
        except json.JSONDecodeError as e:
            raise LLMError(f"Invalid JSON response: {e}") from e

        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError) as e:
            raise LLMError(f"Unexpected API response structure: {e}") from e

        if not isinstance(content, str):
            raise LLMError("Unexpected API response content type")
        return content

    def chat_json(
        self,
        system: str,
        user: str,
        response_schema: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Send a chat completion request and parse the response as JSON.

        Args:
            system: System prompt content.
            user: User prompt content.
            response_schema: Optional JSON schema to include in request
                             (for providers that support guided JSON).

        Returns:
            Parsed JSON dict from the LLM response.
        """
        # happyapi's JSON mode is picky about the literal lowercase token "json".
        # Add it explicitly so both json_object and json_schema requests remain accepted.
        system = system.rstrip() + "\n\nReturn the answer as json only."
        user = user.rstrip() + "\n\njson"

        # Most OpenAI-compatible providers support response_format for JSON mode.
        if response_schema is not None:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "strict": True,
                    "schema": response_schema,
                },
            }
        else:
            response_format = {"type": "json_object"}

        content = self.chat(system=system, user=user, response_format=response_format)

        # Parse JSON from content
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            # Sometimes the LLM wraps JSON in markdown code fences
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1] if "\n" in content else content[3:]
                content = content.rsplit("```", 1)[0]
                result = json.loads(content)
            else:
                raise LLMError(f"LLM returned non-JSON: {content[:300]}")

        return result

    @staticmethod
    def is_configured() -> bool:
        """Check if the API key is set."""
        return bool(settings.openai_api_key) and settings.openai_api_key not in (
            "",
            "sk-your-key-here",
        )

    @staticmethod
    def planner_is_configured() -> bool:
        """Check whether the dedicated planner credentials are set."""
        return bool(getattr(settings, "planner_api_key", "")) and getattr(settings, "planner_api_key", "") not in (
            "",
            "sk-your-key-here",
        )
