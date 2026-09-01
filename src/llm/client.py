from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

import httpx

from src.config import settings

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised when LLM API call fails."""


class LLMClient:

    def __init__(self, api_key=None, base_url=None, model=None) -> None:
        self.api_key = api_key or settings.llm_api_key
        self.base_url = (base_url or settings.llm_base_url).rstrip("/")
        self.model = model or settings.llm_model
        self.max_tokens = settings.llm_max_tokens
        self.temperature = settings.llm_temperature
        self.max_retries = 2

    def _candidate_models(self) -> list[str]:
        fallbacks = [item.strip() for item in settings.llm_fallback_models.split(",") if item.strip()]
        return list(dict.fromkeys([self.model, *fallbacks]))

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

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_error = None
        attempted_models = []
        data = None
        for model_index, model in enumerate(self._candidate_models()):
            body["model"] = model
            attempted_models.append(model)
            for attempt in range(self.max_retries + 1):
                try:
                    resp = httpx.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=body,
                        timeout=httpx.Timeout(180.0, connect=20.0),
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    break
                except httpx.HTTPStatusError as e:
                    response = e.response
                    detail = response.text[:800] if response is not None else str(e)
                    status = response.status_code if response is not None else "unknown"
                    code = ""
                    try:
                        error = response.json().get("error", {}) if response is not None else {}
                        code = error.get("code", "")
                        message = error.get("message", detail)
                        detail = f"{message} (code={code})" if code else str(message)
                    except (ValueError, AttributeError):
                        pass
                    last_error = LLMError(
                        f"API error ({status}, code={code or 'unknown'}): {detail}. "
                        "请检查 LLM_BASE_URL、LLM_MODEL 和上游服务可用性。"
                    )
                    # 429/5xx and upstream gateway failures are commonly transient.
                    retryable = status == 429 or status >= 500 or code in {
                        "upstream_blocked", "upstream_error", "timeout"
                    }
                    if attempt >= self.max_retries:
                        if code == "upstream_blocked" and model_index + 1 < len(self._candidate_models()):
                            logger.warning("Model %s is blocked; trying fallback model", model)
                            break
                        raise last_error from e
                    if not retryable:
                        raise last_error from e
                    delay = 2 ** attempt
                    logger.warning("LLM API attempt %d failed (%s); retrying in %ss", attempt + 1, code or status, delay)
                    time.sleep(delay)
                except httpx.RequestError as e:
                    last_error = LLMError(f"Request failed: {e}. 请检查网络和 LLM_BASE_URL。")
                    if attempt >= self.max_retries:
                        raise last_error from e
                    time.sleep(2 ** attempt)
                except json.JSONDecodeError as e:
                    raise LLMError(f"Invalid JSON response: {e}") from e
            if data is not None:
                break
        if data is None:
            models = ", ".join(attempted_models)
            raise LLMError(f"所有候选模型均不可用（已尝试: {models}）。{last_error}")

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

        try:
            content = self.chat(system=system, user=user, response_format=response_format)
        except LLMError as exc:
            # Some OpenAI-compatible gateways reject response_format for Gemini
            # models. Retry once without it; the prompt still requests JSON.
            if response_format is None or "response_format" not in str(exc).lower():
                raise
            logger.warning("JSON response_format rejected; retrying without it")
            content = self.chat(system=system, user=user)

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
        return bool(settings.llm_api_key) and settings.llm_api_key not in (
            "",
            "sk-your-key-here",
        )

    @staticmethod
    def planner_is_configured() -> bool:
        """Check whether the dedicated planner credentials are set."""
        return bool(settings.llm_api_key) and settings.llm_api_key not in (
            "",
            "sk-your-key-here",
        )
