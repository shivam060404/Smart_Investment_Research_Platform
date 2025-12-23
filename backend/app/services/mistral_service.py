import json
import logging
from typing import Any, Optional

from mistralai import Mistral

from app.config import get_settings

logger = logging.getLogger(__name__)


class MistralServiceError(Exception):
    """Base exception for Mistral service errors."""
    pass


class MistralUnavailableError(MistralServiceError):
    """Raised when Mistral API is unavailable."""
    pass


class MistralTimeoutError(MistralServiceError):
    """Raised when Mistral API request times out."""
    pass


class MistralResponseError(MistralServiceError):
    """Raised when Mistral returns an invalid response."""
    pass


class MistralService:
    """Service for interacting with Mistral API for agent analysis."""

    DEFAULT_MODEL = "mistral-small-latest"
    DEFAULT_MAX_TOKENS = 2048
    DEFAULT_TEMPERATURE = 0.3

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
    ):
        """Initialize the Mistral service.

        Args:
            api_key: Mistral API key. Defaults to settings value.
            model: Model to use. Defaults to mistral-small-latest.
            timeout_seconds: Request timeout. Defaults to settings value.
        """
        settings = get_settings()
        self._api_key = api_key or settings.mistral_api_key
        self._model = model or self.DEFAULT_MODEL
        self._timeout_seconds = timeout_seconds or settings.agent_timeout_seconds
        self._client: Optional[Mistral] = None

    @property
    def client(self) -> Mistral:
        """Get or create Mistral client.

        Returns:
            Mistral client instance.

        Raises:
            MistralUnavailableError: If API key is not configured.
        """
        if not self._api_key:
            raise MistralUnavailableError("Mistral API key not configured")

        if self._client is None:
            self._client = Mistral(api_key=self._api_key)

        return self._client

    async def generate_analysis(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate analysis using Mistral API.

        Args:
            system_prompt: System prompt defining the agent's role and behavior.
            user_prompt: User prompt containing the data to analyze.
            max_tokens: Maximum tokens in response. Defaults to 2048.
            temperature: Sampling temperature. Defaults to 0.3.

        Returns:
            Generated analysis text.

        Raises:
            MistralUnavailableError: If Mistral API is unavailable.
            MistralTimeoutError: If request times out.
            MistralResponseError: If response is invalid.
        """
        max_tokens = max_tokens or self.DEFAULT_MAX_TOKENS
        temperature = temperature or self.DEFAULT_TEMPERATURE

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            logger.debug(f"Sending request to Mistral model: {self._model}")

            response = await self.client.chat.complete_async(
                model=self._model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            if not response.choices or len(response.choices) == 0:
                raise MistralResponseError("Empty response from Mistral API")

            content = response.choices[0].message.content

            if not content:
                raise MistralResponseError("No content in Mistral response")

            logger.debug(f"Received response from Mistral ({len(content)} chars)")
            return content

        except MistralServiceError:
            raise
        except Exception as e:
            error_msg = str(e).lower()

            if "timeout" in error_msg or "timed out" in error_msg:
                logger.error(f"Mistral API timeout: {e}")
                raise MistralTimeoutError(f"Mistral API request timed out: {e}")

            if "unauthorized" in error_msg or "401" in error_msg:
                logger.error(f"Mistral API authentication failed: {e}")
                raise MistralUnavailableError("Invalid Mistral API key")

            if "rate limit" in error_msg or "429" in error_msg:
                logger.error(f"Mistral API rate limited: {e}")
                raise MistralUnavailableError("Mistral API rate limit exceeded")

            logger.error(f"Mistral API error: {e}")
            raise MistralServiceError(f"Mistral API request failed: {e}")

    async def generate_structured_analysis(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> dict[str, Any]:
        """Generate structured JSON analysis using Mistral API.

        Args:
            system_prompt: System prompt defining the agent's role.
            user_prompt: User prompt containing the data to analyze.
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.

        Returns:
            Parsed JSON response as dictionary.

        Raises:
            MistralUnavailableError: If Mistral API is unavailable.
            MistralTimeoutError: If request times out.
            MistralResponseError: If response is invalid or not valid JSON.
        """
        # Append JSON instruction to system prompt
        json_system_prompt = (
            f"{system_prompt}\n\n"
            "IMPORTANT: You must respond with valid JSON only. "
            "Do not include any text before or after the JSON object."
        )

        content = await self.generate_analysis(
            system_prompt=json_system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Try to parse JSON from response
        try:
            # Handle potential markdown code blocks
            cleaned_content = content.strip()
            if cleaned_content.startswith("```json"):
                cleaned_content = cleaned_content[7:]
            if cleaned_content.startswith("```"):
                cleaned_content = cleaned_content[3:]
            if cleaned_content.endswith("```"):
                cleaned_content = cleaned_content[:-3]
            cleaned_content = cleaned_content.strip()

            return json.loads(cleaned_content)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Mistral JSON response: {e}")
            logger.debug(f"Raw response: {content[:500]}...")
            raise MistralResponseError(f"Invalid JSON in Mistral response: {e}")

    async def health_check(self) -> tuple[bool, Optional[str]]:
        """Check Mistral API connectivity.

        Returns:
            Tuple of (is_healthy, error_message).
        """
        try:
            # Simple test request
            response = await self.generate_analysis(
                system_prompt="You are a helpful assistant.",
                user_prompt="Respond with 'OK' only.",
                max_tokens=10,
                temperature=0.0,
            )
            return True, None
        except MistralUnavailableError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Health check failed: {e}"


# Global service instance
_mistral_service: Optional[MistralService] = None


def get_mistral_service() -> MistralService:
    """Get the global Mistral service instance.

    Returns:
        MistralService instance.
    """
    global _mistral_service
    if _mistral_service is None:
        _mistral_service = MistralService()
    return _mistral_service
