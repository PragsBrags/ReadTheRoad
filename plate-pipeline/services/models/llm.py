"""
LLM Clients — OpenAI and Ollama implementations for plate text correction.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from services.config import PipelineConfig
from services.models.base import LLMClient

logger = logging.getLogger(__name__)


class OpenAIClient(LLMClient):
    """OpenAI LLM client for plate text correction."""

    def __init__(self, config: PipelineConfig):
        self._config = config.llm.openai
        self._llm_config = config.llm
        self._client = None
        self._available = False

    def load(self) -> None:
        """Initialize OpenAI client."""
        api_key = os.environ.get(self._config.api_key_env, "")
        if not api_key:
            logger.warning(
                f"OpenAI API key not found in env var '{self._config.api_key_env}'. "
                f"LLM features disabled."
            )
            return

        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key)
            self._available = True
            logger.info(f"OpenAI client initialized (model={self._config.model})")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")

    def correct_plate_text(
        self,
        ocr_text: str,
        image_bytes: Optional[bytes] = None,
    ) -> dict[str, Any]:
        """Correct plate text using OpenAI."""
        if not self._available or not self._client:
            return {"corrected_text": ocr_text, "confidence": 0.0}

        import base64

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a license plate text correction system. "
                    "Given OCR output from a license plate image, correct any "
                    "misread characters. Common confusions: 0/O, 1/I/L, 5/S, "
                    "8/B, 2/Z. Return ONLY the corrected plate text, nothing else."
                ),
            },
        ]

        user_content = []

        if image_bytes:
            b64 = base64.b64encode(image_bytes).decode("utf-8")
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })

        prompt = f"OCR result: '{ocr_text}'" if ocr_text else "Read the license plate text from this image."
        user_content.append({"type": "text", "text": prompt})

        messages.append({"role": "user", "content": user_content})

        try:
            response = self._client.chat.completions.create(
                model=self._config.model,
                messages=messages,
                max_tokens=self._config.max_tokens,
                temperature=self._config.temperature,
            )

            corrected = response.choices[0].message.content.strip()

            return {
                "corrected_text": corrected,
                "confidence": 0.9,
                "reasoning": f"LLM correction of '{ocr_text}' → '{corrected}'",
                "model": self._config.model,
                "tokens_used": response.usage.total_tokens if response.usage else 0,
            }

        except Exception as e:
            logger.error(f"OpenAI correction failed: {e}")
            return {
                "corrected_text": ocr_text,
                "confidence": 0.0,
                "error": str(e),
            }

    def is_available(self) -> bool:
        return self._available


class OllamaClient(LLMClient):
    """Ollama LLM client for local inference."""

    def __init__(self, config: PipelineConfig):
        self._config = config.llm.ollama
        self._available = False

    def load(self) -> None:
        """Check if Ollama is reachable."""
        import urllib.request
        try:
            req = urllib.request.Request(f"{self._config.base_url}/api/tags")
            urllib.request.urlopen(req, timeout=5)
            self._available = True
            logger.info(f"Ollama available at {self._config.base_url}")
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")

    def correct_plate_text(
        self,
        ocr_text: str,
        image_bytes: Optional[bytes] = None,
    ) -> dict[str, Any]:
        """Correct plate text using local Ollama model."""
        if not self._available:
            return {"corrected_text": ocr_text, "confidence": 0.0}

        import base64
        import json
        import urllib.request

        prompt = (
            f"You are a license plate text correction system. "
            f"OCR result: '{ocr_text}'. Correct any misread characters. "
            f"Return ONLY the corrected plate text."
        )

        payload: dict[str, Any] = {
            "model": self._config.model,
            "prompt": prompt,
            "stream": False,
        }

        if image_bytes:
            payload["images"] = [base64.b64encode(image_bytes).decode("utf-8")]

        try:
            req = urllib.request.Request(
                f"{self._config.base_url}/api/generate",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode())

            corrected = result.get("response", ocr_text).strip()
            return {
                "corrected_text": corrected,
                "confidence": 0.8,
                "model": self._config.model,
            }

        except Exception as e:
            logger.error(f"Ollama correction failed: {e}")
            return {"corrected_text": ocr_text, "confidence": 0.0, "error": str(e)}

    def is_available(self) -> bool:
        return self._available
