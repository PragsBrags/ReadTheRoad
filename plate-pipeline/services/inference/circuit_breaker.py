"""
Circuit Breaker — Fault tolerance for LLM calls.

States:
  - CLOSED: Normal operation, requests go through
  - OPEN: Too many failures, requests are blocked
  - HALF_OPEN: Testing if service recovered

Tracks failure rate and auto-disables LLM if unstable.
Falls back to OCR-only pipeline on trip.
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """
    Circuit breaker for LLM calls.

    Tracks failure rate and auto-disables LLM if unstable.
    Falls back to OCR-only pipeline on trip.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 2,
    ):
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._half_open_max_calls = half_open_max_calls

        self._state = self.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0

    @property
    def state(self) -> str:
        if self._state == self.OPEN:
            if time.time() - self._last_failure_time >= self._recovery_timeout:
                self._state = self.HALF_OPEN
                self._half_open_calls = 0
                logger.info("Circuit breaker → HALF_OPEN")
        return self._state

    def allow_request(self) -> bool:
        """Check if a request should be allowed through."""
        state = self.state
        if state == self.CLOSED:
            return True
        elif state == self.HALF_OPEN:
            return self._half_open_calls < self._half_open_max_calls
        return False  # OPEN

    def record_success(self) -> None:
        """Record a successful call."""
        if self.state == self.HALF_OPEN:
            self._half_open_calls += 1
            if self._half_open_calls >= self._half_open_max_calls:
                self._state = self.CLOSED
                self._failure_count = 0
                logger.info("Circuit breaker → CLOSED (recovered)")
        else:
            self._failure_count = max(0, self._failure_count - 1)

    def record_failure(self) -> None:
        """Record a failed call."""
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._failure_count >= self._failure_threshold:
            self._state = self.OPEN
            logger.warning(
                f"Circuit breaker → OPEN (failures={self._failure_count})"
            )
