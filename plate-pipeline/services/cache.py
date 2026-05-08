"""
Cache Service — Redis cache-aside pattern for deduplication and state.

Implements:
  - Cache-aside: cache → compute → store → return
  - Deduplication: hash frame content, skip if recently processed
  - State tracking: stream status, worker health, job results

All cache behavior is controlled via config.yaml.
"""

from __future__ import annotations


import json
import logging
import time
from typing import Any, Optional

from services.config import RedisConfig

logger = logging.getLogger(__name__)


# CACHE SERVICE

class CacheService:
    """
    Redis-based caching service.

    Provides:
      - Result caching (cache-aside pattern)
      - Frame deduplication
      - State management (streams, jobs)
      - TTL-based expiration
    """

    def __init__(self, config: RedisConfig):
        self._config = config
        self._client = None
        self._connected = False
        self._prefix = config.key_prefix

        if config.enabled:
            self._connect()

    def _connect(self) -> None:
        """Establish Redis connection."""
        try:
            import redis
            self._client = redis.Redis.from_url(
                self._config.url,
                decode_responses=False,  # We handle encoding ourselves
            )
            self._client.ping()
            self._connected = True
            logger.info(f"Redis connected at {self._config.url}")
        except ImportError:
            logger.warning("redis package not installed. Caching disabled.")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Caching disabled.")

    def _key(self, *parts: str) -> str:
        """Build a namespaced cache key."""
        return ":".join([self._prefix] + list(parts))

    # RESULT CACHING 

    def get_result(self, content_hash: str) -> Optional[dict[str, Any]]:
        """
        Get a cached inference result by frame content hash.

        Returns None on cache miss.
        """
        if not self._connected:
            return None

        try:
            key = self._key("result", content_hash)
            data = self._client.get(key)
            if data:
                logger.debug(f"Cache HIT: {content_hash[:8]}")
                return json.loads(data)
            return None
        except Exception as e:
            logger.warning(f"Cache get failed: {e}")
            return None

    def store_result(
        self,
        content_hash: str,
        result: dict[str, Any],
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Store an inference result in cache.

        Args:
            content_hash: Frame content hash (key).
            result: Inference result to cache.
            ttl: Optional TTL override (seconds).
        """
        if not self._connected:
            return False

        try:
            key = self._key("result", content_hash)
            data = json.dumps(result, default=str)
            self._client.setex(
                key,
                ttl or self._config.ttl_seconds,
                data,
            )
            logger.debug(f"Cache STORE: {content_hash[:8]}")
            return True
        except Exception as e:
            logger.warning(f"Cache store failed: {e}")
            return False

    def get_or_compute(
        self,
        key: str,
        compute_fn,
        ttl: Optional[int] = None,
    ) -> Any:
        """
        Cache-aside pattern: check cache → compute if miss → store.

        Args:
            key: Cache key.
            compute_fn: Callable that produces the value on cache miss.
            ttl: Optional TTL override.

        Returns:
            Cached or computed value.
        """
        cached = self.get_result(key)
        if cached is not None:
            return cached

        result = compute_fn()
        self.store_result(key, result, ttl)
        return result

    # DEDUPLICATION

    def is_duplicate(self, content_hash: str) -> bool:
        """
        Check if a frame has been recently processed (dedup).

        Returns True if the frame is a duplicate.
        """
        if not self._connected or not self._config.deduplication:
            return False

        try:
            key = self._key("dedup", content_hash)
            exists = self._client.exists(key)
            return bool(exists)
        except Exception as e:
            logger.warning(f"Dedup check failed: {e}")
            return False

    def mark_processed(self, content_hash: str) -> None:
        """Mark a frame as processed for deduplication."""
        if not self._connected or not self._config.deduplication:
            return

        try:
            key = self._key("dedup", content_hash)
            self._client.setex(
                key,
                self._config.dedup_ttl_seconds,
                b"1",
            )
        except Exception as e:
            logger.warning(f"Dedup mark failed: {e}")

    # STATE MANAGEMENT

    def set_stream_status(
        self,
        stream_id: str,
        status: dict[str, Any],
    ) -> None:
        """Store stream status."""
        if not self._connected:
            return

        try:
            key = self._key("stream", stream_id)
            self._client.setex(
                key,
                self._config.ttl_seconds,
                json.dumps(status, default=str),
            )
        except Exception as e:
            logger.warning(f"Stream status store failed: {e}")

    def get_stream_status(self, stream_id: str) -> Optional[dict[str, Any]]:
        """Get stream status."""
        if not self._connected:
            return None

        try:
            key = self._key("stream", stream_id)
            data = self._client.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            logger.warning(f"Stream status get failed: {e}")
            return None

    def store_job_results(
        self,
        job_id: str,
        results: list[dict[str, Any]],
    ) -> None:
        """Store aggregated job results (overwrites existing)."""
        if not self._connected:
            return

        try:
            key = self._key("job", job_id, "results")
            self._client.setex(
                key,
                self._config.ttl_seconds * 2,  # Longer TTL for results
                json.dumps(results, default=str),
            )
        except Exception as e:
            logger.warning(f"Job results store failed: {e}")

    def append_job_plates(
        self,
        job_id: str,
        plates: list[dict[str, Any]],
    ) -> None:
        """
        Atomically append plates to job results using Redis list (RPUSH).

        This avoids the read-modify-write race condition that occurs
        when multiple workers process frames for the same job_id.
        """
        if not self._connected or not plates:
            return

        try:
            key = self._key("job", job_id, "plates")
            pipe = self._client.pipeline()
            for plate in plates:
                pipe.rpush(key, json.dumps(plate, default=str))
            pipe.expire(key, self._config.ttl_seconds * 2)
            pipe.execute()
        except Exception as e:
            logger.warning(f"Job plates append failed: {e}")

    def get_job_results(self, job_id: str) -> Optional[list[dict[str, Any]]]:
        """Get stored job results (supports both list and key-value formats)."""
        if not self._connected:
            return None

        try:
            # Try atomic list format first (plates appended via RPUSH)
            list_key = self._key("job", job_id, "plates")
            items = self._client.lrange(list_key, 0, -1)
            if items:
                return [json.loads(item) for item in items]

            # Fallback to legacy key-value format
            kv_key = self._key("job", job_id, "results")
            data = self._client.get(kv_key)
            return json.loads(data) if data else None
        except Exception as e:
            logger.warning(f"Job results get failed: {e}")
            return None

    # WORKER HEALTH

    def heartbeat(self, worker_id: str) -> None:
        """Record worker heartbeat."""
        if not self._connected:
            return

        try:
            key = self._key("worker", worker_id, "heartbeat")
            self._client.setex(key, 30, str(time.time()))
        except Exception:
            pass

    def get_active_workers(self) -> list[str]:
        """Get list of active workers."""
        if not self._connected:
            return []

        try:
            pattern = self._key("worker", "*", "heartbeat")
            keys = self._client.keys(pattern)
            # Strip prefix and suffix to extract worker_id
            # Key format: {prefix}:worker:{worker_id}:heartbeat
            prefix = self._key("worker") + ":"
            suffix = ":heartbeat"
            results = []
            for k in keys:
                key_str = k.decode() if isinstance(k, bytes) else k
                if key_str.startswith(prefix) and key_str.endswith(suffix):
                    worker_id = key_str[len(prefix):-len(suffix)]
                    results.append(worker_id)
            return results
        except Exception:
            return []

    # UTILITIES

    def flush_all(self) -> None:
        """Flush all keys with our prefix. USE WITH CAUTION."""
        if not self._connected:
            return

        try:
            pattern = self._key("*")
            keys = self._client.keys(pattern)
            if keys:
                self._client.delete(*keys)
                logger.info(f"Flushed {len(keys)} keys")
        except Exception as e:
            logger.error(f"Flush failed: {e}")

    @property
    def is_connected(self) -> bool:
        return self._connected

    def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            self._client.close()
            self._connected = False
