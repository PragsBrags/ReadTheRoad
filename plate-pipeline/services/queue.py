"""
Queue Service — RabbitMQ publisher for frame distribution.

All frame payloads are serialized and published to RabbitMQ queues.
Workers consume from these queues for stateless inference.

Pattern: Publisher → RabbitMQ → Consumer (Celery workers)
"""

from __future__ import annotations

import base64
import json
import logging
import time
import uuid
from typing import Any

from services.config import QueueConfig
from services.decoder import FrameData

logger = logging.getLogger(__name__)


# FRAME SERIALIZATION

def serialize_frame(frame: FrameData, job_id: str = "") -> dict[str, Any]:
    """
    Serialize a FrameData object for queue transport.

    Frame bytes are base64 encoded for JSON serialization.
    """
    return {
        "frame_id": frame.frame_id,
        "job_id": job_id,
        "source": frame.source,
        "frame_b64": base64.b64encode(frame.frame_bytes).decode("utf-8"),
        "frame_index": frame.frame_index,
        "timestamp_ms": frame.timestamp_ms,
        "content_hash": frame.content_hash,
        "extraction_time_ms": frame.extraction_time_ms,
        "metadata": frame.metadata,
        "published_at": time.time(),
    }


def deserialize_frame(data: dict[str, Any]) -> FrameData:
    """Deserialize a frame payload from the queue."""
    return FrameData(
        frame_id=data["frame_id"],
        source=data["source"],
        frame_bytes=base64.b64decode(data["frame_b64"]),
        frame_index=data["frame_index"],
        timestamp_ms=data.get("timestamp_ms", 0.0),
        content_hash=data.get("content_hash", ""),
        extraction_time_ms=data.get("extraction_time_ms", 0.0),
        metadata=data.get("metadata", {}),
    )


# QUEUE PUBLISHER (RabbitMQ via aio-pika)

class QueuePublisher:
    """
    RabbitMQ publisher using aio-pika.

    Publishes serialized frame payloads to the configured queue.
    Includes connection pooling and retry logic.
    """

    def __init__(self, config: QueueConfig):
        self._config = config
        self._connection = None
        self._channel = None
        self._exchange = None
        self._connected = False

    async def connect(self) -> None:
        """Establish connection to RabbitMQ."""
        try:
            import aio_pika

            self._connection = await aio_pika.connect_robust(
                self._config.broker_url,
            )
            self._channel = await self._connection.channel()
            await self._channel.set_qos(prefetch_count=self._config.prefetch_count)

            # Declare queues
            self._frame_queue = await self._channel.declare_queue(
                self._config.frame_queue,
                durable=True,
            )
            self._result_queue = await self._channel.declare_queue(
                self._config.result_queue,
                durable=True,
            )

            self._connected = True
            logger.info(f"Connected to RabbitMQ at {self._config.broker_url}")

        except ImportError:
            logger.warning(
                "aio-pika not installed. Queue publishing disabled. "
                "Install with: pip install aio-pika"
            )
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            self._connected = False

    async def publish_frame(
        self,
        frame: FrameData,
        job_id: str = "",
        retry_count: int = 0,
    ) -> bool:
        """
        Publish a frame payload to the frame queue.

        Args:
            frame: FrameData to publish.
            job_id: Associated job ID.
            retry_count: Current retry attempt.

        Returns:
            True if published successfully.
        """
        if not self._connected:
            logger.warning("Queue not connected. Attempting reconnect...")
            await self.connect()
            if not self._connected:
                logger.error("Cannot publish: queue connection failed")
                return False

        try:
            import aio_pika

            payload = serialize_frame(frame, job_id)
            message = aio_pika.Message(
                body=json.dumps(payload).encode("utf-8"),
                content_type="application/json",
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                message_id=frame.frame_id,
                headers={"job_id": job_id, "frame_index": frame.frame_index},
            )

            await self._channel.default_exchange.publish(
                message,
                routing_key=self._config.frame_queue,
            )

            logger.debug(
                f"Published frame {frame.frame_id} to {self._config.frame_queue}"
            )
            return True

        except Exception as e:
            if retry_count < self._config.max_retries:
                logger.warning(
                    f"Publish failed (attempt {retry_count + 1}/"
                    f"{self._config.max_retries}): {e}"
                )
                import asyncio
                await asyncio.sleep(self._config.retry_delay_seconds)
                return await self.publish_frame(frame, job_id, retry_count + 1)

            logger.error(f"Publish failed after {self._config.max_retries} retries: {e}")
            return False

    async def publish_result(self, result: dict[str, Any]) -> bool:
        """Publish an inference result to the result queue."""
        if not self._connected:
            return False

        try:
            import aio_pika

            message = aio_pika.Message(
                body=json.dumps(result).encode("utf-8"),
                content_type="application/json",
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            )

            await self._channel.default_exchange.publish(
                message,
                routing_key=self._config.result_queue,
            )
            return True

        except Exception as e:
            logger.error(f"Failed to publish result: {e}")
            return False

    async def close(self) -> None:
        """Close the RabbitMQ connection."""
        if self._connection:
            await self._connection.close()
            self._connected = False
            logger.info("RabbitMQ connection closed")

    @property
    def is_connected(self) -> bool:
        return self._connected


# QUEUE CONSUMER (for non-Celery usage)

class QueueConsumer:
    """
    RabbitMQ consumer for direct consumption (non-Celery).

    Used for testing or alternative worker implementations.
    """

    def __init__(self, config: QueueConfig):
        self._config = config
        self._connection = None
        self._channel = None

    async def connect(self) -> None:
        """Connect to RabbitMQ for consuming."""
        try:
            import aio_pika

            self._connection = await aio_pika.connect_robust(
                self._config.broker_url,
            )
            self._channel = await self._connection.channel()
            await self._channel.set_qos(prefetch_count=self._config.prefetch_count)
            logger.info("Queue consumer connected")

        except Exception as e:
            logger.error(f"Consumer connection failed: {e}")

    async def consume(self, callback):
        """
        Start consuming frames from the queue.

        Args:
            callback: Async function called for each frame message.
                      Signature: async def callback(frame_data: dict) -> None
        """
  

        queue = await self._channel.declare_queue(
            self._config.frame_queue,
            durable=True,
        )

        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                async with message.process():
                    try:
                        data = json.loads(message.body.decode("utf-8"))
                        await callback(data)
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")

    async def close(self) -> None:
        """Close consumer connection."""
        if self._connection:
            await self._connection.close()
