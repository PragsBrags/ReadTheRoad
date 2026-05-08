# services — Core service packages for the plate detection pipeline
#
# Packages:
#   config/      — Type-safe configuration (enums, models, loader)
#   models/      — ML model interfaces, implementations, factory, registry
#   inference/   — Inference pipeline stages, router, circuit breaker
#   api/         — FastAPI route handlers
#
# Modules:
#   decoder      — FFmpeg video decoder
#   sampler      — Frame sampling strategies
#   ingestion    — Video ingestion orchestration
#   cache        — Redis cache-aside service
#   aggregation  — Multi-frame result merging
#   monitoring   — Prometheus metrics
#   queue        — RabbitMQ publisher/consumer
#   task_dispatcher — Local/distributed dispatch
#   worker       — Celery worker tasks
#   local_processor — Direct inference (no queue)
