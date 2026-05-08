# License Plate Detection

## Architecture

```
VIDEO INPUT (RTSP / RTMP / FILE)
        ↓
INGESTION SERVICE (FastAPI)
        ↓
FFmpeg DECODER
        ↓
FRAME SAMPLER
        ↓
RABBITMQ QUEUE
        ↓
CELERY WORKERS
        ↓
INFERENCE ROUTER
        ↓
┌─────────────────────────────┐
│  YOLO (Plate Detection)     │
│  OCR (EasyOCR / PaddleOCR)  │
│  LLM (Optional / Gated)     │
└─────────────────────────────┘
        ↓
AGGREGATION + DEDUP
        ↓
REDIS (Cache + State)
        ↓
PROMETHEUS + GRAFANA
```

## Inference Modes

All modes are **config-driven** — no code changes required:

| Mode           | Pipeline         | Description                           |
| -------------- | ---------------- | ------------------------------------- |
| `yolo_only`    | YOLO             | Detect bounding boxes only            |
| `ocr_only`     | OCR              | Text extraction on pre-cropped images |
| `yolo_ocr`     | YOLO → OCR       | Detect → crop → read text (default)   |
| `yolo_ocr_llm` | YOLO → OCR → LLM | Hybrid with LLM correction            |
| `direct_llm`   | YOLO → LLM       | Direct LLM vision inference           |

## Quick Start

### Local Development

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Validate config
python -c "from services.config_loader import load_config; print(load_config())"

# Run API server
python main.py
# or
uvicorn main:app --reload --port 8000
```

### Docker (Full Stack)

```bash
cd docker

# Build and start all services
docker-compose up --build

# Scale workers
docker-compose up --scale celery-worker=5
```

### Access Points

| Service    | URL                                  |
| ---------- | ------------------------------------ |
| API        | http://localhost:8000                |
| API Docs   | http://localhost:8000/docs           |
| Prometheus | http://localhost:9090                |
| Grafana    | http://localhost:3000 (admin/admin)  |
| RabbitMQ   | http://localhost:15672 (guest/guest) |

## API Endpoints

| Method   | Path                  | Description                        |
| -------- | --------------------- | ---------------------------------- |
| `POST`   | `/ingest/file`        | Upload a video file for processing |
| `POST`   | `/ingest/stream`      | Start RTSP/RTMP stream ingestion   |
| `GET`    | `/ingest/streams`     | List active streams                |
| `DELETE` | `/ingest/stream/{id}` | Stop a stream                      |
| `GET`    | `/results/{job_id}`   | Get detection results              |
| `GET`    | `/status`             | System health check                |
| `GET`    | `/metrics`            | Prometheus metrics                 |

## Configuration

All behavior is controlled via `config/config.yaml`. **No runtime overrides.**

Key sections:

```yaml
inference:
  mode: yolo_ocr # Change inference mode here

ocr:
  engine: easyocr # or paddleocr

model_registry:
  yolo:
    path: /models/plate.pt # Your YOLO weights

llm:
  enabled: true
  provider: openai
  mode: gated # gated | always | disabled
```

## YOLO Model

Supply your own license plate detection model:

1. Place your `.pt` file in the `models/` directory
2. Update `model_registry.yolo.path` in `config/config.yaml`
3. Supported models: YOLOv8, YOLOv11 (Ultralytics format)

## Project Structure

```
plate-pipeline/
├── config/config.yaml          # Single source of truth
├── services/
│   ├── config_loader.py        # Pydantic config validation
│   ├── decoder.py              # FFmpeg wrapper (mandatory)
│   ├── sampler.py              # Frame sampling strategies
│   ├── ingestion.py            # Video ingestion + API
│   ├── queue.py                # RabbitMQ publisher
│   ├── worker.py               # Celery task definitions
│   ├── models.py               # Model factory + registry
│   ├── inference_router.py     # Inference mode routing
│   ├── aggregation.py          # Multi-frame deduplication
│   ├── cache.py                # Redis cache service
│   └── monitoring.py           # Prometheus metrics
├── dspy_module/
│   └── optimizer.py            # DSPy prompt optimization
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.worker
│   ├── docker-compose.yml
│   └── prometheus.yml
├── tests/
├── main.py                     # FastAPI entrypoint
├── requirements.txt
└── pyproject.toml
```
