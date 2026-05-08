#!/usr/bin/env bash
# DOCKER DEPLOYMENT — Full distributed stack
#
# Starts all services:
#   - ingestion-api    (FastAPI)
#   - celery-worker    (inference workers, scalable)
#   - rabbitmq         (message broker)
#   - redis            (cache + state)
#   - prometheus       (metrics)
#   - grafana          (dashboards)
#
# Prerequisites:
#   - Docker and Docker Compose installed
#   - YOLO model at ./models/plate.pt (optional, mounts via volume)
#
# Usage:
#   ./scripts/deploy.sh                    # Build and start
#   ./scripts/deploy.sh --scale workers=4  # Scale workers
#   ./scripts/deploy.sh down               # Stop everything
#   ./scripts/deploy.sh logs               # Tail logs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DOCKER_DIR="$PROJECT_DIR/docker"

cd "$DOCKER_DIR"

ACTION="${1:-up}"

echo "============================================"
echo "  PLATE PIPELINE — DOCKER DEPLOYMENT"
echo "============================================"
echo ""

case "$ACTION" in
    up|start)
        echo "  Starting full stack..."
        echo ""
        echo "  Services:"
        echo "    API          : http://localhost:8000"
        echo "    API Docs     : http://localhost:8000/docs"
        echo "    RabbitMQ     : http://localhost:15672 (guest/guest)"
        echo "    Prometheus   : http://localhost:9090"
        echo "    Grafana      : http://localhost:3000 (admin/admin)"
        echo ""

        # Check YOLO model
        if [ ! -f "$PROJECT_DIR/models/plate.pt" ]; then
            echo "WARNING: No YOLO model at ./models/plate.pt"
            echo "         Mount your model or copy it into the models volume."
            echo ""
        fi

        shift || true
        docker compose up --build "$@"
        ;;

    scale)
        # Usage: ./scripts/deploy.sh scale 4
        WORKERS="${2:-2}"
        echo "  Scaling workers to $WORKERS..."
        docker compose up --scale celery-worker="$WORKERS" -d
        ;;

    down|stop)
        echo "  Stopping all services..."
        docker compose down
        ;;

    restart)
        echo "  Restarting..."
        docker compose down
        docker compose up --build -d
        ;;

    logs)
        shift || true
        docker compose logs -f "$@"
        ;;

    status)
        docker compose ps
        echo ""
        echo "API health:"
        curl -s http://localhost:8000/status | python3 -m json.tool 2>/dev/null || echo "  API not reachable"
        ;;

    *)
        echo "Usage: $0 {up|down|restart|scale|logs|status}"
        echo ""
        echo "  up [--build]           Build and start all services"
        echo "  down                   Stop all services"
        echo "  restart                Rebuild and restart"
        echo "  scale N                Scale to N Celery workers"
        echo "  logs [service]         Tail logs"
        echo "  status                 Show service status"
        exit 1
        ;;
esac
