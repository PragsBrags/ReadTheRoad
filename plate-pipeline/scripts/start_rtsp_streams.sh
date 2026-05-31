#!/usr/bin/env bash
# Start RTSP streams for all videos in a folder using MediaMTX and FFmpeg.
#
# Usage:
#   ./scripts/start_rtsp_streams.sh <video_folder> [register_with_pipeline_url]
#
# Example:
#   ./scripts/start_rtsp_streams.sh ./samples http://localhost:8000
set -euo pipefail

VIDEO_DIR="${1:-./samples}"
PIPELINE_URL="${2:-}"

# Check dependencies
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is required to run the RTSP server."
    exit 1
fi
if ! command -v ffmpeg &> /dev/null; then
    echo "ERROR: FFmpeg is required to stream the video files."
    exit 1
fi

# 1. Write a minimal mediamtx config with no auth and no timeouts
MEDIAMTX_CONFIG="/tmp/mediamtx.yml"
cat > "$MEDIAMTX_CONFIG" <<'EOF'
authMethod: internal

authInternalUsers:
  - user: any
    pass:
    permissions:
      - action: publish
      - action: read
      - action: api
      - action: metrics

paths:
  all:

readTimeout: 24h
writeTimeout: 24h
writeQueueSize: 512

api: yes
apiAddress: :9997

EOF

# 2. Remove any existing mediamtx container so fresh config always applies
if [ "$(docker ps -aq -f name=mediamtx)" ]; then
    echo "Removing existing mediamtx container to apply fresh config..."
    docker rm -f mediamtx
fi

# 3. Start MediaMTX with the config mounted
echo "Launching mediamtx RTSP server on port 8554 (no auth, no timeouts)..."
docker run -d --name mediamtx \
    -p 8554:8554 \
    -p 1935:1935 \
    -p 8888:8888 \
    -p 9997:9997 \
    -v "$MEDIAMTX_CONFIG:/mediamtx.yml" \
    bluenviron/mediamtx

# 4. Wait until MediaMTX API responds with 200
echo "Waiting for MediaMTX to be ready..."
for i in $(seq 1 30); do
    http_code=$(curl -s -o /dev/null -w "%{http_code}" \
        "http://localhost:9997/v3/config/global/get")
    if [ "$http_code" = "200" ]; then
        echo "MediaMTX is ready (${i}s)."
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "ERROR: MediaMTX did not become ready in 30 seconds."
        echo "Check: docker logs mediamtx"
        exit 1
    fi
    sleep 1
done

# 5. Scan for videos
echo "Scanning for videos in $VIDEO_DIR..."

shopt -s nullglob
VIDEOS=("$VIDEO_DIR"/*.mp4 "$VIDEO_DIR"/*.mov "$VIDEO_DIR"/*.mkv "$VIDEO_DIR"/*.avi "$VIDEO_DIR"/*.webm)

if [ ${#VIDEOS[@]} -eq 0 ]; then
    echo "No videos found in directory: $VIDEO_DIR"
    exit 0
fi

echo "Found ${#VIDEOS[@]} video(s)."

# 6. Cleanup handler
declare -a PIDS

cleanup() {
    echo ""
    echo "Stopping all RTSP streams..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    rm -f "$MEDIAMTX_CONFIG"
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# 7. Start one FFmpeg stream per video (with auto-restart loop)
for video_path in "${VIDEOS[@]}"; do
    filename=$(basename -- "$video_path")
    stream_path="${filename%.*}"
    stream_path="${stream_path// /_}"

    rtsp_url="rtsp://localhost:8554/$stream_path"
    log_file="/tmp/ffmpeg_${stream_path}.log"

    echo "----------------------------------------"
    echo "Streaming : $filename"
    echo "RTSP URL  : $rtsp_url"
    echo "Log       : $log_file"

    (
        while true; do
            echo "[$(date '+%H:%M:%S')] Starting stream: $rtsp_url" >> "$log_file"
            ffmpeg -re \
                   -stream_loop -1 \
                   -i "$video_path" \
                   -c copy \
                   -avoid_negative_ts make_zero \
                   -fflags +genpts \
                   -rtsp_transport tcp \
                   -f rtsp \
                   "$rtsp_url" \
                   >> "$log_file" 2>&1 && true
            exit_code=$?
            echo "[$(date '+%H:%M:%S')] Stream exited (code $exit_code), restarting in 2s..." >> "$log_file"
            sleep 2
        done
    ) &

    loop_pid=$!
    PIDS+=("$loop_pid")
    echo "PID       : $loop_pid"

    sleep 1

    # 8. Optionally register the stream with the running pipeline
    if [ -n "$PIPELINE_URL" ]; then
        echo "Registering with pipeline at $PIPELINE_URL..."

        if curl -sf "$PIPELINE_URL/status" &>/dev/null; then
            curl -s -X POST "$PIPELINE_URL/ingest/stream" \
                 -H "Content-Type: application/json" \
                 -d "{
                   \"url\": \"$rtsp_url\",
                   \"stream_id\": \"$stream_path\",
                   \"continuous\": true
                 }" | jq '.' || echo "  (Registered stream: $stream_path)"
        else
            echo "  WARNING: Pipeline at $PIPELINE_URL unreachable. Skipping registration."
        fi
    fi
done

echo "----------------------------------------"
echo "All streams running. Press Ctrl+C to stop."
echo ""
echo "Connect with:"
for video_path in "${VIDEOS[@]}"; do
    filename=$(basename -- "$video_path")
    stream_path="${filename%.*}"
    stream_path="${stream_path// /_}"
    echo "  rtsp://localhost:8554/$stream_path"
done
echo ""
echo "Monitor logs:"
echo "  tail -f /tmp/ffmpeg_*.log"
echo ""

# 9. Keep alive
while true; do
    sleep 10
done