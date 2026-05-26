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

# 1. Start MediaMTX container if not already running
if [ ! "$(docker ps -q -f name=mediamtx)" ]; then
    if [ "$(docker ps -aq -f status=exited -f name=mediamtx)" ]; then
        echo "Starting existing mediamtx container..."
        docker start mediamtx
    else
        echo "Launching mediamtx RTSP server on port 8554..."
        docker run -d --name mediamtx -p 8554:8554 -p 1935:1935 -p 8888:8888 bluenviron/mediamtx
    fi
else
    echo "MediaMTX RTSP server is already running."
fi

# Wait for MediaMTX to start up
sleep 2

# 2. Iterate through videos in the directory
echo "Scanning for videos in $VIDEO_DIR..."
# Supported extensions
shopt -s nullglob
VIDEOS=("$VIDEO_DIR"/*.mp4 "$VIDEO_DIR"/*.mov "$VIDEO_DIR"/*.mkv "$VIDEO_DIR"/*.avi "$VIDEO_DIR"/*.webm)

if [ ${#VIDEOS[@]} -eq 0 ]; then
    echo "No videos found in directory: $VIDEO_DIR"
    exit 0
fi

# Track spawned process PIDs so they can be cleaned up
declare -a PIDS

# Cleanup background streams on script exit
cleanup() {
    echo ""
    echo "Stopping all RTSP streams..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

for video_path in "${VIDEOS[@]}"; do
    filename=$(basename -- "$video_path")
    # Clean up name for URL path (replace spaces/special chars)
    stream_path="${filename%.*}"
    stream_path="${stream_path// /_}" # replace spaces with underscores
    
    rtsp_url="rtsp://localhost:8554/$stream_path"
    
    echo "----------------------------------------"
    echo "Streaming: $filename"
    echo "RTSP URL : $rtsp_url"
    
    # Launch ffmpeg stream in background looping infinitely
    ffmpeg -re -stream_loop -1 -i "$video_path" -c copy -f rtsp "$rtsp_url" &>/dev/null &
    ffmpeg_pid=$!
    PIDS+=("$ffmpeg_pid")
    
    # 3. Optionally register the stream with the running pipeline
    if [ -n "$PIPELINE_URL" ]; then
        echo "Registering stream with pipeline at $PIPELINE_URL..."
        
        # Determine URL for the pipeline depending on whether it is running in docker or local
        pipeline_stream_url="rtsp://localhost:8554/$stream_path"
        
        # Check if pipeline URL responds
        if curl -s -f "$PIPELINE_URL/status" &>/dev/null; then
            curl -s -X POST "$PIPELINE_URL/ingest/stream" \
                 -H "Content-Type: application/json" \
                 -d "{
                   \"url\": \"$pipeline_stream_url\",
                   \"stream_id\": \"$stream_path\",
                   \"continuous\": true
                 }" | jq '.' || echo "  (Registered stream: $stream_path)"
        else
            echo "  WARNING: Pipeline API at $PIPELINE_URL is unreachable. Skipping automatic ingestion registration."
        fi
    fi
done

echo "----------------------------------------"
echo "All streams started successfully! Press Ctrl+C to stop all streams."
echo ""

# Keep script running to maintain background streams
while true; do
    sleep 1
done
