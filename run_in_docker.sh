#!/bin/bash

# Script to build and run the code analyzer Docker container

# Get the directory containing this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Docker image name
IMAGE_NAME="code-analyzer"
CONTAINER_NAME="code-analyzer-container"

# Default model
#DEFAULT_MODEL="Qwen/Qwen2.5-Coder-0.5B-Instruct-AWQ"
DEFAULT_MODEL="Qwen/Qwen2.5-Coder-14B-Instruct-AWQ"

# Create persistent directories on host
mkdir -p "$SCRIPT_DIR/data/cache"      # For SQLite database
mkdir -p "$SCRIPT_DIR/data/logs"       # For log files
mkdir -p "$SCRIPT_DIR/data/hf_cache"   # For HuggingFace cache
mkdir -p "$SCRIPT_DIR/data/llm_out"    # For LLM query exports
mkdir -p "$SCRIPT_DIR/data/debug"      # For debug files

# Parse command line arguments
TARGET_PATH=""
MODEL="$DEFAULT_MODEL"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        *)
            if [ -z "$TARGET_PATH" ]; then
                TARGET_PATH="$1"
            else
                echo "Error: Unexpected argument: $1"
                echo "Usage: $0 [--model model_name] <target_directory>"
                exit 1
            fi
            shift
            ;;
    esac
done

# Check if Docker image exists
if [[ "$(docker images -q $IMAGE_NAME 2> /dev/null)" == "" ]]; then
    echo "Building Docker image..."
    docker build -t $IMAGE_NAME \
        --build-arg HOST_UID=$(id -u) \
        --build-arg HOST_GID=$(id -g) \
        -f Dockerfile .
fi

# Stop and remove existing container if it exists
docker stop $CONTAINER_NAME 2>/dev/null
docker rm $CONTAINER_NAME 2>/dev/null

# If a target directory is provided
if [ -n "$TARGET_PATH" ]; then
    TARGET_PATH=$(realpath "$TARGET_PATH")
    echo "Target directory: $TARGET_PATH"
    echo "Using model: $MODEL"
    
    # Base docker run command with shared volumes
    DOCKER_CMD="docker run --gpus all -it --rm \
        --name $CONTAINER_NAME \
        -v $SCRIPT_DIR/data/hf_cache:/home/mluser/.cache/huggingface \
        -v $SCRIPT_DIR/data/cache:/home/mluser/.local/share/code_analyzer/cache \
        -v $SCRIPT_DIR/data/logs:/home/mluser/.local/share/code_analyzer/logs \
        -v $SCRIPT_DIR/data/llm_out:/home/mluser/.local/share/code_analyzer/llm_out \
        -v $SCRIPT_DIR/data/debug:/home/mluser/.local/share/code_analyzer/debug \
        -v $SCRIPT_DIR:/home/mluser/code_analyzer \
        -v $TARGET_PATH:/home/mluser/target \
        -e CODE_ANALYZER_CACHE_DIR=/home/mluser/.local/share/code_analyzer/cache \
        -e CODE_ANALYZER_LOG_DIR=/home/mluser/.local/share/code_analyzer/logs \
        -e CODE_ANALYZER_LLM_DIR=/home/mluser/.local/share/code_analyzer/llm_out \
        -e CODE_ANALYZER_DEBUG_DIR=/home/mluser/.local/share/code_analyzer/debug"
    
    echo "Running analysis on mounted directory..."
    $DOCKER_CMD $IMAGE_NAME python3 -m code_analyzer --export --verbose /home/mluser/target --model "$MODEL"
else
    echo "Error: Please provide a target directory to analyze"
    echo "Usage: $0 [--model model_name] <target_directory>"
    exit 1
fi

# Check exit status
if [ $? -ne 0 ]; then
    echo "Error: Docker container exited with non-zero status"
    exit 1
fi