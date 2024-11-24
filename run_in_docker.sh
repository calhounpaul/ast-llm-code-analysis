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
mkdir -p "$SCRIPT_DIR/data/cache" # For SQLite database
mkdir -p "$SCRIPT_DIR/data/logs" # For log files
mkdir -p "$SCRIPT_DIR/data/hf_cache" # For HuggingFace cache
mkdir -p "$SCRIPT_DIR/data/llm_out" # For LLM query exports
mkdir -p "$SCRIPT_DIR/data/debug" # For debug files
mkdir -p "$SCRIPT_DIR/data/coverage" # For coverage reports

# Parse command line arguments
TARGET_PATH=""
MODEL="$DEFAULT_MODEL"
RUN_TESTS=false
TEST_PATH=""
PYTEST_ARGS=""

print_usage() {
    echo "Usage: $0 [--model model_name] [--test [test_path]] [--pytest-args 'args'] [target_directory]"
    echo "Options:"
    echo " --model name        Specify the model to use"
    echo " --test [path]      Run the test suite. Optionally specify test path/file"
    echo " --pytest-args      Additional arguments to pass to pytest (in quotes)"
    echo " target_directory   Directory to analyze (not needed with --test)"
    echo
    echo "Examples:"
    echo " $0 --test                          # Run all tests"
    echo " $0 --test tests/unit               # Run unit tests only"
    echo " $0 --test tests/test_parser.py     # Run specific test file"
    echo " $0 --test --pytest-args '--cov=code_analyzer/ --cov-report=html'  # Run with coverage"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --test)
            RUN_TESTS=true
            if [[ -n "$2" ]] && [[ "$2" != --* ]]; then
                TEST_PATH="$2"
                shift 2
            else
                TEST_PATH="tests/"
                shift
            fi
            ;;
        --pytest-args)
            PYTEST_ARGS="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            if [ -z "$TARGET_PATH" ]; then
                TARGET_PATH="$1"
            else
                echo "Error: Unexpected argument: $1"
                print_usage
                exit 1
            fi
            shift
            ;;
    esac
done

# Validate arguments
if [ "$RUN_TESTS" = false ] && [ -z "$TARGET_PATH" ]; then
    echo "Error: Please provide a target directory to analyze or use --test"
    print_usage
    exit 1
fi

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

# Base docker run command with shared volumes
DOCKER_CMD="docker run --gpus all -it --rm \
    --name $CONTAINER_NAME \
    -v $SCRIPT_DIR/data/hf_cache:/home/mluser/.cache/huggingface \
    -v $SCRIPT_DIR/data/cache:/home/mluser/.local/share/code_analyzer/cache \
    -v $SCRIPT_DIR/data/logs:/home/mluser/.local/share/code_analyzer/logs \
    -v $SCRIPT_DIR/data/llm_out:/home/mluser/.local/share/code_analyzer/llm_out \
    -v $SCRIPT_DIR/data/debug:/home/mluser/.local/share/code_analyzer/debug \
    -v $SCRIPT_DIR/data/coverage:/home/mluser/code_analyzer/htmlcov \
    -v $SCRIPT_DIR:/home/mluser/code_analyzer \
    -e CODE_ANALYZER_CACHE_DIR=/home/mluser/.local/share/code_analyzer/cache \
    -e CODE_ANALYZER_LOG_DIR=/home/mluser/.local/share/code_analyzer/logs \
    -e CODE_ANALYZER_LLM_DIR=/home/mluser/.local/share/code_analyzer/llm_out \
    -e CODE_ANALYZER_DEBUG_DIR=/home/mluser/.local/share/code_analyzer/debug"

if [ "$RUN_TESTS" = true ]; then
    echo "Running tests..."
    if [ -n "$PYTEST_ARGS" ]; then
        echo "Using pytest arguments: $PYTEST_ARGS"
    fi
    $DOCKER_CMD $IMAGE_NAME bash -c "cd /home/mluser/code_analyzer && \
        pip install pytest-cov && \
        pytest $TEST_PATH $PYTEST_ARGS"
else
    # Add target directory volume mount for analysis mode
    TARGET_PATH=$(realpath "$TARGET_PATH")
    echo "Target directory: $TARGET_PATH"
    echo "Using model: $MODEL"
    DOCKER_CMD="$DOCKER_CMD -v $TARGET_PATH:/home/mluser/target"
    echo "Running analysis on mounted directory..."
    $DOCKER_CMD $IMAGE_NAME python3 -m code_analyzer --export --verbose /home/mluser/target --model "$MODEL"
fi

# Check exit status
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    if [ "$RUN_TESTS" = true ]; then
        echo "Error: Tests failed with exit code $EXIT_CODE"
    else
        echo "Error: Analysis failed with exit code $EXIT_CODE"
    fi
    exit $EXIT_CODE
fi

exit 0